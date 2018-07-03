""" ``xrview.timeseries.base`` """

from collections import OrderedDict

import numpy as np
import pandas as pd

from bokeh.layouts import row, gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import MultiSelect
from bokeh.plotting import figure
from bokeh.io import output_notebook
from bokeh.io.notebook import show_app
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.events import Reset

from bokeh.document import without_document_lock
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from xrview.utils import is_dataset, is_dataarray, get_notebook_url, iterator

from .handlers import ResamplingDataHandler


class Viewer(object):
    """ Base class for timeseries viewers.

    Parameters
    ----------
    data :

    x :

    overlay :

    figsize : iterable, default (700, 500)
        The size of the figure in pixels.

    resolution : int, default 4
        The number of points to render for each pixel.

    max_workers : int, default 10
        The maximum number of workers in the thread pool to perform the
        sub-sampling.
    """

    def __init__(self, data, x, overlay=None, stack=None, tooltips=None,
                 figsize=(700, 500), ncols=1, resolution=4, max_workers=10,
                 lowpass=False, verbose=0):

        # check data
        if is_dataarray(data):
            if data.name is None:
                self.data = data.to_dataset(name='Data')
            else:
                self.data = data.to_dataset()
        elif is_dataset(data):
            self.data = data
        else:
            raise ValueError('data must be xarray DataArray or Dataset.')

        # check x
        if x in self.data.dims:
            self.x = x
        else:
            raise ValueError(
                x + ' is not a dimension of the provided dataset.')

        # check overlay
        if overlay is None or overlay == 'data_vars' \
                or overlay in self.data.dims:
            self.overlay = overlay
        else:
            raise ValueError(
                overlay + ' is not a dimension of the provided dataset.')

        # check stack
        if stack is None or stack == 'data_vars' or stack in self.data.dims:
            self.stack = stack
        else:
            raise ValueError(
                stack + ' is not a dimension of the provided dataset.')

        # TODO: check tooltips
        self.tooltips = tooltips

        # layout parameters
        self.resolution = resolution
        self.figsize = figsize
        self.ncols = ncols

        # sub-sampling parameters
        self.thread_pool = ThreadPoolExecutor(max_workers)
        self.lowpass = lowpass
        self.verbose = verbose

        #
        self.added_figures = []
        self.added_overlays = []
        self.added_interactions = []

        self.figures = None
        self.handlers = None
        self.selection = []

        self.pending_xrange_update = False
        self.xrange_change_buffer = None

        self.tools = 'pan,wheel_zoom,box_zoom,xbox_select,reset,hover'
        self.colors = ['red', 'green', 'blue']

        self.doc = None

        self.layout = None

    @without_document_lock
    @gen.coroutine
    def reset_xrange(self):
        """ """

        yield self.thread_pool.submit(self.handler.reset_data)
        self.doc.add_next_tick_callback(self.handler.update_source)

    @without_document_lock
    @gen.coroutine
    def update_xrange(self):
        """ Update plot_source when xrange changes. """

        yield self.thread_pool.submit(partial(
            self.handler.update_data, start=self.figures[0].x_range.start,
            end=self.figures[0].x_range.end))
        self.doc.add_next_tick_callback(self.handler.update_source)

        if self.xrange_change_buffer is not None:
            self.doc.add_next_tick_callback(self.update_xrange)
            self.xrange_change_buffer = None

    def on_xrange_change(self, attr, old, new):
        """ Callback for xrange change event. """

        if not self.pending_xrange_update:
            self.pending_xrange_update = True
            self.doc.add_next_tick_callback(self.update_xrange)
        else:
            if self.verbose:
                print('Buffering')
            self.xrange_change_buffer = new

    def on_selected_points_change(self, attr, old, new):
        """ Callback for selection event. """

        idx_new = np.array(new['1d']['indices'])
        self.handler.data.selected = np.zeros(
            len(self.handler.data.selected), dtype=bool)
        sel_idx_start = self.handler.source.data['index'][np.min(idx_new)]
        sel_idx_end = self.handler.source.data['index'][np.max(idx_new)]
        self.handler.data.loc[np.logical_and(
            self.handler.data.index >= sel_idx_start,
            self.handler.data.index <= sel_idx_end), 'selected'] = True

    def on_reset(self, event):
        """ Callback for reset event. """

        self.pending_xrange_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

    def _collect(self, data):
        """ Base method for collect. """

        plot_data = {
            str(o) + '_' + str(s): data[s].sel(**{self.overlay: o}).values
            for o in getattr(self.data, self.overlay)
            for s in getattr(self.data, self.stack)
        }

        plot_data['selected'] = np.zeros(
            data.sizes[self.x], dtype=bool)

        return pd.DataFrame(plot_data, index=data[self.x])

    def collect(self):
        """ Collect plottable data in a pandas DataFrame. """

        return self._collect(self.data)

    def make_figures(self):
        """ Make figures. """

        figures = pd.Series()

        for s in iterator(self.data, self.stack):

            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                fig_kwargs = {'x_axis_type': 'datetime'}
            else:
                fig_kwargs = dict()

            # link x axis ranges
            if len(figures) > 0:
                fig_kwargs['x_range'] = figures.iloc[0].x_range

            figures.loc[s] = figure(
                plot_width=self.figsize[0], plot_height=self.figsize[1],
                tools=self.tools, toolbar_location='above', title=str(s),
                **fig_kwargs)

            figures.loc[s].xgrid.grid_line_color = None
            figures.loc[s].ygrid.grid_line_color = None

        for e in self.added_figures:

            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                fig_kwargs = {'x_axis_type': 'datetime'}
            else:
                fig_kwargs = dict()

            figures.loc[e.name] = figure(
                plot_width=self.figsize[0], plot_height=self.figsize[1],
                tools=self.tools, toolbar_location='above', title=e.name,
                x_range=figures.iloc[0].x_range, **fig_kwargs)

        return figures

    def make_handlers(self):
        """ Make handlers. """

        return ResamplingDataHandler(
            self.collect(), self.resolution * self.figsize[0], context=self,
            lowpass=self.lowpass)

    def add_glyphs(self):
        """ Add glyphs. """

        for s in iterator(self.data, self.stack):

            for idx_o, o in enumerate(iterator(self.data, self.overlay)):

                color = self.colors[np.mod(idx_o, len(self.colors))]
                self.figures.loc[s].line(
                    x='index', y='_'.join((o, s)), source=self.handler.source,
                    line_color=color, line_alpha=0.6, legend=o)
                circ = self.figures.loc[s].circle(
                    x='index', y='_'.join((o, s)), source=self.handler.source,
                    color=color, size=0)

        circ.data_source.on_change('selected', self.on_selected_points_change)

    def add_tooltips(self):
        """ Add tooltips. """

        for f in self.figures:
            f.select(HoverTool).tooltips = [('datetime', '@index{%F %T.%3N}')]
            f.select(HoverTool).formatters = {'index': 'datetime'}

    def add_callbacks(self):
        """ Add callbacks. """

        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

    def make_layout(self):
        """ Make the app layout. """

        # make figures
        self.figures = self.make_figures()

        # make handlers
        self.handler = self.make_handlers()

        # add glyphs
        self.add_glyphs()

        # customize hover tooltips
        self.add_tooltips()

        # add callbacks
        self.add_callbacks()

        # create layout
        self.layout = gridplot(self.figures, ncols=self.ncols)

    def make_app(self, doc):
        """ Make the app for displaying in a jupyter notebbok. """

        # add layout to document
        self.doc = doc
        self.make_layout()
        self.doc.add_root(self.layout)

    def add_figure(self, element):
        """ Add a figure to the layout. """

    def add_overlay(self, element, onto=None):
        """ Add an overlay to a figure in the layout. """

    def show(self, notebook_url=None, port=0):
        """ Show the app in a jupyter notebook.

        Parameters
        ----------
        notebook_url : str, optional
            The URL of the notebook.

        port : int, default 0
            The port over which the app will be served. Chosen randomly if
            set to 0.
        """

        if notebook_url is None:
            notebook_url = get_notebook_url()

        output_notebook(hide_banner=True)
        app = Application(FunctionHandler(self.make_app))
        app.create_document()
        show_app(app, None, notebook_url=notebook_url, port=port)
