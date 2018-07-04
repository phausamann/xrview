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
from bokeh.palettes import Paired12

from bokeh.document import without_document_lock
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from xrview.utils import is_dataset, is_dataarray, get_notebook_url

from .handlers import ResamplingDataHandler


_RGB = Paired12[5::-2] + Paired12[-1:5:-2] + Paired12[4::-2] + Paired12[-2:4:-2]


def _map_vars_and_dims(data, x, overlay):
    """ Map data variables and dimensions to figures and overlays. """

    figure_map = OrderedDict()

    if overlay == 'dim':

        for v in data.data_vars:
            if x not in data[v].dims:
                raise ValueError(x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                figure_map[v] = None
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != x][0]
                figure_map[v] = tuple(data[dim].values)
            else:
                raise ValueError(v + ' has too many dimensions')

    elif overlay == 'var':

        for v in data.data_vars:
            if x not in data[v].dims:
                raise ValueError(x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                if 'dim' not in locals():
                    dim = None
                elif dim is not None:
                    raise ValueError(
                        'Dimensions of all data variables must match')
            elif len(data[v].dims) == 2:
                if 'dim' not in locals():
                    dim = [d for d in data[v].dims if d != x][0]
                elif dim not in data[v].dims:
                    raise ValueError(
                        'Dimensions of all data variables must match')
            else:
                raise ValueError(v + ' has too many dimensions')

        figure_map = {d: tuple(data.data_vars) for d in data[dim].values}

    else:
        raise ValueError('overlay must be "dims" or "data_vars"')

    return figure_map


class Viewer(object):
    """ Base class for timeseries viewers.

    Parameters
    ----------
    data :

    x :

    overlay : str, default 'dim'

    figsize : iterable, default (900, 400)
        The size of the figure in pixels.

    resolution : int, default 4
        The number of points to render for each pixel.

    max_workers : int, default 10
        The maximum number of workers in the thread pool to perform the
        sub-sampling.
    """

    def __init__(self, data, x, overlay='dim', tooltips=None,
                 figsize=(900, 400), ncols=1, palette=None, resolution=4,
                 max_workers=10, lowpass=False, verbose=0):

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
        if overlay in ('dim', 'var'):
            self.overlay = overlay
        else:
            raise ValueError('overlay must be "dim" or "var"')

        # TODO: check tooltips
        self.tooltips = tooltips

        # layout parameters
        self.figsize = figsize
        self.ncols = ncols

        if palette is None:
            self.palette = _RGB
        else:
            self.palette = palette

        # sub-sampling parameters
        self.resolution = resolution
        self.thread_pool = ThreadPoolExecutor(max_workers)
        self.lowpass = lowpass
        self.verbose = verbose

        #
        self.added_figures = []
        self.added_overlays = []
        self.added_interactions = []

        self.handlers = None
        self.figure_map = None
        self.figures = None
        self.selection = []

        self.x_range = None
        self.pending_xrange_update = False
        self.xrange_change_buffer = None

        self.tools = 'pan,wheel_zoom,box_zoom,xbox_select,reset,hover'

        self.doc = None

        self.layout = None

    @without_document_lock
    @gen.coroutine
    def reset_xrange(self):
        """ """

        for h in self.handlers:
            yield self.thread_pool.submit(h.reset_data)
            self.doc.add_next_tick_callback(h.update_source)

    @without_document_lock
    @gen.coroutine
    def update_xrange(self):
        """ Update plot_source when xrange changes. """

        for h in self.handlers:
            yield self.thread_pool.submit(partial(
                h.update_data,
                start=self.figures[0].x_range.start,
                end=self.figures[0].x_range.end))
            self.doc.add_next_tick_callback(h.update_source)

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

        for h in self.handlers:
            h.data.selected = np.zeros(len(h.data.selected), dtype=bool)
            sel_idx_start = h.source.data['index'][np.min(idx_new)]
            sel_idx_end = h.source.data['index'][np.max(idx_new)]
            h.data.loc[np.logical_and(
                h.data.index >= sel_idx_start,
                h.data.index <= sel_idx_end), 'selected'] = True

    def on_reset(self, event):
        """ Callback for reset event. """

        self.pending_xrange_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

    def _collect(self, data):
        """ Base method for collect. """

        plot_data = dict()

        for v in data.data_vars:
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                plot_data[v] = data[v].values
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for d in data[dim].values:
                    plot_data[v + '_' + str(d)] = data[v].sel(**{dim: d}).values
            else:
                raise ValueError(v + ' has too many dimensions')

        plot_data['selected'] = np.zeros(data.sizes[self.x], dtype=bool)

        return pd.DataFrame(plot_data, index=data[self.x])

    def collect(self):
        """ Collect plottable data in a pandas DataFrame. """

        return self._collect(self.data)

    def make_handlers(self):
        """ Make handlers. """

        # default handler
        handlers = [ResamplingDataHandler(
            self.collect(), self.resolution * self.figsize[0], context=self,
            lowpass=self.lowpass)]

        for element in self.added_figures + self.added_overlays:
            if element.handler is not None:
                handlers.append(element.handler)
            else:
                pass
                # TODO: element.update(self.handlers[0])

        return handlers

    def make_figure_map(self, data, handler):
        """ Make the figure map. """

        figure_list = []

        for v in data.data_vars:
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                figure_list.append((v, None, None))
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for dval in data[dim].values:
                    figure_list.append((v, dim, dval))
            else:
                raise ValueError(v + ' has too many dimensions')

        figure_map = pd.DataFrame(
            figure_list, columns=['var', 'dim', 'dim_val'])

        figure_map.loc[:, 'handler'] = handler

        # TODO: create figure column with index of self.figures here

        return figure_map

    def make_figures(self):
        """ Make figures. """

        # TODO: check if OrderedDict with figure id is better suited
        self.figures = []

        # add base figures
        figure_map = self.make_figure_map(self.data, self.handlers[0])

        if self.overlay == 'dim':
            column_vals = figure_map['var']
        else:
            if len(np.unique(figure_map['dim'])) > 1:
                raise ValueError(
                    'Dimensions of all data variables must match')
            else:
                column_vals = figure_map['dim_val']

        iterator = np.unique(column_vals)

        for it in iterator:

            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                fig_kwargs = {'x_axis_type': 'datetime'}
            else:
                fig_kwargs = dict()

            # link x axis ranges
            if len(self.figures) > 0:
                fig_kwargs['x_range'] = self.figures[0].x_range

            self.figures.append(figure(
                plot_width=self.figsize[0]//self.ncols,
                plot_height=self.figsize[1]//len(iterator)*self.ncols,
                tools=self.tools, toolbar_location='above', title=str(it),
                **fig_kwargs))

            figure_map.loc[column_vals == it, 'figure'] = self.figures[-1]

        # add additional figures
        for f in self.added_figures:

            f_map = self.make_figure_map(f.data, f.handler)

            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                fig_kwargs = {'x_axis_type': 'datetime'}
            else:
                fig_kwargs = dict()

            fig_kwargs['x_range'] = self.figures[0].x_range

            self.figures.append(figure(
                plot_width=self.figsize[0]//self.ncols,
                plot_height=self.figsize[1]//len(iterator)*self.ncols,
                tools=self.tools, toolbar_location='above', # title=str(it),
                **fig_kwargs))

            f_map.loc[:, 'figure'] = self.figures[-1]

            figure_map = pd.concat([figure_map, f_map])

        return figure_map

    def add_glyphs(self, figure_map):
        """ Add glyphs. """

        if self.overlay == 'var':
            legend_col = 'var'
        else:
            legend_col = 'dim_val'

        colormap = {v: self.palette[i]
                    for i, v in enumerate(pd.unique(figure_map[legend_col]))}

        for idx, f in figure_map.iterrows():
            if f['dim_val'] is None:
                source_col = str(f['var'])
            else:
                source_col = '_'.join((str(f['var']), str(f['dim_val'])))
            f.figure.line(
                x='index', y=source_col, source=f.handler.source,
                line_alpha=0.6, legend=f[legend_col],
                color=colormap[f[legend_col]])
            circ = f.figure.circle(
                x='index', y=source_col, source=f.handler.source, size=0)
            circ.data_source.on_change(
                'selected', self.on_selected_points_change)

    def add_tooltips(self):
        """ Add tooltips. """

    def add_callbacks(self):
        """ Add callbacks. """

        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

    def make_layout(self):
        """ Make the app layout. """

        for element in self.added_figures:
            element.attach(self)

        # make handlers
        self.handlers = self.make_handlers()

        # make figures
        figure_map = self.make_figures()

        # add glyphs
        self.add_glyphs(figure_map)

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

        self.added_figures.append(element)

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
