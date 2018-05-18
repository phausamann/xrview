""" ``xrview.timeseries.base`` """

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

from xrview.utils import is_dataset, is_dataarray

from .handlers import ResamplingDataHandler


class BaseViewer(object):
    """ Base class for timeseries viewers.

    Parameters
    ----------
    data :

    sample_dim :

    axis_dim :

    figsize : iterable, default (700, 500)
        The size of the figure in pixels.

    resolution : int, default 4
        The number of points to render for each pixel.

    max_workers : int, default 10
        The maximum number of workers in the thread pool to perform the
        sub-sampling.
    """

    def __init__(self, data, sample_dim='sample', axis_dim='axis',
                 figsize=(700, 500), resolution=4, max_workers=10):

        # check data
        if is_dataarray(data):
            self.data = data.to_dataset(name='Data')
        elif is_dataset(data):
            self.data = data
        else:
            raise ValueError('data must be xarray DataArray or Dataset.')

        # check sample_dim
        if sample_dim in self.data.dims:
            self.sample_dim = sample_dim
        else:
            raise ValueError(
                sample_dim + ' is not a dimension of the provided dataset.')

        # check axis_dim
        if axis_dim in self.data.dims:
            self.axis_dim = axis_dim
        else:
            raise ValueError(
                axis_dim + ' is not a dimension of the provided dataset.')

        self.resolution = resolution
        self.figsize = figsize

        self.thread_pool = ThreadPoolExecutor(max_workers)

        self.doc = None
        self.figures = None

        self.handler = None

        self.selection = []

        self.pending_xrange_update = False
        self.xrange_change_buffer = None

        self.tools = 'pan,wheel_zoom,box_zoom,xbox_select,reset,hover'
        self.colors = ['red', 'green', 'blue']

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
            str(a) + '_' + str(v): data[v].sel(**{self.axis_dim: a}).values
            for a in data[self.axis_dim].values
            for v in data.data_vars
        }

        plot_data['selected'] = np.zeros(
            data.sizes[self.sample_dim], dtype=bool)

        return pd.DataFrame(plot_data, index=data[self.sample_dim])

    def collect(self):
        """ Collect plottable data in a pandas DataFrame. """

        return self._collect(self.data)

    def make_figures(self):

        self.figures = []
        for i_v, v in enumerate(self.data.data_vars):

            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.sample_dim], pd.DatetimeIndex):
                fig_kwargs = {'x_axis_type': 'datetime'}
            else:
                fig_kwargs = dict()

            # link x axis ranges
            if i_v > 0:
                fig_kwargs['x_range'] = self.figures[0].x_range

            self.figures.append(figure(
                plot_width=self.figsize[0], plot_height=self.figsize[1],
                tools=self.tools, toolbar_location='above', title=v,
                **fig_kwargs))

            self.figures[i_v].xgrid.grid_line_color = None
            self.figures[i_v].ygrid.grid_line_color = None

    def plot_lines(self):

        for i_v, v in enumerate(self.data.data_vars):
            for i_a, a in enumerate(self.data[self.axis_dim].values):
                c = self.colors[np.mod(i_a, len(self.colors))]
                self.figures[i_v].line(
                    x='index', y='_'.join((a, v)), source=self.handler.source,
                    line_color=c, line_alpha=0.6, legend=a)
                circ = self.figures[i_v].circle(
                    x='index', y='_'.join((a, v)), source=self.handler.source,
                    color=c, size=0)

        circ.data_source.on_change('selected', self.on_selected_points_change)

    def add_callbacks(self):

        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

    def make_layout(self):
        """ Make the app layout. """

        # create figures
        self.make_figures()

        # create layout
        self.layout = gridplot(self.figures, ncols=1)

        # create handler
        self.handler = ResamplingDataHandler(
            self.collect(), self.resolution * self.figsize[0], context=self)

        # plot lines
        self.plot_lines()

        # customize hover tooltips
        for f in self.figures:
            f.select(HoverTool).tooltips = [('datetime', '@index{%F %T.%3N}')]
            f.select(HoverTool).formatters = {'index': 'datetime'}

        # add callbacks
        self.add_callbacks()

    def make_app(self, doc):
        """ Make the app for displaying in a jupyter notebbok. """

        # add layout to document
        self.doc = doc
        self.make_layout()
        self.doc.add_root(self.layout)

    def show(self, notebook_url, port=0):
        """ Show the app in a jupyter notebook.

        Parameters
        ----------
        notebook_url : str
            The URL of the notebook.

        port : int, default 0
            The port over which the app will be served. Chosen randomly if
            set to 0.
        """

        output_notebook()
        app = Application(FunctionHandler(self.make_app))
        app.create_document()
        show_app(app, None, notebook_url=notebook_url, port=port)


class MultiSelectMixin(BaseViewer):
    """"""

    def collect(self, coord_vals=None):
        """ Collect plottable data in a pandas DataFrame.

        Parameters
        ----------
        coord_vals : sequence of str, optional
            If specified, collect the subset of self.data where the values of
            the coordinate self.select_coord match any of the values in
            coord_vals.
        """

        if coord_vals is not None:
            idx = np.zeros(self.data.sizes[self.sample_dim], dtype=bool)
            for c in coord_vals:
                idx = idx | (self.data[self.select_coord].values == c)
            return self._collect(self.data.isel(**{self.sample_dim: idx}))
        else:
            return self._collect(self.data)

    def on_selected_coord_change(self, attr, old, new):
        """ Callback for multi-select change event. """

        self.handler.data = self.collect(new)
        start, end = self.handler.get_range(
            self.figures[0].x_range.start, self.figures[0].x_range.end)
        self.handler.update_data(start, end)
        self.handler.update_source()

        if self.handler.source.selected is not None:
            self.handler.source.selected.indices = []

    def make_multi_select(self):
        """ """

        options = [(v, v) for v in np.unique(self.data[self.select_coord])]

        multi_select = MultiSelect(
            title=self.select_coord, value=[options[0][0]], options=options)
        multi_select.size = len(options)
        multi_select.on_change('value', self.on_selected_coord_change)

        self.layout = row(self.layout, multi_select)
        self.handler.data = self.collect([options[0][1]])

    def make_layout(self):
        """ """

        super(MultiSelectMixin, self).make_layout()

        self.make_multi_select()


class VLinesMixin(BaseViewer):
    """"""

    def _collect(self, data):
        """ """

        plot_data = {
            str(a) + '_' + str(v): data[v].sel(**{self.axis_dim: a}).values
            for a in data[self.axis_dim].values
            for v in data.data_vars
        }

        plot_data['selected'] = np.zeros(
            data.sizes[self.sample_dim], dtype=bool)

        if self.vlines_coord is not None:
            plot_data['vlines'] = data[self.vlines_coord].values

        return pd.DataFrame(plot_data, index=data[self.sample_dim])

    def get_vlines_dict(self):
        """ Get sub-sampled vlines locations as a dict.

        Returns
        -------
        vlines_source_data : dict
            A dict with the source data.
        """

        df = self.handler.source.to_df()
        df = df.loc[df.vlines, ['index']]
        vlines_source_data = df.to_dict(orient='list')
        vlines_source_data['0'] = np.zeros(df.shape[0])

        return vlines_source_data

    def get_vlines_dict_from_range(self, start, end):
        """ Get sub-sampled vlines locations as a dict.

        Returns
        -------
        vlines_source_data : dict
            A dict with the source data.
        """

        factor = self.vlines_resolution * self.resolution * self.figsize[0]
        df = self.handler.from_range(self.handler.data, factor, start, end)
        vline_idx = df.vlines.astype(bool)

        return {'index': df.index[vline_idx], '0': np.zeros(np.sum(vline_idx))}

    def add_vlines(self):
        """ """

        self.vlines_source_data = \
            self.get_vlines_dict_from_range(*self.handler.get_range())
        self.vlines_source = ColumnDataSource(self.vlines_source_data)

        for i_v in range(len(self.figures)):
            self.figures[i_v].ray(
                x='index', y='0', length=0, line_width=1, angle=90,
                angle_units='deg', color='grey', alpha=0.5,
                source=self.vlines_source)
            self.figures[i_v].ray(
                x='index', y='0', length=0, line_width=1, angle=270,
                angle_units='deg', color='grey', alpha=0.5,
                source=self.vlines_source)

    def on_selected_coord_change(self, attr, old, new):
        """ Callback for multi-select change event. """

        self.handler.data = self.collect(new)
        start, end = self.handler.get_range(
            self.figures[0].x_range.start, self.figures[0].x_range.end)
        self.handler.update_data(start, end)
        self.handler.update_source()

        if self.vlines_coord is not None:
            self.vlines_source_data = \
                self.get_vlines_dict_from_range(start, end)
            self.vlines_source.data = self.vlines_source_data

        if self.handler.source.selected is not None:
            self.handler.source.selected.indices = []

    def update_data(self):
        """ Update self.vlines_source_data. """

        start, end = self.handler.get_range(
            self.figures[0].x_range.start, self.figures[0].x_range.end)

        if self.vlines_coord is not None:
            self.vlines_source_data = \
                self.get_vlines_dict_from_range(start, end)

    def update_source(self):
        """ Update self.vlines_source. """

        if self.vlines_coord is not None:
            self.vlines_source.data = self.vlines_source_data

    def make_layout(self):
        """ """

        super(VLinesMixin, self).make_layout()

        if self.vlines_coord is not None:
            self.add_vlines()

        self.handler.add_callback('update_data', self.update_data)
        self.handler.add_callback('update_source', self.update_source)


class TimeseriesViewer(MultiSelectMixin, VLinesMixin, BaseViewer):
    """ """

    def __init__(self, data, select_coord=None, vlines_coord=None,
                 vlines_resolution=10, **kwargs):

        super(TimeseriesViewer, self).__init__(data, **kwargs)

        # check select_coord
        if select_coord is not None and select_coord not in self.data.coords:
            raise ValueError(
                select_coord + ' is not a coordinate of the provided dataset.')
        else:
            self.select_coord = select_coord

        # check vlines_coord
        if vlines_coord is not None and vlines_coord not in self.data.coords:
            raise ValueError(
                vlines_coord + ' is not a coordinate of the provided dataset.')
        else:
            self.vlines_coord = vlines_coord

        self.vlines_resolution = vlines_resolution
        self.vlines_source = None
        self.vlines_source_data = None
