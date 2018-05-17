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

from xrview.utils import is_dataset, is_dataarray


class BaseViewer(object):
    """ Base class for timeseries viewers.

    Parameters
    ----------
    data : pandas DataFrame
        The data to be displayed.

    resolution : int, default 4
        The number of points to render for each pixel.

    max_workers : int, default 10
        The maximum number of workers in the thread pool to perform the
        sub-sampling.

    figsize : sequence of int, default (700, 500)
        The size of the figure in pixels.
    """

    def __init__(self, data, resolution=4, max_workers=10, figsize=(700, 500)):

        self.plot_data = data

        self.resolution = resolution
        self.figsize = figsize

        self.plot_source = None
        self.plot_source_data = None

        self.thread_pool = ThreadPoolExecutor(max_workers)

        self.doc = None
        self.figures = None

        self.selection = []

        self.pending_xrange_update = False
        self.xrange_change_buffer = None

        self.tools = 'pan,wheel_zoom,box_zoom,xbox_select,reset,hover'

    def _init_source(self, with_range=False):
        """ Init self.plot_source """

        if with_range:
            self.plot_source_data = self.get_dict_from_range(
                self.plot_data.index[0], self.plot_data.index[-1])
            self.plot_source = ColumnDataSource(self.plot_source_data)
        else:
            self.plot_source = ColumnDataSource(self.plot_data)
            self.plot_source.add(self.plot_data.index, 'index')
            self.plot_source_data = self.plot_source.data

    @staticmethod
    def from_range(data, factor, start, end):
        """ Get sub-sampled pandas DataFrame from index range.

        Parameters
        ----------
        data : pandas DataFrame
            The data to be sub-sampled

        factor : numeric
            The subsampling factor.

        start : numeric
            The start of the range to be sub-sampled.

        end : numeric
            The end of the range to be sub-sampled.

        Returns
        -------
        data_new : pandas DataFrame
            A sub-sampled slice of the data.
        """

        if start is None:
            start = 0
        else:
            start = data.index.get_loc(start, method='nearest')

        if end is None:
            end = data.shape[0]
        else:
            end = data.index.get_loc(end, method='nearest') + 1

        step = int(np.ceil((end-start) / factor))

        # TODO: handle NaNs at start/end
        if step == 0:
            # hacky solution for range reset
            data_new = pd.concat((data.iloc[:1], data.iloc[-1:]))
        else:
            data_new = data.iloc[start:end:step]
            # hacky solution for range reset
            if start > 0:
                data_new = pd.concat((data.iloc[:1], data_new))
            if end < data.shape[0]-1:
                data_new = data_new.append(data.iloc[-1])

        return data_new

    def get_range(self, start=None, end=None):
        """ Get the range of valid indexes for the data to be displayed.

        Parameters
        ----------
        start : numeric
            The start of the range to be displayed.

        end : numeric
            The end of the range to be displayed.

        Returns
        -------
        start : numeric
            The adjusted start.

        end : numeric
            The adjusted end.
        """

        # convert to timestamp if necessary
        if isinstance(self.plot_data.index, pd.DatetimeIndex):
            start = pd.to_datetime(start, unit='ms')
            end = pd.to_datetime(end, unit='ms')

        # get new start and end
        if start is not None:
            if start < self.plot_source.data['index'][0]:
                start = max(self.plot_data.index[0], start)
            elif start > self.plot_source.data['index'][-1]:
                start = min(self.plot_data.index[-1], start)
            elif start < self.plot_data.index[0]:
                start = self.plot_data.index[0]
            elif start > self.plot_data.index[-1]:
                start = self.plot_data.index[-1]
        elif len(self.plot_source.data['index']) > 0:
            start = self.plot_source.data['index'][0]
        else:
            start = self.plot_data.index[0]

        if end is not None:
            if end < self.plot_source.data['index'][0]:
                end = max(self.plot_data.index[0], end)
            elif end > self.plot_source.data['index'][-1]:
                end = min(self.plot_data.index[-1], end)
            elif end < self.plot_data.index[0]:
                end = self.plot_data.index[0]
            elif end > self.plot_data.index[-1]:
                end = self.plot_data.index[-1]
        elif len(self.plot_source.data['index']) > 0:
            end = self.plot_source.data['index'][-1]
        else:
            end = self.plot_data.index[-1]

        return start, end

    def get_dict_from_range(self, start, end):
        """ Get sub-sampled source data from index range as a dict.

        Parameters
        ----------
        start : numeric
            The start of the range to be displayed.

        end : numeric
            The end of the range to be displayed.

        Returns
        -------
        new_source_data : dict
            The sub-sampled slice of the data to be displayed.
        """

        df = self.from_range(
            self.plot_data, self.resolution * self.figsize[0], start, end)
        new_source_data = df.to_dict(orient='list')
        new_source_data['index'] = df.index

        for k in list(new_source_data):
            if isinstance(k, tuple):
                new_source_data['_'.join(k)] = new_source_data.pop(k)

        return new_source_data

    def update_data(self):
        """ Update data and selection to be displayed. """

        start, end = self.get_range(
            self.figures.x_range.start, self.figures.x_range.end)

        self.plot_source_data = self.get_dict_from_range(start, end)

        # update source selection
        if self.plot_source.selected is not None \
                and np.sum(self.plot_data.selected) > 0:
            self.selection = list(
                np.where(self.plot_source_data['selected'])[0])
        else:
            self.selection = []

    def reset_data(self):
        """ Reset data and selection to be displayed. """

        self.plot_source_data = self.get_dict_from_range(None, None)

        self.plot_data.selected = np.zeros(self.plot_data.shape[0], dtype=bool)
        self.selection = []

    @gen.coroutine
    def update_source(self):
        """ Update data and selected.indices of self.plot_source """

        self.plot_source.data = self.plot_source_data
        self.plot_source.selected.indices = self.selection
        self.pending_xrange_update = False

    @without_document_lock
    @gen.coroutine
    def reset_xrange(self):
        """ """

        yield self.thread_pool.submit(self.reset_data)
        self.doc.add_next_tick_callback(self.update_source)

    @without_document_lock
    @gen.coroutine
    def update_xrange(self):
        """ Update plot_source when xrange changes. """

        yield self.thread_pool.submit(self.update_data)
        self.doc.add_next_tick_callback(self.update_source)

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
        self.plot_data.selected = np.zeros(
            len(self.plot_data.selected), dtype=bool)
        sel_idx_start = self.plot_source.data['index'][np.min(idx_new)]
        sel_idx_end = self.plot_source.data['index'][np.max(idx_new)]
        self.plot_data.loc[np.logical_and(
            self.plot_data.index >= sel_idx_start,
            self.plot_data.index <= sel_idx_end), 'selected'] = True

    def on_reset(self, event):
        """ Callback for reset event. """

        self.pending_xrange_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

    def make_app(self, doc):
        """ Make the app for display in a jupyter notebook. """

        self.doc = doc

        # assign data source
        self._init_source(with_range=True)

        # create main figure
        if isinstance(self.plot_data.index, pd.DatetimeIndex):
            fig_kwargs = {'x_axis_type': 'datetime'}
        else:
            fig_kwargs = dict()

        self.figures = [figure(
            plot_width=self.figsize[0], plot_height=self.figsize[1],
            tools=self.tools, toolbar_location='above', **fig_kwargs)]

        self.figures[0].line(x='index', y='y', source=self.plot_source)
        circ = self.figures[0].circle(
            x='index', y='y', source=self.plot_source, size=0)

        # add callbacks
        circ.data_source.on_change('selected', self.on_selected_points_change)

        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

        self.doc.add_root(self.figures)

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


class TimeseriesViewer(BaseViewer):
    """

    Parameters
    ----------
    data :
    sample_dim :
    axis_dim :
    select_coord :
    vlines_coord :
    """

    def __init__(self, data, sample_dim='sample', axis_dim='axis',
                 select_coord=None, vlines_coord=None,
                 vlines_resolution=10, resolution=4, figsize=(700, 500)):

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

        # check select_coord
        if select_coord is not None and select_coord not in self.data.coords:
            raise ValueError(
                select_coord + ' is not a coordinate of the provided dataset.')
        else:
            self.select_coord = select_coord

        # check vlines_coord
        if vlines_coord is not None and select_coord not in self.data.coords:
            raise ValueError(
                vlines_coord + ' is not a coordinate of the provided dataset.')
        else:
            self.vlines_coord = vlines_coord

        self.vlines_resolution = vlines_resolution
        self.vlines_source = None
        self.vlines_source_data = None

        super(TimeseriesViewer, self).__init__(self.data,
                                               resolution=resolution,
                                               figsize=figsize)

        self.colors = ['red', 'green', 'blue']

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

            plot_data = {
                a + '_' + v: self.data[v].sel(**{self.axis_dim: a}).values[idx]
                for a in self.data[self.axis_dim].values
                for v in self.data.data_vars
            }

            plot_data['selected'] = np.zeros(np.sum(idx), dtype=bool)

            if self.vlines_coord is not None:
                plot_data['vlines'] = \
                    self.data[self.vlines_coord].values[idx]

            self.plot_data = pd.DataFrame(
                plot_data, index=self.data[self.sample_dim][idx])

        else:

            plot_data = {
                a + '_' + v: self.data[v].sel(**{self.axis_dim: a}).values
                for a in self.data[self.axis_dim].values
                for v in self.data.data_vars
            }

            plot_data['selected'] = np.zeros(
                self.data.sizes[self.sample_dim], dtype=bool)

            if self.vlines_coord is not None:
                plot_data['vlines'] = self.data[self.vlines_coord].values

            self.plot_data = pd.DataFrame(
                plot_data, index=self.data[self.sample_dim])

    def get_vlines_dict(self):
        """ Get sub-sampled vlines locations as a dict.

        Returns
        -------
        vlines_source_data : dict
            A dict with the source data.
        """

        df = self.plot_source.to_df()
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
        df = self.from_range(self.plot_data, factor, start, end)
        vline_idx = df.vlines.astype(bool)

        return {'index': df.index[vline_idx], '0': np.zeros(np.sum(vline_idx))}

    def add_vlines(self):

        self.vlines_source_data = \
            self.get_vlines_dict_from_range(*self.get_range())
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

    def update_data(self):
        """ Update data and selection to be displayed. """

        start, end = self.get_range(self.figures[0].x_range.start,
                                    self.figures[0].x_range.end)

        self.plot_source_data = self.get_dict_from_range(start, end)

        if self.vlines_coord is not None:
            self.vlines_source_data = \
                self.get_vlines_dict_from_range(start, end)

        # update source selection
        if self.plot_source.selected is not None \
                and np.sum(self.plot_data.selected) > 0:
            self.selection = list(
                np.where(self.plot_source_data['selected'])[0])
        else:
            self.selection = []

    @gen.coroutine
    def update_source(self):
        """ Update data and selected.indices of self.plot_source """

        self.plot_source.data = self.plot_source_data
        self.plot_source.selected.indices = self.selection

        if self.vlines_coord is not None:
            self.vlines_source.data = self.vlines_source_data

        self.pending_xrange_update = False

    def on_selected_coord_change(self, attr, old, new):
        """ Callback for multi-select change event. """

        self.collect(new)

        start, end = self.get_range(
            self.figures[0].x_range.start, self.figures[0].x_range.end)

        self.plot_source.data = self.get_dict_from_range(start, end)

        if self.vlines_coord is not None:
            # self.vlines_source_data = self.get_vlines_dict()
            self.vlines_source_data = \
                self.get_vlines_dict_from_range(start, end)
            self.vlines_source.data = self.vlines_source_data

        if self.plot_source.selected is not None:
            self.plot_source.selected.indices = []

    def make_app(self, doc):
        """ Make the app for displaying in a jupyter notebbok. """

        self.doc = doc

        # create figures
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

        # create layout
        layout = gridplot([[f] for f in self.figures])

        # create multi_select
        if self.select_coord is not None:
            options = [
                (v, v) for v in np.unique(self.data[self.select_coord])]
            multi_select = MultiSelect(
                title=self.select_coord, value=[options[0][0]], options=options)
            multi_select.size = len(options)
            multi_select.on_change('value', self.on_selected_coord_change)
            layout = row(layout, multi_select)
            self.collect([options[0][1]])
        else:
            self.collect()

        # initialize the data source
        self._init_source(with_range=True)

        # add vertical lines
        if self.vlines_coord is not None:
            self.add_vlines()

        # plot lines
        for i_v, v in enumerate(self.data.data_vars):
            for i_a, a in enumerate(self.data[self.axis_dim].values):
                c = self.colors[np.mod(i_a, len(self.colors))]
                self.figures[i_v].line(
                    x='index', y='_'.join((a, v)), source=self.plot_source,
                    line_color=c, line_alpha=0.6, legend=a)
                circ = self.figures[i_v].circle(
                    x='index', y='_'.join((a, v)), source=self.plot_source,
                    color=c, size=0)

        # customize hover tooltips
        for f in self.figures:
            f.select(HoverTool).tooltips = [('datetime', '@index{%F %T.%3N}')]
            f.select(HoverTool).formatters = {'index': 'datetime'}

        # add callbacks
        circ.data_source.on_change('selected', self.on_selected_points_change)

        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

        # add layout to document
        self.doc.add_root(layout)
