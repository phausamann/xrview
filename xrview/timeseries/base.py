""" ``xrview.timeseries.base`` """

import numpy as np
import pandas as pd

from bokeh.layouts import row, column
from bokeh.models import Span, ColumnDataSource
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


class BaseViewer(object):
    """"""

    def __init__(self, data,
                 points_per_pixel=4,
                 max_workers=10,
                 figsize=(700, 500)):

        self.plot_data = data
        self.points_per_pixel = points_per_pixel
        self.figsize = figsize

        self.thread_pool = ThreadPoolExecutor(max_workers)

        self.doc = None
        self.figure = None
        self.plot_source = None

        self.source_data = None
        self.selection = []

        self.pending_xrange_update = False
        self.xrange_change_buffer = None

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

        # get new start and end
        if start is not None:
            if start < self.plot_source.data['index'][0]:
                start = max(self.plot_data.index[0], int(start))
            else:
                start = int(start)
        elif len(self.plot_source.data['index']) > 0:
            start = self.plot_source.data['index'][0]
        else:
            start = self.plot_data.index[0]

        if end is not None:
            if end > self.plot_source.data['index'][-1]:
                end = min(self.plot_data.index[-1], int(end))
            else:
                end = int(end)
        elif len(self.plot_source.data['index']) > 0:
            end = self.plot_source.data['index'][-1]
        else:
            end = self.plot_data.index[-1]

        return start, end

    def from_range(self, start, end):
        """ Get sub-sampled source data from index range.

        Parameters
        ----------
        start : numeric
            The start of the range to be displayed.

        end : numeric
            The end of the range to be displayed.

        Returns
        -------
        df : pandas DataFrame
            The sub-sampled slice of the data to be displayed.
        """

        if start is None:
            start = 0
        else:
            start = self.plot_data.index.get_loc(start)

        if end is None:
            end = self.plot_data.shape[0]
        else:
            end = self.plot_data.index.get_loc(end)

        factor = int(np.ceil(
            (end-start) / (self.points_per_pixel*self.figsize[0])))

        df = self.plot_data.iloc[start:end:factor]

        # hacky solution for range reset
        if start > 0:
            df = pd.concat((self.plot_data.iloc[:1], df))
        if end < self.plot_data.shape[0]-1:
            df = df.append(self.plot_data.iloc[-1])

        return df

    def get_new_source_data(self, start, end):
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

        df = self.from_range(start, end)
        new_source_data = df.to_dict(orient='list')
        new_source_data['index'] = df.index

        return new_source_data

    def get_updated_data(self):
        """

        Returns
        -------
        new_source_data : dict
            The sub-sampled slice of the data to be displayed.

        new_selection : list
            The indices of the selected points in the sub-sampled slice.
        """

        start, end = self.get_range(self.figure.x_range.start,
                                    self.figure.x_range.end)

        new_source_data = self.get_new_source_data(start, end)

        # update source selection
        if self.plot_source.selected is not None \
                and np.sum(self.plot_data.selected) > 0:
            sel_idx = self.plot_data[self.plot_data.selected].index
            source_idx = new_source_data['index']
            matched_idx = source_idx.intersection(pd.Index(sel_idx))
            new_selection = list(source_idx.get_indexer(matched_idx))
        else:
            new_selection = []

        return new_source_data, new_selection

    def get_reset_data(self):
        """

        Returns
        -------
        new_source_data : dict
            The sub-sampled slice of the data to be displayed.

        new_selection : list
            The indices of the selected points in the sub-sampled slice.
        """
        return self.get_new_source_data(None, None), []

    @gen.coroutine
    def update_source(self):
        """ Update data and selected.indices of self.plot_source """

        self.plot_source.data = self.source_data
        self.plot_source.selected.indices = self.selection
        self.pending_xrange_update = False

    @without_document_lock
    @gen.coroutine
    def reset_xrange(self):
        """ """

        self.source_data, self.selection = \
            yield self.thread_pool.submit(self.get_reset_data)
        self.doc.add_next_tick_callback(self.update_source)

    @without_document_lock
    @gen.coroutine
    def update_xrange(self):
        """  """

        self.source_data, self.selection = \
            yield self.thread_pool.submit(self.get_updated_data)
        self.doc.add_next_tick_callback(self.update_source)

        if self.xrange_change_buffer is not None:
            self.doc.add_next_tick_callback(self.update_xrange)
            self.xrange_change_buffer = None

    def on_xrange_change(self, attr, old, new):
        """  """

        if not self.pending_xrange_update:
            self.pending_xrange_update = True
            self.doc.add_next_tick_callback(self.update_xrange)
        else:
            self.xrange_change_buffer = new

    def on_reset(self, event):
        """  """

        self.pending_xrange_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

    def make_app(self, doc):
        """ """

        TOOLS = 'pan,wheel_zoom,xbox_select,reset'

        self.doc = doc

        # create main figure
        self.figure = figure(
            plot_width=self.figsize[0], plot_height=self.figsize[1],
            tools=TOOLS, toolbar_location='above')

        self.plot_source = ColumnDataSource(self.plot_data)
        self.plot_source.data = self.get_new_source_data(*self.get_range())

        self.figure.line(x='index', y='y', source=self.plot_source)
        self.figure.circle(x='index', y='y', source=self.plot_source, size=0)

    def show(self, notebook_url, port=0):
        """

        Parameters
        ----------
        notebook_url : str
        port : int, default 0
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
    span_coord :
    """

    def __init__(self, data, sample_dim='sample', axis_dim='axis',
                 select_coord=None, span_coord=None, figsize=(700, 500)):

        self.data = data
        self.sample_dim = sample_dim
        self.axis_dim = axis_dim
        self.select_coord = select_coord
        self.span_coord = span_coord

        super(TimeseriesViewer, self).__init__(self.data, figsize=figsize)

        self.span_data = None

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
            sel_idx = np.zeros(self.data.sizes[self.sample_dim], dtype=bool)
            for c in coord_vals:
                sel_idx = sel_idx | (self.data[self.select_coord].values == c)
            plot_data = {
                axis: self.data.sel(**{self.axis_dim: axis}).values[sel_idx]
                for axis in self.data[self.axis_dim].values}
            plot_data['selected'] = np.zeros(np.sum(sel_idx), dtype=bool)
        else:
            plot_data = {
                axis: self.data.sel(**{self.axis_dim: axis}).values
                for axis in self.data[self.axis_dim].values}
            plot_data['selected'] = np.zeros(
                self.data.sizes[self.sample_dim], dtype=bool)

        self.plot_data = pd.DataFrame(plot_data)

    def on_selected_coord_change(self, attr, old, new):
        """ """

        self.collect(new)

        start, end = self.get_range(
            self.figure.x_range.start, self.figure.x_range.end)

        self.plot_source.data = self.get_new_source_data(start, end)

        if self.plot_source.selected is not None:
            self.plot_source.selected.indices = []

    def on_selected_points_change(self, attr, old, new):
        """ """

        idx_new = np.array(new['1d']['indices'])
        self.plot_data.selected = np.zeros(
            len(self.plot_data.selected), dtype=bool)
        sel_idx_start = self.plot_source.data['index'][np.min(idx_new)]
        sel_idx_end = self.plot_source.data['index'][np.max(idx_new)]
        self.plot_data.selected[np.logical_and(
            self.plot_data.index >= sel_idx_start,
            self.plot_data.index <= sel_idx_end)] = True

        if self.span_coord is not None:

            if len(idx_new) == 0:
                for s in self.vlines['span']:
                    s.visible = True
            else:
                for s in self.vlines['span']:
                    s.visible = False
                s_idx = self.vlines.index[
                    np.logical_and(self.vlines.index >= min(idx_new),
                                   self.vlines.index <= max(idx_new))]
                for s in self.vlines.loc[s_idx, 'span']:
                    s.visible = True

    def make_app(self, doc):

        TOOLS = 'pan,wheel_zoom,xbox_select,reset'
        COLORS = ['red', 'green', 'blue']

        self.doc = doc

        # create main figure
        self.figure = figure(
            plot_width=self.figsize[0], plot_height=self.figsize[1],
            tools=TOOLS, toolbar_location='above')

        self.figure.xgrid.grid_line_color = None
        self.figure.ygrid.grid_line_color = None

        # create dropdown
        if self.select_coord is not None:
            options = [
                (v, v) for v in np.unique(self.data[self.select_coord])]
            multi_select = MultiSelect(
                title=self.select_coord, value=[options[0][0]], options=options)
            multi_select.size = len(options)
            layout = row(self.figure, multi_select)
            # create data source
            self.collect([options[0][1]])
        else:
            layout = self.figure
            # create data source
            self.collect()

        self.plot_source = ColumnDataSource(self.plot_data)
        self.plot_source.data = self.get_new_source_data(*self.get_range())

        # create vertical spans
        if self.span_coord is not None:

            self.vlines = []
            locations = np.where(
                np.diff(np.array(self.data[self.span_coord],
                                 dtype=float)) > 0)[0]

            for l in locations:
                self.vlines.append(Span(
                    location=l, dimension='height', line_color='grey',
                    line_width=1, line_alpha=0.5))
                self.figure.add_layout(self.vlines[-1])

            self.vlines = pd.DataFrame({'span': self.vlines}, index=locations)

        for idx, axis in enumerate(self.data.axis.values):

            c = COLORS[np.mod(idx, len(COLORS))]
            self.figure.line(x='index', y=axis, source=self.plot_source,
                        line_color=c, line_alpha=0.6)
            circ = self.figure.circle(
                x='index', y=axis, source=self.plot_source, color=c, size=0)

        circ.data_source.on_change('selected', self.on_selected_points_change)

        self.figure.x_range.on_change('start', self.on_xrange_change)
        self.figure.x_range.on_change('end', self.on_xrange_change)
        self.figure.on_event(Reset, self.on_reset)

        if self.select_coord is not None:
            multi_select.on_change('value', self.on_selected_coord_change)

        self.doc.add_root(layout)
