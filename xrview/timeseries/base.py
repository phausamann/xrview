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

    def show(self, notebook_url, port=0):

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
        self.figsize = figsize

        self.points_per_pixel = 4

        self.plot_data = None
        self.render_data = None
        self.span_data = None

        self.plot_source = None

        self.thread_pool = ThreadPoolExecutor(10)

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

    def get_range(self, start=None, end=None):

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

        if start is None:
            start = 0

        if end is None:
            end = self.plot_data.shape[0]

        factor = int(np.ceil(
            (end-start) / (self.points_per_pixel*self.figsize[0])))

        df = self.plot_data.iloc[start:end:factor]

        if start > 0:
            df = pd.concat((self.plot_data.iloc[:1], df))

        if end < self.plot_data.shape[0]:
            df.append(self.plot_data.iloc[-1])

        return df

    def update_source(self, start, end):

        # get data to render
        df = self.from_range(start, end)

        # update source data
        df_dict = df.to_dict(orient='list')
        df_dict['index'] = df.index
        self.plot_source.data = df_dict

        # update source selection
        if self.plot_source.selected is not None \
                and np.sum(self.plot_data.selected) > 0:
            sel_idx = self.plot_data[self.plot_data.selected].index
            matched_idx = \
                self.plot_source.data['index'].intersection(pd.Index(sel_idx))
            self.plot_source.selected.indices = list(
                self.plot_source.data['index'].get_indexer(matched_idx))

    def make_app(self, doc):

        TOOLS = 'pan,wheel_zoom,xbox_select,reset'
        COLORS = ['red', 'green', 'blue']

        # create main figure
        p_plot = figure(
            plot_width=self.figsize[0], plot_height=self.figsize[1],
            tools=TOOLS, toolbar_location='above')

        p_plot.xgrid.grid_line_color = None
        p_plot.ygrid.grid_line_color = None

        # create dropdown
        if self.select_coord is not None:
            options = [
                (v, v) for v in np.unique(self.data[self.select_coord])]
            multi_select = MultiSelect(
                title=self.select_coord, value=[options[0][0]], options=options)
            multi_select.size = len(options)
            layout = row(p_plot, multi_select)
            # create data source
            self.collect([options[0][1]])
        else:
            layout = p_plot
            # create data source
            self.collect()

        global pending_xrange_update, xrange_change_buffer, \
            source_data, selection

        pending_xrange_update = False
        xrange_change_buffer = None

        self.plot_source = ColumnDataSource(self.plot_data)
        self.update_source(*self.get_range())

        source_data = self.plot_source.data
        selection = []

        # create vertical spans
        if self.span_coord is not None:

            vlines = []
            locations = np.where(
                np.diff(np.array(self.data[self.span_coord],
                                 dtype=float)) > 0)[0]

            for l in locations:
                vlines.append(Span(
                    location=l, dimension='height', line_color='grey',
                    line_width=1, line_alpha=0.5))
                p_plot.add_layout(vlines[-1])

            vlines = pd.DataFrame({'span': vlines}, index=locations)

        for idx, axis in enumerate(self.data.axis.values):
            c = COLORS[np.mod(idx, len(COLORS))]
            p_plot.line(x='index', y=axis, source=self.plot_source,
                        line_color=c, line_alpha=0.6)
            circ = p_plot.circle(
                x='index', y=axis, source=self.plot_source, color=c, size=0)

        def get_updated_data():

            start, end = self.get_range(p_plot.x_range.start,
                                        p_plot.x_range.end)

            df = self.from_range(start, end)
            new_source_data = df.to_dict(orient='list')
            new_source_data['index'] = df.index

            # update source selection
            if self.plot_source.selected is not None \
                    and np.sum(self.plot_data.selected) > 0:
                sel_idx = self.plot_data[self.plot_data.selected].index
                source_idx = self.plot_source.data['index']
                matched_idx = source_idx.intersection(pd.Index(sel_idx))
                new_selection = list(source_idx.get_indexer(matched_idx))
            else:
                new_selection = []

            return new_source_data, new_selection

        def get_reset_data():

            df = self.from_range(None, None)

            new_source_data = df.to_dict(orient='list')
            new_source_data['index'] = df.index

            return new_source_data

        @gen.coroutine
        def update_source_data():

            global pending_xrange_update, source_data, selection
            self.plot_source.data = source_data
            self.plot_source.selected.indices = selection
            pending_xrange_update = False

        @without_document_lock
        @gen.coroutine
        def reset_xrange():

            global source_data
            source_data = yield self.thread_pool.submit(get_reset_data)
            doc.add_next_tick_callback(update_source_data)

        @without_document_lock
        @gen.coroutine
        def update_xrange():

            global xrange_change_buffer, source_data, selection
            # print('Updating...')
            source_data, selection = \
                yield self.thread_pool.submit(get_updated_data)
            doc.add_next_tick_callback(update_source_data)

            if xrange_change_buffer is not None:
                doc.add_next_tick_callback(update_xrange)
                xrange_change_buffer = None

        def on_xrange_change(attr, old, new):

            global pending_xrange_update, xrange_change_buffer

            if not pending_xrange_update:
                pending_xrange_update = True
                doc.add_next_tick_callback(update_xrange)
            else:
                # print('buffering...')
                xrange_change_buffer = new

        def on_reset(event):

            global pending_xrange_update

            pending_xrange_update = True

            doc.add_next_tick_callback(reset_xrange)

            if self.plot_source.selected is not None:
                self.plot_source.selected.indices = []

        def on_selected_coord_change(attr, old, new):

            self.collect(new)

            start, end = self.get_range(
                p_plot.x_range.start, p_plot.x_range.end)

            self.update_source(start, end)

            if self.plot_source.selected is not None:
                self.plot_source.selected.indices = []

        def on_selected_points_change(attr, old, new):

            idx_new = np.array(new['1d']['indices'])
            idx_old = np.array(old['1d']['indices'])

            self.plot_data.selected = np.zeros(
                len(self.plot_data.selected), dtype=bool)
            sel_idx_start = self.plot_source.data['index'][np.min(idx_new)]
            sel_idx_end = self.plot_source.data['index'][np.max(idx_new)]
            self.plot_data.selected[np.logical_and(
                self.plot_data.index >= sel_idx_start,
                self.plot_data.index <= sel_idx_end)] = True

            if self.span_coord is not None:

                if len(idx_new) == 0:
                    for s in vlines['span']:
                        s.visible = True
                else:
                    for s in vlines['span']:
                        s.visible = False
                    s_idx = vlines.index[
                        np.logical_and(vlines.index >= min(idx_new),
                                       vlines.index <= max(idx_new))]
                    for s in vlines.loc[s_idx, 'span']:
                        s.visible = True

        circ.data_source.on_change('selected', on_selected_points_change)

        p_plot.x_range.on_change('start', on_xrange_change)
        p_plot.x_range.on_change('end', on_xrange_change)
        p_plot.on_event(Reset, on_reset)

        if self.select_coord is not None:
            multi_select.on_change('value', on_selected_coord_change)

        doc.add_root(layout)
