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


class TimeseriesDataHandler(object):

    def __init__(self, data, width, sample_dim='sample', axis_dim='axis',
                 select_coord=None, span_coord=None):

        self.data = data
        self.width = width
        self.sample_dim = sample_dim
        self.axis_dim = axis_dim
        self.select_coord = select_coord
        self.span_coord = span_coord

        self.points_per_pixel = 5

        self.plot_data = None
        self.render_data = None
        self.span_data = None

    def collect(self, coord_vals=None):

        if coord_vals is not None:
            sel_idx = np.zeros(self.data.sizes[self.sample_dim], dtype=bool)
            for c in coord_vals:
                sel_idx = sel_idx | (self.data[self.select_coord].values == c)
            plot_data = {
                axis: self.data.sel(**{self.axis_dim: axis}).values[sel_idx]
                for axis in self.data[self.axis_dim].values}
        else:
            plot_data = {
                axis: self.data.sel(**{self.axis_dim: axis}).values
                for axis in self.data[self.axis_dim].values}

        self.plot_data = pd.DataFrame(plot_data)

    def get_range(self, plot_source, start=None, end=None):

        # get new start and end
        if start is not None:
            if start < plot_source.data['index'][0]:
                start = max(self.plot_data.index[0], int(start))
            else:
                start = int(start)
        elif len(plot_source.data['index']) > 0:
            start = plot_source.data['index'][0]
        else:
            start = self.plot_data.index[0]

        if end is not None:
            if end > plot_source.data['index'][-1]:
                end = min(self.plot_data.index[-1], int(end))
            else:
                end = int(end)
        elif len(plot_source.data['index']) > 0:
            end = plot_source.data['index'][-1]
        else:
            end = self.plot_data.index[-1]

        return start, end

    def from_range(self, start, end):

        if start is None:
            start = 0

        if end is None:
            end = self.plot_data.shape[0]

        factor = int(np.ceil((end-start)/(self.points_per_pixel*self.width)))

        return self.plot_data.iloc[start:end:factor]

    def update_source(self, plot_source, start, end):

        # get data to render
        df = self.from_range(start, end)

        # update source selection
        if plot_source.selected is not None \
                and len(plot_source.selected.indices) > 0:
            sel_idx = plot_source.selected.indices
            sel_idx_start = plot_source.data['index'][np.min(sel_idx)]
            sel_idx_end = plot_source.data['index'][np.max(sel_idx)]
            plot_source.selected.indices = list(np.arange(
                df.index.get_loc(sel_idx_start, method='nearest'),
                df.index.get_loc(sel_idx_end, method='nearest')))

        # update source data
        df_dict = df.to_dict(orient='list')
        df_dict['index'] = df.index
        plot_source.data = df_dict


class BaseViewer(object):
    """"""

    def show(self, data, notebook_url, port=0):

        self.handler.data = data

        output_notebook()
        app = Application(FunctionHandler(self._app))
        app.create_document()
        show_app(app, None, notebook_url=notebook_url, port=port)


class TimeseriesViewer(BaseViewer):
    """

    Parameters
    ----------
    sample_dim :
    axis_dim :
    select_coord :
    span_coord :
    """

    def __init__(self, sample_dim='sample', axis_dim='axis', select_coord=None,
                 span_coord=None, figsize=(700, 500)):

        self.sample_dim = sample_dim
        self.axis_dim = axis_dim
        self.select_coord = select_coord
        self.span_coord = span_coord
        self.figsize = figsize

        self.handler = TimeseriesDataHandler(
            None, self.figsize[0], sample_dim=self.sample_dim,
            axis_dim=self.axis_dim, select_coord=self.select_coord,
            span_coord=self.span_coord
        )

    def _app(self, doc):

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
                (v, v) for v in np.unique(self.handler.data[self.select_coord])]
            multi_select = MultiSelect(
                title=self.select_coord, value=[options[0][0]], options=options)
            multi_select.size = len(options)
            layout = row(p_plot, multi_select)
            # create data source
            self.handler.collect([options[0][1]])
        else:
            layout = p_plot
            # create data source
            self.handler.collect()

        self.plot_source = ColumnDataSource(self.handler.plot_data)
        self.handler.update_source(
            self.plot_source, *self.handler.get_range(self.plot_source))

        # create vertical spans
        if self.span_coord is not None:

            vlines = []
            locations = np.where(
                np.diff(np.array(self.handler.data[self.span_coord],
                                 dtype=float)) > 0)[0]

            for l in locations:
                vlines.append(Span(
                    location=l, dimension='height', line_color='grey',
                    line_width=1, line_alpha=0.5))
                p_plot.add_layout(vlines[-1])

            vlines = pd.DataFrame({'span': vlines}, index=locations)

        for idx, axis in enumerate(self.handler.data.axis.values):
            c = COLORS[np.mod(idx, len(COLORS))]
            p_plot.line(x='index', y=axis, source=self.plot_source,
                        line_color=c, line_alpha=0.6)
            circ = p_plot.circle(
                x='index', y=axis, source=self.plot_source, color=c, size=0)

        def on_xrange_change(attr, old, new):

            start, end = self.handler.get_range(self.plot_source, **{attr: new})

            if start == self.plot_source.data['index'][0] \
                    and end == self.plot_source.data['index'][-1]:
                return

            self.handler.update_source(self.plot_source, start, end)

        def on_selected_coord_change(attr, old, new):

            self.handler.collect(new)

            start, end = self.handler.get_range(
                self.plot_source, p_plot.x_range.start, p_plot.x_range.end)

            self.handler.update_source(self.plot_source, start, end)

            if self.plot_source.selected is not None:
                self.plot_source.selected.indices = []

        def on_selected_points_change(attr, old, new):

            idx_new = np.array(new['1d']['indices'])
            idx_old = np.array(old['1d']['indices'])

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

        if self.select_coord is not None:
            multi_select.on_change('value', on_selected_coord_change)

        doc.add_root(layout)
