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


class BaseViewer(object):
    """"""

    def show(self, X, notebook_url, port=0):

        self.X = X

        output_notebook()
        handler = FunctionHandler(self._app)
        app = Application(handler)
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

        self.X = None

    def _collect(self, coord_vals=None):

        if coord_vals is not None:
            sel_idx = np.zeros(self.X.sizes[self.sample_dim], dtype=bool)
            for c in coord_vals:
                sel_idx = sel_idx | (self.X[self.select_coord].values == c)
            plot_data = {
                axis: self.X.sel(**{self.axis_dim: axis}).values[sel_idx]
                for axis in self.X[self.axis_dim].values}
        else:
            plot_data = {
                axis: self.X.sel(**{self.axis_dim: axis}).values
                for axis in self.X[self.axis_dim].values}

        return pd.DataFrame(plot_data)

    def _from_range(self, plot_data, start=None, end=None):

        if start is None:
            start = 0

        if end is None:
            end = plot_data.shape[0]

        factor = int(np.ceil((end-start)/(5*self.figsize[0])))
        return plot_data.iloc[start:end:factor]

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
            options = [(v, v) for v in np.unique(self.X[self.select_coord])]
            multi_select = MultiSelect(
                title=self.select_coord, value=[options[0][0]],
                options=options)
            multi_select.size = len(options)
            layout = row(p_plot, multi_select)
            # create data source
            plot_data = self._collect([options[0][1]])
        else:
            layout = p_plot
            # create data source
            plot_data = self._collect()

        plot_source = ColumnDataSource(
            self._from_range(plot_data, plot_data.index[0], plot_data.index[-1])
        )

        # create vertical spans
        if self.span_coord is not None:

            vlines = []
            locations = np.where(
                np.diff(np.array(self.X[self.span_coord], dtype=float)) > 0)[0]

            for l in locations:
                vlines.append(Span(
                    location=l, dimension='height', line_color='grey',
                    line_width=1, line_alpha=0.5))
                p_plot.add_layout(vlines[-1])

            vlines = pd.DataFrame({'span': vlines}, index=locations)

        for idx, axis in enumerate(self.X.axis.values):
            c = COLORS[np.mod(idx, len(COLORS))]
            p_plot.line(x='index', y=axis, source=plot_source, line_color=c,
                        line_alpha=0.6)
            circ = p_plot.circle(x='index', y=axis, source=plot_source, color=c,
                                 size=0)

        def on_xrange_change(attr, old, new):

            if attr == 'start':
                if new < plot_source.data['index'][0]:
                    start = max(plot_data.index[0], int(new))
                else:
                    start = int(new)
                if start == plot_source.data['index'][0]:
                    return
                end = plot_source.data['index'][-1]
            elif attr == 'end':
                if new > plot_source.data['index'][-1]:
                    end = min(plot_data.index[-1], int(new))
                else:
                    end = int(new)
                if end == plot_source.data['index'][-1]:
                    return
                start = plot_source.data['index'][0]
            else:
                raise ValueError('Unexpected callback attribute.')

            df = self._from_range(plot_data, start, end)
            df_dict = df.to_dict(orient='list')
            df_dict['index'] = df.index
            plot_source.data = df_dict

        def on_selected_coord_change(attr, old, new):

            nonlocal plot_data

            plot_data = self._collect(new)
            start = max(plot_data.index[0], int(p_plot.x_range.start))
            end = min(plot_data.index[-1], int(p_plot.x_range.end))
            df = self._from_range(plot_data, start, end)
            df_dict = df.to_dict(orient='list')
            df_dict['index'] = df.index
            plot_source.data = df_dict

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
