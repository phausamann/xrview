""" ``xrview.motion`` """

import numpy as np
import pandas as pd

from bokeh.layouts import row, column, widgetbox
from bokeh.models import \
    BoxSelectTool, LassoSelectTool, CategoricalColorMapper, Band
from bokeh.models.widgets import Div
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_notebook, show
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.palettes import d3


class FeatureMapViewer(object):
    """

    Parameters
    ----------
    feature_var :
    data_var :
    sample_dim :
    time_dim :
    axis_dim :
    highlight_coord :
    show_selected_coord :
    figsize :
    """

    def __init__(self, feature_var='Feature', data_var='Acceleration',
                 sample_dim='sample', time_dim='timepoint', axis_dim='axis',
                 highlight_coord=None, show_selected_coord=None,
                 figsize=(900, 600)):

        self.feature_var = feature_var
        self.data_var = data_var
        self.sample_dim = sample_dim
        self.time_dim = time_dim
        self.axis_dim = axis_dim
        self.highlight_coord = highlight_coord
        self.show_selected_coord = show_selected_coord
        self.figsize = figsize

        self.X = None

    def _app(self, doc):

        df = pd.DataFrame(
            {'scatter_x': self.X[self.feature_var][:, 0],
             'scatter_y': self.X[self.feature_var][:, 1]},
            index=self.X.sample)

        scatter_args = dict()

        if self.highlight_coord is not None:
            hc_vals = self.X[self.highlight_coord].isel(
                **{d: 0 for d in self.X[self.highlight_coord].dims
                 if d != self.sample_dim}).values
            df = df.assign(**{self.highlight_coord: hc_vals})
            hc_uvals = np.unique(hc_vals)
            palette = d3['Category10'][len(hc_uvals)]
            color_map = CategoricalColorMapper(
                factors=hc_uvals, palette=palette)
            scatter_args['color'] = {
                'field': self.highlight_coord, 'transform': color_map}
            scatter_args['legend'] = self.highlight_coord

        SCATTER_TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"
        PLOT_TOOLS = "pan,wheel_zoom,reset"
        COLORS = ['red', 'green', 'blue']

        # create the scatter plot
        p_scatter = figure(
            tools=SCATTER_TOOLS, plot_width=int(2/3*self.figsize[0]),
            plot_height=self.figsize[1], toolbar_location='above',
            x_axis_location=None,  y_axis_location=None,
            title=self.feature_var)

        p_scatter.select(BoxSelectTool).select_every_mousemove = False
        p_scatter.select(LassoSelectTool).select_every_mousemove = False

        scatter = p_scatter.scatter(
            x='scatter_x', y='scatter_y', source=ColumnDataSource(df),
            **scatter_args)

        xv_default = np.arange(self.X.sizes[self.time_dim])
        yv_default = np.zeros(self.X.sizes[self.time_dim])
        bv_default = {'x': xv_default, 'lower': yv_default, 'upper': yv_default}

        # create plots for each axis
        p_lines, lines, bands = dict(), dict(), dict()
        x_range, y_range = None, None

        for idx, axis in enumerate(self.X[self.axis_dim].values):

            c = COLORS[np.mod(idx, len(COLORS))]
            w = int(1/3*self.figsize[0])
            h = int(self.figsize[1]/self.X.sizes[self.axis_dim])

            p_lines[axis] = figure(
                toolbar_location='above', tools=PLOT_TOOLS,
                plot_width=w, plot_height=h, y_axis_location='right',
                title=self.data_var + ' (' + str(axis) + ')')

            lines[axis] = p_lines[axis].line(
                xv_default, yv_default, color=c, line_width=2)

            bands[axis] = Band(
                base='x', lower='lower', upper='upper', line_width=1,
                source=ColumnDataSource(bv_default), level='underlay',
                fill_alpha=0.5, line_color=c, fill_color=c)
            p_lines[axis].add_layout(bands[axis])

            if x_range is None:
                x_range = p_lines[axis].x_range
                y_range = p_lines[axis].y_range
            else:
                p_lines[axis].x_range = x_range
                p_lines[axis].y_range = y_range

        layout = row(p_scatter, column(*tuple(p_lines[k] for k in p_lines)))

        if self.show_selected_coord is not None:
            div = Div(text=self.show_selected_coord + ': None',
                      width=self.figsize[0])
            layout = column(layout, widgetbox(div))

        doc.add_root(layout)

        def update(attr, old, new):

            inds = np.array(new['1d']['indices'])

            for axis in p_lines:

                if len(inds) != 0:
                    X_inds = self.X.isel(**{self.sample_dim: inds})
                    X_k = X_inds[self.data_var].sel(**{self.axis_dim: axis})
                    mean = X_k.mean(dim=self.sample_dim).values
                    std = X_k.std(dim=self.sample_dim).values

                    lines[axis].data_source.data['y'] = mean
                    bands[axis].source.data['lower'] = mean - std
                    bands[axis].source.data['upper'] = mean + std

                    if self.show_selected_coord is not None:
                        div.text = self.show_selected_coord + ': ' + \
                            str(np.unique(X_inds[self.show_selected_coord]))

                else:
                    lines[axis].data_source.data['y'] = yv_default
                    bands[axis].source.data['lower'] = bv_default['lower']
                    bands[axis].source.data['upper'] = bv_default['upper']

                    if self.show_selected_coord is not None:
                        div.text = self.show_selected_coord + ': None'

        scatter.data_source.on_change('selected', update)

    def show(self, X, notebook_url):

        self.X = X

        output_notebook()
        handler = FunctionHandler(self._app)
        app = Application(handler)
        app.create_document()
        show(app, notebook_url=notebook_url)
