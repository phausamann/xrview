""" ``xrview.motion`` """

import numpy as np
import pandas as pd

from bokeh.layouts import row, column, gridplot
from bokeh.models import (
    BoxSelectTool, LassoSelectTool, CategoricalColorMapper,
    LinearColorMapper, Band, Spacer, HoverTool, ColumnDataSource, CDSView,
    GroupFilter)
from bokeh.plotting import figure
from bokeh.palettes import d3, viridis
from bokeh.events import Reset

from .base import BaseViewer


class FeatureMapViewer(BaseViewer):
    """ Viewer for 2D embeddings of high-dimensional data.

    Parameters
    ----------
    data : xarray Dataset
        The data to display.

    feature_var : str, default 'Feature'
        The name of the 2D feature variable.

    data_var : str, default 'Acceleration'
        The name of the variable containing the raw data.

    sample_dim : str, default 'sample'
        The name of the sample dimension.

    time_dim :str, default 'timepoint'
        The name of the time dimension.

    axis_dim :str, default 'axis'
        The name of the axis dimension.

    highlight_coord : str, optional
        The name of the coordinate to be highlighted in different colors in
        the feature map.

    marker_coord : str, optional
        The name of the coordinate to be highlighted with different markers in
        the feature map.

    bar_coord : str, optional
        The name of the coordinate whose distribution to display as a
        histogram below the feature map.

    hover_coords : list of str, optional
        The names of the coordinates to show as tooltips when hovering over
        the feature map.

    show_orientation : bool, default False
        If true, show the distribution of orientations below the feature map.

    figsize : tuple, default (900, 600)
        The size of the figure.
    """

    def __init__(self, data, feature_var='Feature', data_var='Acceleration',
                 sample_dim='sample', time_dim='timepoint', axis_dim='axis',
                 highlight_coord=None, marker_coord=None, size_coord=None,
                 bar_coord=None, hover_coords=None, show_orientation=False,
                 figsize=(900, 600)):

        self.data = data

        self.feature_var = feature_var
        self.data_var = data_var
        self.sample_dim = sample_dim
        self.time_dim = time_dim
        self.axis_dim = axis_dim
        self.highlight_coord = highlight_coord
        self.marker_coord = marker_coord
        self.size_coord = size_coord
        self.bar_coord = bar_coord
        self.hover_coords = hover_coords
        self.show_orientation = show_orientation
        self.figsize = figsize

    def _as_1d(self, coord):

        return self.data[coord].isel(
            **{d: 0 for d in self.data[coord].dims
               if d != self.sample_dim}).values

    def make_app(self, doc):

        df_scatter = pd.DataFrame(
            {'scatter_x': self.data[self.feature_var][:, 0],
             'scatter_y': self.data[self.feature_var][:, 1]},
            index=self.data.sample)

        scatter_args = dict()

        SCATTER_TOOLS = \
            'pan,wheel_zoom,box_select,lasso_select,tap,hover,reset,save'
        PLOT_TOOLS = 'pan,wheel_zoom,reset'
        COLORS = ['red', 'green', 'blue']
        MARKERS = ['asterisk', 'circle', 'circle_cross', 'circle_x', 'cross',
                   'diamond', 'diamond_cross', 'hex', 'inverted_triangle',
                   'square', 'square_x', 'square_cross', 'triangle', 'x', '*',
                   '+', 'o', 'ox', 'o+']

        # create the scatter plot
        p_scatter = figure(
            tools=SCATTER_TOOLS, plot_width=int(2/3*self.figsize[0]),
            plot_height=self.figsize[1], toolbar_location='above',
            x_axis_location=None,  y_axis_location=None,
            title=self.feature_var)

        p_scatter.select(BoxSelectTool).select_every_mousemove = False
        p_scatter.select(LassoSelectTool).select_every_mousemove = False
        p_scatter.xgrid.grid_line_color = None
        p_scatter.ygrid.grid_line_color = None

        # add hover tooltips
        if self.hover_coords is not None:
            p_scatter.select(HoverTool).tooltips = [
                (c, '@' + c) for c in self.hover_coords]
            for c in self.hover_coords:
                c_vals = self._as_1d(c)
                df_scatter = df_scatter.assign(**{c: c_vals})

        # highlight scatter points based on coordinate
        if self.highlight_coord is not None:

            hc_vals = self._as_1d(self.highlight_coord)

            df_scatter = df_scatter.assign(**{self.highlight_coord: hc_vals})
            hc_uvals = np.unique(hc_vals)

            if np.issubdtype(hc_uvals.dtype, np.number):
                palette = viridis(len(hc_uvals))
                color_map = LinearColorMapper(
                    low=np.min(hc_uvals), high=np.max(hc_uvals),
                    palette=palette)
            else:
                if len(hc_uvals) <= 10:
                    # hack for only two catergories
                    palette = d3['Category10'][max(len(hc_uvals), 3)]
                elif len(hc_uvals) <= 20:
                    palette = d3['Category20'][len(hc_uvals)]
                else:
                    palette = viridis[len(hc_uvals)]
                color_map = CategoricalColorMapper(
                    factors=hc_uvals, palette=palette)

            scatter_args['color'] = {
                'field': self.highlight_coord, 'transform': color_map}
            scatter_args['legend'] = self.highlight_coord

        # resize scatter points based on coordinate
        if self.size_coord is not None:
            df_scatter = df_scatter.assign(
                **{self.size_coord: self._as_1d(self.size_coord)})
            scatter_args['size'] = self.size_coord
        # change markers based on coordinate
        if self.marker_coord is not None:
            df_scatter = df_scatter.assign(
                **{self.marker_coord: self._as_1d(self.marker_coord)})
            source = ColumnDataSource(df_scatter)
            for i_m, m in enumerate(np.unique(df_scatter[self.marker_coord])):
                view = CDSView(source=source, filters=[GroupFilter(
                    column_name=self.marker_coord, group=m)])
                scatter = p_scatter.scatter(
                    x='scatter_x', y='scatter_y', source=source,
                    view=view, marker=MARKERS[i_m], **scatter_args)
        else:
            source = ColumnDataSource(df_scatter)
            scatter = p_scatter.scatter(
                x='scatter_x', y='scatter_y', source=source, **scatter_args)

        xv_default = np.arange(self.data.sizes[self.time_dim])
        yv_default = np.zeros(self.data.sizes[self.time_dim])
        bv_default = {'x': xv_default, 'lower': yv_default, 'upper': yv_default}

        # create plots for each axis
        p_lines, lines, bands = dict(), dict(), dict()
        x_range, y_range = None, None

        for idx, axis in enumerate(self.data[self.axis_dim].values):

            c = COLORS[np.mod(idx, len(COLORS))]
            w = int(self.figsize[0]/3)
            h = int(self.figsize[1] / self.data.sizes[self.axis_dim])

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

        layout = gridplot(
            [p_scatter, column(*[p_lines[k] for k in p_lines])], ncols=2)

        # add bar plot below
        if self.bar_coord is not None:

            bc_xvals, bc_counts = np.unique(
                self.data[self.bar_coord], return_counts=True)

            p_bar = figure(
                x_range=bc_xvals, toolbar_location=None, y_range=(0, 1),
                plot_width=p_scatter.plot_width, plot_height=self.figsize[1]//3,
                y_axis_location='right', title=self.bar_coord + ' count')

            p_bar.xgrid.grid_line_color = None
            p_bar.xaxis.major_label_orientation = np.pi / 4
            bar = p_bar.vbar(x=bc_xvals, top=np.ones(len(bc_xvals)),
                             width=0.9)

            if not self.show_orientation:
                spacer = Spacer(
                    width=int(self.figsize[0]/3), height=int(self.figsize[1]/3))
                layout = column(layout, row(p_bar, spacer))

        # add orientation plot
        if self.show_orientation:

            X_data = self.data[self.data_var]
            orientations = np.degrees(np.arctan(
                X_data.sel(**{self.axis_dim: 'z'}).mean(dim=self.time_dim) /
                X_data.sel(**{self.axis_dim: 'x'}).mean(dim=self.time_dim)))

            hv_default, ev_default = np.histogram(
                orientations, density=True, bins=50)

            if self.bar_coord is None:

                p_hist = figure(
                    toolbar_location=None, plot_width=p_scatter.plot_width,
                    plot_height=self.figsize[1]//3, y_axis_location='right',
                    x_range=(-180, 180), title='Orientation')

                spacer = Spacer(
                    width=int(self.figsize[0]/3), height=int(self.figsize[1]/3))

                layout = column(layout, row(p_hist, spacer))

            else:

                p_hist = figure(
                    width=int(self.figsize[0]/3), height=int(self.figsize[1]/3),
                    toolbar_location=None, x_range=(-180, 180),
                    y_axis_location='right', title='Orientation (degrees)')

                layout = column(layout, row(p_bar, p_hist))

            p_hist.y_range.start = 0
            hist = p_hist.quad(
                top=hv_default, bottom=0, left=ev_default[:-1],
                right=ev_default[1:])

        doc.add_root(layout)

        # callback for scatter point selection
        def update(attr, old, new):

            inds = np.array(new['1d']['indices'])

            if len(inds) != 0:

                X_inds = self.data.isel(**{self.sample_dim: inds})

                for axis in p_lines:
                    X_k = X_inds[self.data_var].sel(**{self.axis_dim: axis})
                    mean = X_k.mean(dim=self.sample_dim).values
                    std = X_k.std(dim=self.sample_dim).values
                    lines[axis].data_source.data['y'] = mean
                    bands[axis].source.data['lower'] = mean - std
                    bands[axis].source.data['upper'] = mean + std

                if self.bar_coord is not None:
                    bc_ind_counts = np.array(
                        [np.sum(X_inds[self.bar_coord] == v)
                         for v in bc_xvals])
                    bar.data_source.data['top'] = bc_ind_counts/bc_counts

                if self.show_orientation:
                    orientations = np.degrees(np.arctan(
                        X_inds[self.data_var].sel(
                            **{self.axis_dim: 'z'}).mean(dim=self.time_dim) /
                        X_inds[self.data_var].sel(
                            **{self.axis_dim: 'x'}).mean(dim=self.time_dim)))
                    hv, ev = np.histogram(orientations, density=True, bins=50)
                    hist.data_source.data['top'] = hv
                    hist.data_source.data['left'] = ev[:-1]
                    hist.data_source.data['right'] = ev[1:]

            else:

                reset(None)

        # callback for scatter plot reset
        def reset(event):

            for axis in p_lines:
                lines[axis].data_source.data['y'] = yv_default
                bands[axis].source.data['lower'] = bv_default['lower']
                bands[axis].source.data['upper'] = bv_default['upper']

            if self.bar_coord is not None:
                bar.data_source.data['top'] = np.ones(len(bc_xvals))

            if self.show_orientation:
                hist.data_source.data['top'] = hv_default
                hist.data_source.data['left'] = ev_default[:-1]
                hist.data_source.data['right'] = ev_default[1:]

        scatter.data_source.on_change('selected', update)
        p_scatter.on_event(Reset, reset)
