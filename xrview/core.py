""" ``xrview.core`` """

import abc
import six

import numpy as np
import pandas as pd

from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure
from bokeh.models import HoverTool

from xrview.mappers import map_figures_and_glyphs
from xrview.utils import rsetattr, is_dataarray, is_dataset, clone_models
from xrview.elements import get_glyph
from xrview.palettes import RGB
from xrview.handlers import DataHandler


class BaseLayout(object):
    """ Base class for all layouts. """

    def __init__(self):

        self.doc = None
        self.layout = None

        self.handlers = None
        self.glyph_map = None
        self.figure_map = None
        self.figures = None

        # added
        self.added_figures = []
        self.added_overlays = []
        self.added_overlay_figures = []

    @abc.abstractmethod
    def _make_layout(self):
        """ Make the layout. """

    @abc.abstractmethod
    def show(self):
        """ Show the layout. """

    def _inplace_update(self):
        """ Update the current layout in place. """

        self.doc.roots[0].children[0] = self.layout

    def update_inplace(self, other):
        """ Update this instance with the properties of another layout.

        Parameters
        ----------
        other : xrview.notebook.base.NotebookServer
            The instance that replaces the current instance.
        """
        doc = self.doc
        self.__dict__ = other.__dict__  # TODO: make this safer
        self._make_layout()
        self.doc = doc

        self.doc.add_next_tick_callback(self._inplace_update)

    def copy(self, with_data=False):
        """ Create a copy of this instance.

        Parameters
        ----------
        with_data : bool, default False
            If true, also copy the data.

        Returns
        -------
        new : xrview.notebook.base.NotebookServer
            The copied object.
        """
        from copy import copy

        new = self.__new__(type(self))
        new.__dict__ = {k: (copy(v) if (k != 'data' or with_data) else v)
                        for k, v in self.__dict__.items()}

        return new


class BasePlot(BaseLayout):
    """ Base class for all plots.

    Parameters
    ----------
    data : xarray DataArray or Dataset
        The data to display.

    x : str
        The name of the dimension in ``data`` that contains the x-axis values.

    glyph : str, default 'line'
        The glyph to use for plotting.

    overlay : 'dims' or 'data_vars', default 'dims'
        If 'dims', make one figure for each data variable and overlay the
        dimensions. If 'data_vars', make one figure for each dimension and
        overlay the data variables. In the latter case, all variables must
        have the same dimensions.

    tooltips : dict, optional
        Names of tooltips mapping to glyph properties or source columns, e.g.
        {'datetime': '@index{%F %T.%3N}'}.

    tools : str, optional
        bokeh tool string.

    figsize : iterable, default (900, 400)
        The size of the figure in pixels.

    ncols : int, default 1
        The number of columns of the layout.

    palette : iterable, optional
        The palette to use when overlaying multiple glyphs.

    ignore_index : bool, default False
        If True, replace the x-axis values of the data by an appropriate
        evenly spaced index.
    """
    def __init__(self, data, x, overlay='dims', glyph='line', tooltips=None,
                 tools=None, figsize=(900, 400), ncols=1, palette=None,
                 ignore_index=False, **fig_kwargs):

        super(BasePlot, self).__init__()

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
        if overlay in ('dims', 'data_vars'):
            self.overlay = overlay
        else:
            raise ValueError('overlay must be "dim" or "var"')

        if isinstance(glyph, str):
            self.glyph = get_glyph(glyph)
        else:
            self.glyph = glyph

        self.tooltips = tooltips

        # layout parameters
        self.ncols = ncols
        self.figsize = figsize
        self.fig_kwargs = fig_kwargs

        if palette is None:
            self.palette = RGB
        else:
            self.palette = palette

        if tools is None:
            self.tools = 'pan,wheel_zoom,box_zoom,xbox_select,reset'
            if self.tooltips is not None:
                self.tools += ',hover'
        else:
            self.tools = tools

        self.ignore_index = ignore_index

    def _collect_data(self, data, coords=None):
        """ Base method for _collect. """

        plot_data = dict()

        for v in list(data.data_vars) + (coords or []):
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                plot_data[v] = data[v].values
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for d in data[dim].values:
                    plot_data[v + '_' + str(d)] = \
                        data[v].sel(**{dim: d}).values
            else:
                raise ValueError(v + ' has too many dimensions')

        plot_data['selected'] = np.zeros(data.sizes[self.x], dtype=bool)

        # TODO: doesn't work for irregularly sampled data
        if self.ignore_index:
            if isinstance(data.indexes[self.x], pd.DatetimeIndex):
                if data.indexes[self.x].freq is None:
                    freq = data.indexes[self.x][1] - data.indexes[self.x][0]
                else:
                    freq = data.indexes[self.x].freq
                index = pd.DatetimeIndex(
                    start=0, freq=freq, periods=data.sizes[self.x])
            else:
                index = np.arange(data.sizes[self.x])
        else:
            index = data[self.x]

        return pd.DataFrame(plot_data, index=index)

    def _collect(self, hooks=None, coords=None):
        """ Collect plottable data in a pandas DataFrame. """
        data = self.data
        if hooks is not None:
            for h in hooks:
                data = h(data)
        return self._collect_data(data, coords=coords)

    def _attach_elements(self):
        """ Attach additional elements to this layout. """
        for element in self.added_figures + self.added_overlays:
            element.attach(self)

    def _make_handlers(self):
        """ Make handlers. """
        self.handlers = [DataHandler(self._collect())]
        for element in self.added_figures + self.added_overlays:
            self.handlers.append(element.handler)

    def _make_maps(self):
        """ Make the figure and glyph map. """
        self.figure_map, self.glyph_map = map_figures_and_glyphs(self.data,
                                                                 self.x,
                                                                 self.handlers,
                                                                 self.glyph,
                                                                 self.overlay,
                                                                 self.fig_kwargs,
                                                                 self.added_figures,
                                                                 self.added_overlays,
                                                                 self.added_overlay_figures,
                                                                 self.palette)

    def _make_figures(self):
        """ Make figures. """

        # TODO: check if we can put this in self.figure_map.figure
        self.figures = []

        for _, f in self.figure_map.iterrows():

            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                f.fig_kwargs['x_axis_type'] = 'datetime'

            # link x axis ranges
            if len(self.figures) > 0:
                f.fig_kwargs['x_range'] = self.figures[0].x_range

            # TODO: link y axis ranges if requested

            width = self.figsize[0]//self.ncols
            height = self.figsize[1]//self.figure_map.shape[0]*self.ncols
            self.figures.append(figure(
                plot_width=width, plot_height=height, tools=self.tools,
                **f.fig_kwargs))

            # TODO: make this configurable
            self.figures[-1].xgrid.visible = False

    def _add_glyphs(self):
        """ Add glyphs. """
        for g_idx, g in self.glyph_map.iterrows():
            glyph_kwargs = clone_models(g.glyph_kwargs)
            getattr(self.figures[g.figure], g.method)(
                source=g.handler.source, **glyph_kwargs)

    def _add_tooltips(self):
        """ Add tooltips. """
        if self.tooltips is not None:
            tooltips = [(k, v) for k, v in self.tooltips.items()]
            for f in self.figures:
                f.select(HoverTool).tooltips = tooltips
                if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                    f.select(HoverTool).formatters = {'index': 'datetime'}

    def _finalize_layout(self):
        """ Finalize layout. """
        self.layout = gridplot(self.figures, ncols=self.ncols)

    def _make_layout(self):
        """ Make the layout. """
        self._attach_elements()
        self._make_handlers()
        self._make_maps()
        self._make_figures()
        self._add_glyphs()
        self._add_tooltips()
        self._finalize_layout()

    def add_figure(self, element):
        """ Add a figure to the layout.

        Parameters
        ----------
        element : Element
            The element to add as a figure.
        """
        self.added_figures.append(element)

    def add_overlay(self, element, onto=None):
        """ Add an overlay to a figure in the layout.

        Parameters
        ----------
        element : Element
            The element to overlay.

        onto : str or int, optional
            Title or index of the figure on which the element will be
            overlaid. By default, the element is overlaid on all figures.
        """
        self.added_overlays.append(element)
        self.added_overlay_figures.append(onto)

    def modify_figure(self, idx, modfiers):
        """ Modify the attributes of a figure.

        Parameters
        ----------
        idx : int
            The index of the figure to modify.

        modfiers : dict
            The attributes to modify. Keys can reference sub-attributes,
            e.g. 'xaxis.axis_label'.
        """
        f = self.figures[idx]
        for m in modfiers:
            if self.doc is not None:
                mod_func = lambda: rsetattr(f, m, modfiers[m])
                self.doc.add_next_tick_callback(mod_func)
            else:
                rsetattr(f, m, modfiers[m])


class BaseViewer(BasePlot):
    """ Base class for interactive viewers. """

    def __init__(self, *args, **kwargs):

        super(BaseViewer, self).__init__(*args, **kwargs)
        self.added_interactions = []

    def on_selected_points_change(self, attr, old, new):
        """ Callback for selection event. """

        idx_new = np.array(new['1d']['indices'])

        for h in self.handlers:
            # find the handler whose source emitted the selection change
            if h.source.selected._id == new._id:
                sel_idx_start = h.source.data['index'][np.min(idx_new)]
                sel_idx_end = h.source.data['index'][np.max(idx_new)]
                break
        else:
            raise ValueError('The source that emitted the selection change '
                             'was not found in this object\'s handlers.')

        # Update the selection of each handler
        for h in self.handlers:
            h.data.selected = np.zeros(len(h.data.selected), dtype=bool)
            h.data.loc[np.logical_and(
                h.data.index >= sel_idx_start,
                h.data.index <= sel_idx_end), 'selected'] = True

        # Update handlers
        self.update_handlers()

    def update_handlers(self):
        """ Update handlers. """

        for h in self.handlers:
            h.update_data()
            self.doc.add_next_tick_callback(h.update_source)

    def _make_handlers(self):
        """ Make handlers. """

        # default handler
        self.handlers = [DataHandler(self._collect())]

        for element in self.added_figures + self.added_overlays:
            self.handlers.append(element.handler)

    def _update_handlers(self, hooks=None):
        """ Update handlers. """

        if hooks is None:
            # TODO: check if this breaks co-dependent hooks
            hooks = [i.collect_hook for i in self.added_interactions]

        element_list = self.added_figures + self.added_overlays

        for h_idx, h in enumerate(self.handlers):

            if h_idx == 0:
                h.data = self._collect(hooks)
            else:
                h.data = element_list[h_idx-1]._collect(hooks)

            h.update_data()
            h.update_source()

            if h.source.selected is not None:
                h.source.selected.indices = []

    def _attach_elements(self):
        """ Attach additional elements to this viewer. """

        # TODO: rename

        for element in self.added_figures + self.added_overlays:
            element.attach(self)

        for interaction in self.added_interactions:
            interaction.attach(self)

    def _add_glyphs(self):
        """ Add glyphs. """

        for g_idx, g in self.glyph_map.iterrows():

            glyph_kwargs = clone_models(g.glyph_kwargs)

            glyph = getattr(self.figures[g.figure], g.method)(
                source=g.handler.source, **glyph_kwargs)

            if g.method != 'circle':
                circle = self.figures[g.figure].circle(
                    source=g.handler.source, size=0,
                    **{'x': glyph_kwargs[g.x_arg],
                       'y': glyph_kwargs[g.y_arg]})
                circle.data_source.on_change(
                    'selected', self.on_selected_points_change)
            else:
                glyph.data_source.on_change(
                    'selected', self.on_selected_points_change)

    def _add_callbacks(self):
        """ Add callbacks. """

    def _finalize_layout(self):
        """ Finalize layout. """

        self.layout = gridplot(self.figures, ncols=self.ncols)

        interactions = {
            loc: [i.layout_hook() for i in self.added_interactions if
                  i.location == loc]
            for loc in ['above', 'below', 'left', 'right']
        }

        layout_v = []
        layout_h = []

        if len(interactions['above']) > 0:
            layout_v.append(row(*interactions['above']))
        if len(interactions['left']) > 0:
            layout_h.append(column(*interactions['left']))
        layout_h.append(self.layout)
        if len(interactions['right']) > 0:
            layout_h.append(column(*interactions['right']))
        layout_v.append(row(*layout_h))
        if len(interactions['below']) > 0:
            layout_v.append(row(*interactions['below']))

        self.layout = column(layout_v)

    def _make_layout(self):
        """ Make the layout. """
        self._attach_elements()
        self._make_handlers()
        self._make_maps()
        self._make_figures()
        self._add_glyphs()
        self._add_tooltips()
        self._add_callbacks()
        self._finalize_layout()

    def add_interaction(self, interaction):
        """ Add an interaction to the layout.

        Parameters
        ----------
        interaction : Interaction
            The interaction to add.
        """

        self.added_interactions.append(interaction)
