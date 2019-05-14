""" ``xrview.core`` """

import abc

from functools import partial

import numpy as np
import pandas as pd

from tornado import gen

from bokeh.document import Document, without_document_lock
from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure
from bokeh.models import HoverTool, FactorRange, Glyph, Spacer
from bokeh.io import export_png, export_svgs

from svgutils.compose import Figure, SVG

from xrview.mappers import map_figures_and_glyphs, _get_overlay_figures
from xrview.utils import rsetattr, is_dataarray, is_dataset, clone_models
from xrview.elements import get_glyph, Element, InteractiveElement
from xrview.palettes import RGB
from xrview.handlers import DataHandler, InteractiveDataHandler


class BasePanel(object):
    """ Base class for all panels. """

    def __init__(self):

        self.layout = None
        self.doc = None

        self.handlers = None
        self.glyph_map = None
        self.figure_map = None
        self.figures = None

        # TODO: improve
        self.added_figures = []
        self.added_overlays = []
        self.added_overlay_figures = []
        self.added_annotations = []
        self.added_annotation_figures = []
        self.modifiers = []

    @abc.abstractmethod
    def make_layout(self):
        """ Make the layout. """

    @abc.abstractmethod
    def show(self):
        """ Show the layout. """

    def _export(self, func, backend, filename):
        """"""
        backends = []
        for f in self.figures:
            if hasattr(f, 'output_backend'):
                backends.append(f.output_backend)
                f.output_backend = backend
        func(self.layout, filename=filename)
        for f in self.figures:
            if hasattr(f, 'output_backend'):
                f.output_backend = backends.pop(0)

    def export(self, filename, mode='auto'):
        """ Export the layout as as png or svg file.

        Parameters
        ----------
        filename : str
            The path of the exported file.

        mode : 'auto', 'png' or 'svg', default 'auto'
            Whether to export as png or svg. Note that multi-figure layouts
            will be split into individual files for each figure in the svg
            mode. 'auto' will try to determine the mode automatically from
            the file extension.
        """
        if self.layout is None:
            self.make_layout()

        if mode == 'auto':
            mode = filename.split('.')[-1]
            if mode not in ('png', 'svg'):
                raise ValueError('Could not determine mode from file '
                                 'extension')

        if mode == 'png':
            # TODO: TEST
            for c in self.layout.children:
                if hasattr(c, 'toolbar_location'):
                    c.toolbar_location = None
            self._export(export_png, 'canvas', filename)
            # TODO: TEST
            for c in self.layout.children:
                if hasattr(c, 'toolbar_location'):
                    c.toolbar_location = self.toolbar_location
        elif mode == 'svg':
            self._export(export_svgs, 'svg', filename)
        else:
            raise ValueError('Unrecognized mode')

    def make_doc(self):
        """ Make the document. """
        self.doc = Document()
        self.doc.add_root(row(self.layout))

    def copy(self, with_data=False):
        """ Create a copy of this instance.

        Parameters
        ----------
        with_data : bool, default False
            If true, also copy the data.

        Returns
        -------
        new : xrview.core.BasePanel
            The copied object.
        """
        from copy import copy

        new = self.__new__(type(self))
        new.__dict__ = {k: (copy(v) if (k != 'data' or with_data) else v)
                        for k, v in self.__dict__.items()}

        return new


class GridPlot(BasePanel):
    """ """

    def __init__(self, panels, ncols=1, toolbar_location='above'):

        self.panels = panels
        self.ncols = ncols
        self.toolbar_location = toolbar_location

        self.make_layout()

    def make_layout(self):

        self.figures = []
        for p in self.panels:
            if p.layout is None:
                p.make_layout()
            # TODO: TEST
            for c in p.layout.children:
                if hasattr(c, 'toolbar_location'):
                    c.toolbar_location = None
            self.figures += p.figures

        self.layout = gridplot(
            [p.layout for p in self.panels], ncols=self.ncols,
            toolbar_location=self.toolbar_location)

        return self.layout


class SpacerPanel(BasePanel):

    def __init__(self):

        self.figures = [Spacer()]

        self.make_layout()

    def make_layout(self):

        self.layout = column(*self.figures)

        return self.layout


class BasePlot(BasePanel):
    """ Base class for all plots.

    Parameters
    ----------
    data : xarray DataArray or Dataset
        The data to display.

    x : str
        The name of the dimension in ``data`` that contains the x-axis values.

    glyphs : str, BaseGlyph or iterable, default 'line'
        The glyph to use for plotting.

    figsize : iterable, default (900, 400)
        The size of the figure in pixels.

    ncols : int, default 1
        The number of columns of the layout.

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

    palette : iterable, optional
        The palette to use when overlaying multiple glyphs.

    ignore_index : bool, default Falseh
        If True, replace the x-axis values of the data by an appropriate
        evenly spaced index.
    """
    element_type = Element
    handler_type = DataHandler
    default_tools = 'pan,wheel_zoom,save,reset,'

    def __init__(self,
                 data,
                 x,
                 overlay='dims',
                 coords=None,
                 glyphs='line',
                 title=None,
                 share_y=False,
                 tooltips=None,
                 tools=None,
                 toolbar_location='above',
                 figsize=(900, 400),
                 ncols=1,
                 palette=None,
                 ignore_index=False,
                 **fig_kwargs):

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
            raise ValueError('overlay must be "dims" or "data_vars"')

        self.coords = coords

        if isinstance(glyphs, str):
            self.glyphs = [get_glyph(glyphs)]
        else:
            try:
                iter(glyphs)
            except TypeError:
                self.glyphs = [glyphs]
            else:
                self.glyphs = [g for g in glyphs]

        self.tooltips = tooltips

        # layout parameters
        self.title = title
        self.share_y = share_y
        self.ncols = ncols
        self.figsize = figsize
        self.fig_kwargs = fig_kwargs

        if palette is None:
            self.palette = RGB
        else:
            self.palette = palette

        if tools is None:
            self.tools = self.default_tools
            if self.tooltips is not None:
                self.tools += 'hover,'
        else:
            self.tools = tools

        self.toolbar_location = toolbar_location

        self.ignore_index = ignore_index

    def _collect_data(self, data, coords=None):
        """ Base method for _collect. """

        plot_data = dict()

        if coords is True:
            coords = [c for c in data.coords
                      if self.x in data[c].dims and c != self.x]

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
            if isinstance(data.indexes[self.x], pd.MultiIndex):
                index = [tuple(str(i) for i in idx)
                         for idx in self.data.indexes[self.x].tolist()]
                for n in data.indexes[self.x].names:
                    plot_data[n] = data.indexes[self.x].get_level_values(n)
            else:
                index = data[self.x].values

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
        self.handlers = [self.handler_type(self._collect(coords=self.coords))]
        for element in self.added_figures + self.added_overlays:
            self.handlers.append(element.handler)

    def _make_maps(self):
        """ Make the figure and glyph map. """
        self.figure_map, self.glyph_map = map_figures_and_glyphs(
            self.data, self.x, self.handlers, self.glyphs, self.overlay,
            self.fig_kwargs, self.added_figures, self.added_overlays,
            self.added_overlay_figures, self.palette, self.title)

    def _make_figures(self):
        """ Make figures. """
        # TODO: check if we can put this in self.figure_map.figure
        self.figures = []

        for _, f in self.figure_map.iterrows():
            # adjust x axis type for datetime x values
            if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                f.fig_kwargs['x_axis_type'] = 'datetime'

            # set axis ranges
            if len(self.figures) == 0:
                if isinstance(self.data.indexes[self.x], pd.MultiIndex):
                    f.fig_kwargs['x_range'] = FactorRange(
                        *(tuple(str(i) for i in idx)
                          for idx in self.data.indexes[self.x].tolist()),
                        range_padding=0.1)
            else:
                f.fig_kwargs['x_range'] = self.figures[0].x_range
                if self.share_y:
                    f.fig_kwargs['y_range'] = self.figures[0].y_range

            if self.figsize is not None:
                width = self.figsize[0]//self.ncols
                height = self.figsize[1]//self.figure_map.shape[0]*self.ncols
                f.fig_kwargs.update(dict(plot_width=width, plot_height=height))

            self.figures.append(figure(tools=self.tools, **f.fig_kwargs))

    def _add_glyphs(self):
        """ Add glyphs. """
        for g_idx, g in self.glyph_map.iterrows():
            glyph_kwargs = clone_models(g.glyph_kwargs)
            if isinstance(g.method, str):
                getattr(self.figures[g.figure], g.method)(
                    source=g.handler.source, **glyph_kwargs)
            else:
                self.figures[g.figure].add_layout(
                    g.method(source=g.handler.source, ** glyph_kwargs))
            # add an invisible circle glyph to make glyph selectable
            if g.method != 'circle':
                self.figures[g.figure].circle(
                    source=g.handler.source, size=0,
                    **{'x': glyph_kwargs[g.x_arg], 'y': glyph_kwargs[g.y_arg]})

    def _add_annotations(self):
        """ Add annotations. """
        for idx, a in enumerate(self.added_annotations):
            f_idx = _get_overlay_figures(
                self.added_annotation_figures[idx], self.figure_map)
            for f in f_idx:
                if isinstance(a, Glyph):
                    self.figures[f].add_glyph(a)
                else:
                    self.figures[f].add_layout(a)

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
        self.layout = gridplot(self.figures, ncols=self.ncols,
                               toolbar_location=self.toolbar_location)

    def _modify_figures(self):
        """ Modify the attributes of multiple figures. """
        for figures, modifiers in self.modifiers:
            if figures is None:
                figures = self.figures
            elif isinstance(figures, int):
                figures = [self.figures[figures]]
            else:
                figures = [self.figures[idx] for idx in figures]

            for f in figures:
                self._modify_figure(modifiers, f)

    def _modify_figure(self, modifiers, f):
        """ Modify the attributes of a figure. """
        for m in modifiers:
            rsetattr(f, m, modifiers[m])

    def make_layout(self):
        """ Make the layout. """
        self._attach_elements()
        self._make_handlers()
        self._make_maps()
        self._make_figures()
        self._modify_figures()
        self._add_glyphs()
        self._add_annotations()
        self._add_tooltips()
        self._finalize_layout()

        return self.layout

    def add_figure(self, glyphs, data, coords=None, name=None):
        """ Add a figure to the layout.

        Parameters
        ----------
        glyphs :
        data :
        coords :
        name :
        """
        element = self.element_type(glyphs, data, coords, name)
        self.added_figures.append(element)

    def add_overlay(self, glyphs, data, coords=None, name=None, onto=None):
        """ Add an overlay to a figure in the layout.

        Parameters
        ----------
        glyphs :
        data :
        coords :
        name :
        onto : str or int, optional
            Title or index of the figure on which the element will be
            overlaid. By default, the element is overlaid on all figures.
        """
        element = self.element_type(glyphs, data, coords, name)
        self.added_overlays.append(element)
        self.added_overlay_figures.append(onto)

    def add_annotation(self, annotation, onto=None):
        """ Add an annotation to a figure in the layout.

        Parameters
        ----------
        annotation :
        onto : str or int, optional
            Title or index of the figure on which the annotation will be
            overlaid. By default, the annotation is overlaid on all figures.
        """
        self.added_annotations.append(annotation)
        self.added_annotation_figures.append(onto)

    def modify_figures(self, modifiers, figures=None):
        """ Modify the attributes of a figure.

        Parameters
        ----------
        modifiers : dict
            The attributes to modify. Keys can reference sub-attributes,
            e.g. 'xaxis.axis_label'.

        figures : int or iterable of int, optional
            The index(es) of the figure(s) to modify.
        """
        self.modifiers.append((figures, modifiers))


class BaseViewer(BasePlot):
    """ Base class for interactive viewers. """
    element_type = InteractiveElement
    handler_type = InteractiveDataHandler

    def __init__(self, *args, **kwargs):

        super(BaseViewer, self).__init__(*args, **kwargs)

        self.verbose = False

        self.added_interactions = []

        self.pending_handler_update = False
        self.handler_update_buffer = None

    # --  Callbacks -- #
    def on_selected_points_change(self, attr, old, new):
        """ Callback for selection event. """
        for handler in self.handlers:
            if handler.source.selected.indices is new:
                break
        else:
            raise ValueError('The source that emitted the selection change '
                             'was not found in this object\'s handlers.')

        if new is handler.selection:
            return

        # Update the selection bounds of the handlers
        if len(new) == 0:
            for h in self.handlers:
                h.selection_bounds = None
        else:
            new_start = np.min(new)
            new_end = np.max(new)
            idx_start = handler.source.data['index'][new_start]
            idx_end = handler.source.data['index'][new_end]
            for h in self.handlers:
                h.selection_bounds = (idx_start, idx_end)

        # Update handlers
        self.doc.add_next_tick_callback(self.update_handlers)

    @without_document_lock
    @gen.coroutine
    def update_handler(self, handler):
        """ Update a single handler. """
        handler.update()

    @without_document_lock
    @gen.coroutine
    def update_handlers(self, handlers=None):
        """ Update handlers. """
        if handlers is None:
            handlers = self.handlers
        for h in handlers:
            if not h.pending_update:
                self.doc.add_next_tick_callback(
                    partial(self.update_handler, h))
            else:
                if self.verbose:
                    print('Buffering')
                h.update_buffer = partial(self.update_handlers, [h])

    @without_document_lock
    @gen.coroutine
    def reset_handlers(self):
        """ Reset handlers. """
        for h in self.handlers:
            h.reset()

    def on_reset(self, event):
        """ Callback for reset event. """
        self.doc.add_next_tick_callback(self.reset_handlers)

    # --  Private methods -- #
    def _make_handlers(self):
        """ Make handlers. """
        self.handlers = [
            self.handler_type(self._collect(coords=self.coords), context=self)]
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
            h.selection_bounds = None
            self.update_handler(h)

    def _attach_elements(self):
        """ Attach additional elements to this viewer. """
        super(BaseViewer, self)._attach_elements()
        for interaction in self.added_interactions:
            interaction.attach(self)

    def _add_glyphs(self):
        """ Add glyphs. """
        for g_idx, g in self.glyph_map.iterrows():
            glyph_kwargs = clone_models(g.glyph_kwargs)
            if isinstance(g.method, str):
                glyph = getattr(self.figures[g.figure], g.method)(
                    source=g.handler.source, **glyph_kwargs)
            else:
                glyph = self.figures[g.figure].add_layout(
                    g.method(source=g.handler.source, ** glyph_kwargs))
            if g.method != 'circle':
                circle = self.figures[g.figure].circle(
                    source=g.handler.source, size=0,
                    **{'x': glyph_kwargs[g.x_arg], 'y': glyph_kwargs[g.y_arg]})
                circle.data_source.selected.on_change(
                    'indices', self.on_selected_points_change)
            else:
                glyph.data_source.selected.on_change(
                    'indices', self.on_selected_points_change)

    def _add_callbacks(self):
        """ Add callbacks. """

    def _finalize_layout(self):
        """ Finalize layout. """
        super(BaseViewer, self)._finalize_layout()

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

    def _modify_figure(self, modifiers, f):
        """ Modify the attributes of a figure. """
        for m in modifiers:
            if self.doc is not None:
                mod_func = lambda: rsetattr(f, m, modifiers[m])
                self.doc.add_next_tick_callback(mod_func)
            else:
                rsetattr(f, m, modifiers[m])

    def _inplace_update(self):
        """ Update the current layout in place. """
        self.doc.roots[0].children[0] = self.layout

    # --  Public methods -- #
    def make_layout(self):
        """ Make the layout. """
        self._attach_elements()
        self._make_handlers()
        self._make_maps()
        self._make_figures()
        self._modify_figures()
        self._add_glyphs()
        self._add_tooltips()
        self._add_callbacks()
        self._finalize_layout()

        return self.layout

    def update_inplace(self, other):
        """ Update this instance with the properties of another layout.

        Parameters
        ----------
        other : xrview.core.BaseViewer
            The instance that replaces the current instance.
        """
        doc = self.doc
        self.__dict__ = other.__dict__  # TODO: make this safer
        self.make_layout()
        self.doc = doc

        self.doc.add_next_tick_callback(self._inplace_update)

    def add_interaction(self, interaction):
        """ Add an interaction to the layout.

        Parameters
        ----------
        interaction : Interaction
            The interaction to add.
        """
        self.added_interactions.append(interaction)
