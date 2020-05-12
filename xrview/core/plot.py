import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import FactorRange, Glyph, HoverTool
from bokeh.plotting import figure

from xrview.core.panel import BasePanel
from xrview.elements import Element
from xrview.glyphs import get_glyph_list
from xrview.handlers import DataHandler
from xrview.mappers import map_figures_and_glyphs, _get_overlay_figures
from xrview.palettes import RGB
from xrview.utils import is_dataarray, is_dataset, clone_models, rsetattr


class BasePlot(BasePanel):
    """ Base class for plots. """

    element_type = Element
    handler_type = DataHandler
    default_tools = 'pan,wheel_zoom,save,reset,'

    def __init__(self, data, x,
                 overlay='dims',
                 coords=None,
                 glyphs='line',
                 title=None,
                 share_y=False,
                 tooltips=None,
                 tools=None,
                 toolbar_location='right',
                 figsize=(600, 300),
                 ncols=1,
                 palette=None,
                 ignore_index=False,
                 theme=None,
                 **fig_kwargs):
        """ Constructor.

        Parameters
        ----------
        data : xarray DataArray or Dataset
            The data to display.

        x : str
            The name of the dimension in ``data`` that contains the x-axis
            values.

        glyphs : str, BaseGlyph or iterable, default 'line'
            The glyph to use for plotting.

        figsize : iterable, default (600, 300)
            The size of the figure in pixels.

        ncols : int, default 1
            The number of columns of the layout.

        overlay : 'dims' or 'data_vars', default 'dims'
            If 'dims', make one figure for each data variable and overlay the
            dimensions. If 'data_vars', make one figure for each dimension and
            overlay the data variables. In the latter case, all variables must
            have the same dimensions.

        tooltips : dict, optional
            Names of tooltips mapping to glyph properties or source columns,
            e.g. datetime': '@index{%F %T.%3N}'}.

        tools : str, optional
            bokeh tool string.

        palette : iterable, optional
            The palette to use when overlaying multiple glyphs.

        ignore_index : bool, default Falseh
            If True, replace the x-axis values of the data by an appropriate
            evenly spaced index.
        """
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

        self.glyphs = get_glyph_list(glyphs)

        self.tooltips = tooltips

        # layout parameters
        self.title = title
        self.share_y = share_y
        self.ncols = ncols
        self.figsize = figsize
        self.fig_kwargs = fig_kwargs
        self.theme = theme

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

    def add_figure(self, data, glyphs='line', coords=None, name=None):
        """ Add a figure to the layout.

        Parameters
        ----------
        data : xarray.DataArray
            The data to display.

        glyphs : str, BaseGlyph or iterable thereof, default 'line'
            The glyph (or glyphs) to display.

        coords : iterable of str, optional
            The coordinates of the DataArray to include. This is necessary
            for composite glyphs such as BoxWhisker.

        name : str, optional
            The name of the DataArray which will be used as the title of the
            figure. If not provided, the name of the DataArray will be used.
        """
        element = self.element_type(glyphs, data, coords, name)
        self.added_figures.append(element)

    def add_overlay(self, data, glyphs='line', coords=None, name=None,
                    onto=None):
        """ Add an overlay to a figure in the layout.

        Parameters
        ----------
        data : xarray.DataArray
            The data to display.

        glyphs : str, BaseGlyph or iterable thereof, default 'line'
            The glyph (or glyphs) to display.

        coords : iterable of str, optional
            The coordinates of the DataArray to include. This is necessary
            for composite glyphs such as BoxWhisker.

        name : str, optional
            The name of the DataArray which will be used to identify the
            overlay. If not provided, the name of the DataArray will be used.

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
