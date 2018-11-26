""" ``xrview.core`` """

import abc
import six
from copy import copy

import numpy as np
import pandas as pd

from bokeh.io import output_notebook
from bokeh.io.notebook import show_app
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure
from bokeh.models import HoverTool

from xrview.utils import \
    is_dataset, is_dataarray, get_notebook_url, rsetattr, clone_models
from xrview.elements import get_glyph
from xrview.palettes import RGB
from xrview.handlers import DataHandler


@six.add_metaclass(abc.ABCMeta)
class NotebookServer(object):
    """ Base class for bokeh notebook apps.

    Parameters
    ----------
    data : xarray DataArray or Dataset
        The data to display.

    figsize : iterable
        The size of the figure in pixels.
    """

    def __init__(self, data, figsize):

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

        self.figsize = figsize

        self.doc = None
        self.layout = None

    @abc.abstractmethod
    def _make_layout(self):
        """ Make the app layout. """

    def _make_app(self, doc):
        """ Make the app for displaying in a jupyter notebook. """

        self.doc = doc
        self._make_layout()
        self.doc.add_root(row(self.layout))

    def _inplace_update(self):
        """ Update the current layout in place. """

        self.doc.roots[0].children[0] = self.layout

    def update_inplace(self, other):
        """ Update this instance with the properties of another viewer instance.

        Parameters
        ----------
        other : xrview.core.NotebookServer
            The instance that replaces the current instance.
        """

        doc = self.doc
        self.__dict__ = other.__dict__ # TODO: make this safer
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
        new : xrview.core.NotebookServer
            The copied object.
        """

        from copy import copy

        new = self.__new__(type(self))

        new.__dict__ = {k: (copy(v) if (k != 'data' or with_data) else v)
                        for k, v in self.__dict__.items()}

        return new

    def show(self, notebook_url=None, port=0, verbose=True):
        """ Show the app in a jupyter notebook.

        Parameters
        ----------
        notebook_url : str, optional
            The URL of the notebook. Will be determined automatically if not
            specified.

        port : int, default 0
            The port over which the app will be served. Chosen randomly if
            set to 0.

        verbose : bool, default True
            If True, create the document once again outside of show_app in
            order to show errors.
        """

        if notebook_url is None:
            notebook_url = get_notebook_url()

        output_notebook(hide_banner=True)
        app = Application(FunctionHandler(self._make_app))

        if verbose:
            app.create_document()

        show_app(app, None, notebook_url=notebook_url, port=port)


class NotebookViewer(NotebookServer):
    """ Base class for timeseries viewers.

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

        super(NotebookViewer, self).__init__(data, figsize)

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

        self.tooltips = tooltips

        # layout parameters
        self.ncols = ncols
        self.element = get_glyph(glyph)
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

        #
        self.added_figures = []
        self.added_overlays = []
        self.added_overlay_figures = []
        self.added_interactions = []

        self.handlers = None
        self.glyph_map = None
        self.figure_map = None
        self.figures = None

        self.doc = None
        self.layout = None

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

    def _make_glyph_map(
            self, data, handler, method, x_arg, y_arg, glyph_kwargs):
        """ Make a glyph map. """

        data_list = []

        for v in data.data_vars:
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                data_list.append((v, None, None))
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for dval in data[dim].values:
                    data_list.append((v, dim, dval))
            else:
                raise ValueError(v + ' has too many dimensions')

        glyph_map = pd.DataFrame(
            data_list, columns=['var', 'dim', 'dim_val'])

        glyph_map.loc[:, 'handler'] = handler
        glyph_map.loc[:, 'method'] = method
        glyph_map.loc[:, 'x_arg'] = x_arg
        glyph_map.loc[:, 'y_arg'] = y_arg
        glyph_map.loc[:, 'glyph_kwargs'] = \
            [copy(glyph_kwargs) for _ in range(glyph_map.shape[0])]

        return glyph_map

    def _make_maps(self):
        """ Make the figure and glyph map. """

        glyph_map = self._make_glyph_map(
            self.data, self.handlers[0], self.element.method,
            self.element.x_arg, self.element.y_arg, self.element.glyph_kwargs)
        figure_map = pd.DataFrame(columns=['figure', 'fig_kwargs'])

        if self.overlay == 'dims':
            figure_names = glyph_map['var']
        else:
            if len(np.unique(glyph_map['dim'])) > 1:
                raise ValueError(
                    'Dimensions of all data variables must match')
            else:
                figure_names = glyph_map['dim_val']

        if self.overlay == 'data_vars':
            legend_col = 'var'
        else:
            legend_col = 'dim_val'

        # make figure map for base figures
        for f_idx, f_name in enumerate(np.unique(figure_names)):
            glyph_map.loc[figure_names == f_name, 'figure'] = f_idx
            figure_map = figure_map.append(
                {'figure': None, 'fig_kwargs': copy(self.fig_kwargs)},
                ignore_index=True)
            figure_map.iloc[-1]['fig_kwargs'].update({'title': str(f_name)})

        # add additional figures
        for added_idx, element in enumerate(self.added_figures):

            if hasattr(element, 'glyphs'):
                added_glyph_map = pd.concat([self._make_glyph_map(
                    element.data, element.handler, g.glyph, g.x_arg, g.y_arg,
                    g.glyph_kwargs) for g in element.glyphs], ignore_index=True)
            else:
                added_glyph_map = self._make_glyph_map(
                    element.data, element.handler, element.method,
                    element.x_arg, element.y_arg, element.glyph_kwargs)

            added_glyph_map.loc[:, 'figure'] = f_idx + added_idx + 1
            glyph_map = glyph_map.append(added_glyph_map, ignore_index=True)

            figure_map = figure_map.append(
                {'figure': None, 'fig_kwargs': copy(self.fig_kwargs)},
                ignore_index=True)
            figure_map.iloc[-1]['fig_kwargs'].update({'title': element.name})

        # add additional overlays
        for added_idx, element in enumerate(self.added_overlays):

            if hasattr(element, 'glyphs'):
                added_glyph_map = pd.concat([self._make_glyph_map(
                    element.data, element.handler, g.method, g.x_arg, g.y_arg,
                    g.glyph_kwargs) for g in element.glyphs],
                    ignore_index=True)
            else:
                added_glyph_map = self._make_glyph_map(
                    element.data, element.handler, element.method,
                    element.x_arg, element.y_arg, element.glyph_kwargs)

            # find the indices of the figures to overlay
            if self.added_overlay_figures[added_idx] is None:
                figure_idx = figure_map.index.values
            elif isinstance(self.added_overlay_figures[added_idx], int):
                figure_idx =[self.added_overlay_figures[added_idx]]
            else:
                titles = np.array(
                    [a['title'] for a in figure_map['fig_kwargs']])
                _, title_idx = np.unique(titles, return_index=True)
                titles = titles[np.sort(title_idx)]
                figure_idx = figure_map.index[
                    titles == self.added_overlay_figures[added_idx]].values

            for f_idx in figure_idx:
                added_glyph_map.loc[:, 'figure'] = f_idx
                glyph_map = glyph_map.append(
                    added_glyph_map, ignore_index=True)

        # update glyph_kwargs
        colormap = {v: self.palette[i]
                    for i, v in enumerate(pd.unique(glyph_map[legend_col]))}

        for idx, g in glyph_map.iterrows():

            if g['dim_val'] is None:
                y_col = str(g['var'])
            else:
                y_col = '_'.join((str(g['var']), str(g['dim_val'])))

            if g[legend_col] is not None:
                legend = str(g[legend_col])
                color = colormap[g[legend_col]]
            else:
                legend = None
                color = None

            glyph_kwargs = {g.x_arg: 'index', g.y_arg: y_col,
                            'legend': legend, 'color': color}
            glyph_kwargs.update(glyph_map.loc[idx, 'glyph_kwargs'])
            glyph_map.loc[idx, 'glyph_kwargs'].update(glyph_kwargs)

        glyph_map.loc[:, 'figure'] = glyph_map.loc[:, 'figure'].astype(int)

        self.figure_map = figure_map
        self.glyph_map = glyph_map

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

            self.figures[-1].xgrid.visible = False

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

    def _add_tooltips(self):
        """ Add tooltips. """

        if self.tooltips is not None:
            tooltips = [(k, v) for k, v in self.tooltips.items()]
            for f in self.figures:
                f.select(HoverTool).tooltips = tooltips
                if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                    f.select(HoverTool).formatters = {'index': 'datetime'}

    def _attach_elements(self):
        """ Attach additional elements to this viewer. """

        # TODO: rename

        for element in self.added_figures + self.added_overlays:
            element.attach(self)

        for interaction in self.added_interactions:
            interaction.attach(self)

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
        """ Make the app layout. """

        # attach elements
        self._attach_elements()

        # make handlers
        self._make_handlers()

        # make maps
        self._make_maps()

        # make figures
        self._make_figures()

        # add glyphs
        self._add_glyphs()

        # customize hover tooltips
        self._add_tooltips()

        # add callbacks
        self._add_callbacks()

        # finalize layout
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

    def add_interaction(self, interaction):
        """ Add an interaction to the layout.

        Parameters
        ----------
        interaction : Interaction
            The interaction to add.
        """

        self.added_interactions.append(interaction)

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
