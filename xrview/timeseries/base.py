""" ``xrview.timeseries.base`` """

from copy import copy

import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.palettes import Paired12
from bokeh.io import output_notebook
from bokeh.io.notebook import show_app
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.events import Reset

from bokeh.document import without_document_lock
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from xrview.utils import is_dataset, is_dataarray, get_notebook_url

from .handlers import ResamplingDataHandler


_RGB = Paired12[5::-2] + Paired12[-1:5:-2] + Paired12[4::-2] + Paired12[-2:4:-2]


class Viewer(object):
    """ Base class for timeseries viewers.

    Parameters
    ----------
    data : xarray DataArray or Dataset
        The data to display.

    x : str
        The name of the dimension in ``data`` that contains the x-axis values.

    overlay : 'dims' or 'data_vars', default 'dims'

    tooltips : dict, optional

    tools : str, optional

    figsize : iterable, default (900, 400)
        The size of the figure in pixels.

    ncols : int, default 1

    palette : iterable, optional

    resolution : int, default 4
        The number of points to render for each pixel.

    max_workers : int, default 10
        The maximum number of workers in the thread pool to perform the
        down-sampling.

    lowpass : bool, default False
        If True, filter the values with a low-pass filter before down-sampling.

    verbose : int, default 0
        The level of verbosity.
    """

    def __init__(self, data, x, overlay='dims', tooltips=None, tools=None,
                 figsize=(900, 400), ncols=1, palette=None,
                 ignore_index=False, resolution=4, max_workers=10,
                 lowpass=False, verbose=0):

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

        self.tooltips = tooltips

        self.ignore_index = ignore_index

        # layout parameters
        self.figsize = figsize
        self.ncols = ncols
        self.glyph = 'line'
        self.fig_kwargs = {}
        self.glyph_kwargs = {}

        if palette is None:
            self.palette = _RGB
        else:
            self.palette = palette

        if tools is None:
            self.tools = 'pan,wheel_zoom,box_zoom,xbox_select,reset'
            if self.tooltips is not None:
                self.tools += ',hover'
        else:
            self.tools = tools

        # sub-sampling parameters
        self.resolution = resolution
        self.thread_pool = ThreadPoolExecutor(max_workers)
        self.lowpass = lowpass
        self.verbose = verbose

        #
        self.added_figures = []
        self.added_overlays = []
        self.added_overlay_figures = []
        self.added_interactions = []

        self.handlers = None
        self.glyph_map = None
        self.figure_map = None
        self.figures = None

        self.pending_xrange_update = False
        self.xrange_change_buffer = None

        self.doc = None
        self.layout = None

    @without_document_lock
    @gen.coroutine
    def reset_xrange(self):
        """ """

        for h in self.handlers:
            yield self.thread_pool.submit(h.reset_data)
            self.doc.add_next_tick_callback(h.update_source)

    @without_document_lock
    @gen.coroutine
    def update_xrange(self):
        """ Update plot_source when xrange changes. """

        for h in self.handlers:
            yield self.thread_pool.submit(partial(
                h.update_data,
                start=self.figures[0].x_range.start,
                end=self.figures[0].x_range.end))
            self.doc.add_next_tick_callback(h.update_source)

        if self.xrange_change_buffer is not None:
            self.doc.add_next_tick_callback(self.update_xrange)
            self.xrange_change_buffer = None

    def on_xrange_change(self, attr, old, new):
        """ Callback for xrange change event. """

        if not self.pending_xrange_update:
            self.pending_xrange_update = True
            self.doc.add_next_tick_callback(self.update_xrange)
        else:
            if self.verbose:
                print('Buffering')
            self.xrange_change_buffer = new

    def on_selected_points_change(self, attr, old, new):
        """ Callback for selection event. """

        idx_new = np.array(new['1d']['indices'])

        for h in self.handlers:
            h.data.selected = np.zeros(len(h.data.selected), dtype=bool)
            sel_idx_start = h.source.data['index'][np.min(idx_new)]
            sel_idx_end = h.source.data['index'][np.max(idx_new)]
            h.data.loc[np.logical_and(
                h.data.index >= sel_idx_start,
                h.data.index <= sel_idx_end), 'selected'] = True

    def on_reset(self, event):
        """ Callback for reset event. """

        self.pending_xrange_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

    def attach_elements(self):
        """ Attach additional elements to this viewer. """

        # TODO: rename

        for element in self.added_figures + self.added_overlays:
            element.attach(self)

        for interaction in self.added_interactions:
            interaction.attach(self)

    def _collect(self, data):
        """ Base method for collect. """

        plot_data = dict()

        for v in data.data_vars:
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                plot_data[v] = data[v].values
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for d in data[dim].values:
                    plot_data[v + '_' + str(d)] = data[v].sel(**{dim: d}).values
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

    def collect(self, hooks=None):
        """ Collect plottable data in a pandas DataFrame. """

        data = self.data

        if hooks is not None:
            for h in hooks:
                data = h(data)

        return self._collect(data)

    def make_handlers(self):
        """ Make handlers. """

        # default handler
        self.handlers = [ResamplingDataHandler(
            self.collect(), self.resolution * self.figsize[0], context=self,
            lowpass=self.lowpass)]

        for element in self.added_figures + self.added_overlays:
            self.handlers.append(element.handler)

    def update_handlers(self, hooks=None):
        """ Update handlers. """

        if hooks is None:
            hooks = [i.collect_hook for i in self.added_interactions]

        element_list = self.added_figures + self.added_overlays

        for h_idx, h in enumerate(self.handlers):

            if h_idx == 0:
                h.data = self.collect(hooks)
            else:
                h.data = element_list[h_idx-1].collect(hooks)

            start, end = h.get_range(
                self.figures[0].x_range.start, self.figures[0].x_range.end)

            h.update_data(start, end)
            h.update_source()

            if h.source.selected is not None:
                h.source.selected.indices = []

    def make_glyph_map(self, data, handler, glyph, glyph_kwargs):
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
        glyph_map.loc[:, 'glyph'] = glyph
        glyph_map.loc[:, 'glyph_kwargs'] = \
            [copy(glyph_kwargs) for _ in range(glyph_map.shape[0])]

        return glyph_map

    def make_maps(self):
        """ Make the figure and glyph map. """

        glyph_map = self.make_glyph_map(
            self.data, self.handlers[0], self.glyph, self.glyph_kwargs)
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
                added_glyph_map = pd.concat([self.make_glyph_map(
                    element.data, element.handler, g.glyph, g.glyph_kwargs)
                    for g in element.glyphs], ignore_index=True)
            else:
                added_glyph_map = self.make_glyph_map(
                    element.data, element.handler, element.glyph,
                    element.glyph_kwargs)

            added_glyph_map.loc[:, 'figure'] = f_idx + added_idx + 1
            glyph_map = glyph_map.append(added_glyph_map, ignore_index=True)

            figure_map = figure_map.append(
                {'figure': None, 'fig_kwargs': copy(self.fig_kwargs)},
                ignore_index=True)
            figure_map.iloc[-1]['fig_kwargs'].update({'title': element.name})

        # add additional overlays
        for added_idx, element in enumerate(self.added_overlays):

            if hasattr(element, 'glyphs'):
                added_glyph_map = pd.concat([self.make_glyph_map(
                    element.data, element.handler, g.glyph, g.glyph_kwargs)
                    for g in element.glyphs], ignore_index=True)
            else:
                added_glyph_map = self.make_glyph_map(
                    element.data, element.handler, element.glyph,
                    element.glyph_kwargs)

            # find the indices of the figures to overlay
            if self.added_overlay_figures[added_idx] is None:
                figure_idx = figure_map.index.values
            elif isinstance(self.added_overlay_figures[added_idx], int):
                figure_idx =[self.added_overlay_figures[added_idx]]
            else:
                figure_idx = figure_map.index[
                    np.unique([a['title'] for a in figure_map['fig_kwargs']]) ==
                    self.added_overlay_figures[added_idx]].values

            for f_idx in figure_idx:
                added_glyph_map.loc[:, 'figure'] = f_idx
                glyph_map = glyph_map.append(added_glyph_map, ignore_index=True)

        # update glyph_kwargs
        colormap = {v: self.palette[i]
                    for i, v in enumerate(pd.unique(glyph_map[legend_col]))}

        for idx, g in glyph_map.iterrows():

            if g['dim_val'] is None:
                source_col = str(g['var'])
            else:
                source_col = '_'.join((str(g['var']), str(g['dim_val'])))

            if g[legend_col] is not None:
                legend = str(g[legend_col])
                color = colormap[g[legend_col]]
            else:
                legend = None
                color = None

            glyph_map.loc[idx, 'source_col'] = source_col
            glyph_kwargs = {'legend': legend, 'color': color}
            glyph_kwargs.update(glyph_map.loc[idx, 'glyph_kwargs'])
            glyph_map.loc[idx, 'glyph_kwargs'].update(glyph_kwargs)

        glyph_map.loc[:, 'figure'] = glyph_map.loc[:, 'figure'].astype(int)

        self.figure_map = figure_map
        self.glyph_map = glyph_map

    def make_figures(self):
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

            self.figures.append(figure(
                plot_width=self.figsize[0]//self.ncols,
                plot_height=self.figsize[1]//self.figure_map.shape[0]*self.ncols,
                tools=self.tools, **f.fig_kwargs))

            self.figures[-1].xgrid.visible = False

    def add_glyphs(self):
        """ Add glyphs. """

        for g_idx, g in self.glyph_map.iterrows():

            g.glyph = getattr(self.figures[g.figure], g.glyph)(
                x='index', y=g.source_col, source=g.handler.source,
                **g.glyph_kwargs)

            circle = self.figures[g.figure].circle(
                x='index', y=g.source_col, source=g.handler.source, size=0)
            circle.data_source.on_change(
                'selected', self.on_selected_points_change)

    def add_tooltips(self):
        """ Add tooltips. """

        if self.tooltips is not None:
            tooltips = [(k, v) for k, v in self.tooltips.items()]
            for f in self.figures:
                f.select(HoverTool).tooltips = tooltips
                if isinstance(self.data.indexes[self.x], pd.DatetimeIndex):
                    f.select(HoverTool).formatters = {'index': 'datetime'}

    def add_callbacks(self):
        """ Add callbacks. """

        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

    def finalize_layout(self):
        """ Finalize layout. """

        self.layout = gridplot(self.figures, ncols=self.ncols)

        for interaction in self.added_interactions:
            interaction.layout_hook()

    def make_layout(self):
        """ Make the app layout. """

        # attach elements
        self.attach_elements()

        # make handlers
        self.make_handlers()

        # make maps
        self.make_maps()

        # make figures
        self.make_figures()

        # add glyphs
        self.add_glyphs()

        # customize hover tooltips
        self.add_tooltips()

        # add callbacks
        self.add_callbacks()

        # finalize layout
        self.finalize_layout()

    def make_app(self, doc):
        """ Make the app for displaying in a jupyter notebbok. """

        self.doc = doc
        self.make_layout()
        self.doc.add_root(self.layout)

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
            overlayed. By default, the element is overlayed on all figures.
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

    def show(self, notebook_url=None, port=0):
        """ Show the app in a jupyter notebook.

        Parameters
        ----------
        notebook_url : str, optional
            The URL of the notebook.

        port : int, default 0
            The port over which the app will be served. Chosen randomly if
            set to 0.
        """

        if notebook_url is None:
            notebook_url = get_notebook_url()

        output_notebook(hide_banner=True)
        app = Application(FunctionHandler(self.make_app))
        show_app(app, None, notebook_url=notebook_url, port=port)
