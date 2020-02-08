""" ``xrview.timeseries.base`` """

from bokeh.events import Reset

from bokeh.document import without_document_lock
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from xrview.core import BaseViewer
from xrview.handlers import ResamplingDataHandler
from xrview.elements import ResamplingElement


class TimeseriesViewer(BaseViewer):
    """ Interactive viewer for large time series datasets."""

    element_type = ResamplingElement
    handler_type = ResamplingDataHandler

    def __init__(self, data, x, overlay='dims', glyphs='line', tooltips=None,
                 tools=None, figsize=(600, 300), ncols=1, palette=None,
                 ignore_index=False, resolution=4, max_workers=10,
                 lowpass=False, verbose=0, **fig_kwargs):
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
            e.g. {'datetime': '@index{%F %T.%3N}'}.

        tools : str, optional
            bokeh tool string.

        palette : iterable, optional
            The palette to use when overlaying multiple glyphs.

        ignore_index : bool, default Falseh
            If True, replace the x-axis values of the data by an appropriate
            evenly spaced index.

        resolution : int, default 4
            The number of points to render for each pixel.

        max_workers : int, default 10
            The maximum number of workers in the thread pool to perform the
            down-sampling.

        lowpass : bool, default False
            If True, filter the values with a low-pass filter before
            down-sampling.

        verbose : int, default 0
            The level of verbosity.
        """
        super(TimeseriesViewer, self).__init__(
            data, x, overlay=overlay, glyphs=glyphs, tooltips=tooltips,
            tools=tools, figsize=figsize, ncols=ncols, palette=palette,
            ignore_index=ignore_index, **fig_kwargs)

        # sub-sampling parameters
        self.resolution = resolution
        self.thread_pool = ThreadPoolExecutor(max_workers)
        self.lowpass = lowpass
        self.verbose = verbose

    @without_document_lock
    @gen.coroutine
    def update_handler(self, handler):
        """ Update a single handler. """
        yield self.thread_pool.submit(
            partial(handler.update,
                    start=self.figures[0].x_range.start,
                    end=self.figures[0].x_range.end))

    @without_document_lock
    @gen.coroutine
    def reset_handlers(self):
        """ Reset handlers. """
        for h in self.handlers:
            yield self.thread_pool.submit(h.reset)

    def on_xrange_change(self, attr, old, new):
        """ Callback for xrange change event. """
        self.doc.add_next_tick_callback(self.update_handlers)

    def _make_handlers(self):
        """ Make handlers. """
        self.handlers = [self.handler_type(
            self._collect(coords=self.coords), self.resolution*self.figsize[0],
            context=self, lowpass=self.lowpass)]
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
                h.data = element_list[h_idx - 1]._collect(hooks)

            start, end = h.get_range(
                self.figures[0].x_range.start, self.figures[0].x_range.end)

            h.update_data(start, end)
            h.update_source()

            if h.source.selected is not None:
                h.source.selected.indices = []

    def _add_callbacks(self):
        """ Add callbacks. """
        self.figures[0].x_range.on_change('start', self.on_xrange_change)
        self.figures[0].x_range.on_change('end', self.on_xrange_change)
        self.figures[0].on_event(Reset, self.on_reset)

    def add_figure(self, data, glyphs='line', coords=None, name=None,
                   resolution=None):
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

        resolution : float, optional
            The number of points to render for each pixel of this figure. If
            not specified, the resolution of the viewer is used.
        """
        element = self.element_type(glyphs, data, coords, name, resolution)
        self.added_figures.append(element)

    def add_overlay(self, data, glyphs='line', coords=None, name=None,
                    onto=None, resolution=None):
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
            The name of the DataArray which will be used as the title of the
            figure. If not provided, the name of the DataArray will be used.

        onto : str or int, optional
            Title or index of the figure on which the element will be
            overlaid. By default, the element is overlaid on all figures.

        resolution : float, optional
            The number of points to render for each pixel of this figure. If
            not specified, the resolution of the viewer is used.
        """
        element = self.element_type(glyphs, data, coords, name, resolution)
        self.added_overlays.append(element)
        self.added_overlay_figures.append(onto)
