""" ``xrview.timeseries.base`` """

import numpy as np

from bokeh.events import Reset

from bokeh.document import without_document_lock
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from xrview.core import BaseViewer
from xrview.notebook.base import NotebookServer
from xrview.handlers import ResamplingDataHandler
from xrview.elements import ResamplingElement


class TimeseriesViewer(BaseViewer):
    """ Interactive viewer for large time series datasets.

    Parameters
    ----------
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
    element_type = ResamplingElement
    handler_type = ResamplingDataHandler

    def __init__(self, data, x, overlay='dims', glyphs='line', tooltips=None,
                 tools=None, figsize=(900, 400), ncols=1, palette=None,
                 ignore_index=False, resolution=4, max_workers=10,
                 lowpass=False, verbose=0, **fig_kwargs):

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
    def reset_xrange(self):
        """ """
        for h in self.handlers:
            yield self.thread_pool.submit(h.reset_data)
            self.doc.add_next_tick_callback(h.update_source)

    @without_document_lock
    @gen.coroutine
    def update_handlers(self, handlers=None):
        """ Update handlers. """
        if handlers is None:
            handlers = self.handlers
        for h in handlers:
            yield self.thread_pool.submit(partial(
                h.update_data,
                start=self.figures[0].x_range.start,
                end=self.figures[0].x_range.end))
            self.doc.add_next_tick_callback(h.update_source)

        if self.handler_update_buffer is not None:
            self.doc.add_next_tick_callback(self.handler_update_buffer)
            self.handler_update_buffer = None

    def on_xrange_change(self, attr, old, new):
        """ Callback for xrange change event. """
        if not self.pending_handler_update:
            self.pending_handler_update = True
            self.doc.add_next_tick_callback(self.update_handlers)
        else:
            if self.verbose:
                print('Buffering')
            self.handler_update_buffer = \
                lambda: self.on_xrange_change(attr, old, new)

    def on_reset(self, event):
        """ Callback for reset event. """
        self.pending_handler_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

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


class TimeseriesNotebookViewer(TimeseriesViewer, NotebookServer):
    """"""
