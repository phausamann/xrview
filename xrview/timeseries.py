""" ``xrview.timeseries.base`` """

import numpy as np

from bokeh.events import Reset

from bokeh.document import without_document_lock
from tornado import gen
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from xrview.core import NotebookViewer

from xrview.handlers import ResamplingDataHandler


class TimeseriesViewer(NotebookViewer):
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

    def __init__(self, data, x, overlay='dims', glyph='line', tooltips=None,
                 tools=None, figsize=(900, 400), ncols=1, palette=None,
                 ignore_index=False, resolution=4, max_workers=10,
                 lowpass=False, verbose=0, **fig_kwargs):

        super(TimeseriesViewer, self).__init__(
            data, x, overlay=overlay, glyph=glyph, tooltips=tooltips,
            tools=tools, figsize=figsize, ncols=ncols, palette=palette,
            ignore_index=ignore_index, **fig_kwargs)

        # sub-sampling parameters
        self.resolution = resolution
        self.thread_pool = ThreadPoolExecutor(max_workers)
        self.lowpass = lowpass
        self.verbose = verbose

        self.pending_handler_update = False
        self.xrange_change_buffer = None

    @without_document_lock
    @gen.coroutine
    def reset_xrange(self):
        """ """

        for h in self.handlers:
            yield self.thread_pool.submit(h.reset_data)
            self.doc.add_next_tick_callback(h.update_source)

    @without_document_lock
    @gen.coroutine
    def update_handlers(self):
        """ Update handlers. """

        for h in self.handlers:
            yield self.thread_pool.submit(partial(
                h.update_data,
                start=self.figures[0].x_range.start,
                end=self.figures[0].x_range.end))
            self.doc.add_next_tick_callback(h.update_source)

        if self.xrange_change_buffer is not None:
            self.doc.add_next_tick_callback(self.update_handlers)
            self.xrange_change_buffer = None

    def on_xrange_change(self, attr, old, new):
        """ Callback for xrange change event. """

        if not self.pending_handler_update:
            self.pending_handler_update = True
            self.doc.add_next_tick_callback(self.update_handlers)
        else:
            if self.verbose:
                print('Buffering')
            self.xrange_change_buffer = new

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

        # push out a handler update to update all sources
        if not self.pending_handler_update:
            self.pending_handler_update = True
            self.doc.add_next_tick_callback(self.update_handlers)

    def on_reset(self, event):
        """ Callback for reset event. """

        self.pending_handler_update = True
        self.doc.add_next_tick_callback(self.reset_xrange)

    def _make_handlers(self):
        """ Make handlers. """

        # default handler
        self.handlers = [ResamplingDataHandler(
            self._collect(), self.resolution * self.figsize[0], context=self,
            lowpass=self.lowpass)]

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
