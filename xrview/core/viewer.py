from functools import partial

import numpy as np
from bokeh.document import without_document_lock
from bokeh.layouts import row, column
from tornado import gen

from xrview.core import BasePlot
from xrview.elements import InteractiveElement
from xrview.handlers import InteractiveDataHandler
from xrview.utils import clone_models, rsetattr


class BaseViewer(BasePlot):
    """ Interactive viewer."""

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
                self.doc.add_next_tick_callback(
                    lambda: rsetattr(f, m, modifiers[m]))
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
        other : xrview.core.viewer.BaseViewer
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
        interaction : xrview.interactions.BaseInteraction
            The interaction to add.
        """
        self.added_interactions.append(interaction)
