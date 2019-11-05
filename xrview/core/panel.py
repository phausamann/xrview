import abc

from bokeh.document import Document
from bokeh.io import export_png, export_svgs
from bokeh.layouts import row, gridplot, column
from bokeh.models import Spacer


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
    def show(self, *args, **kwargs):
        """ Show the layout. """

    def _export(self, func, backend, filename):
        """ Export. """
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
        self.doc.theme = self.theme
        self.doc.add_root(row(self.layout))

    def copy(self, with_data=False):
        """ Create a copy of this instance.

        Parameters
        ----------
        with_data : bool, default False
            If true, also copy the data.

        Returns
        -------
        new : xrview.core.panel.BasePanel
            The copied object.
        """
        from copy import copy

        new = self.__new__(type(self))
        new.__dict__ = {k: (copy(v) if (k != 'data' or with_data) else v)
                        for k, v in self.__dict__.items()}

        return new


class GridPlot(BasePanel):
    """ Base class for grid plots. """

    def __init__(self, panels, ncols=1, toolbar_location='above'):
        """ Constructor. """
        self.panels = panels
        self.ncols = ncols
        self.toolbar_location = toolbar_location
        self.make_layout()

    def make_layout(self):
        """ Make the layout. """
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
    """ Base class for spacers. """

    def __init__(self):
        """ Constructor. """
        self.figures = [Spacer()]
        self.make_layout()

    def make_layout(self):
        """ Make the layout. """
        self.layout = column(*self.figures)
        return self.layout
