""""""
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import output_notebook, show, curdoc
from bokeh.layouts import row

from xrview.core import BasePanel, BasePlot, BaseViewer, GridPlot, SpacerPanel
from xrview.notebook.utils import get_notebook_url
from xrview.timeseries import TimeseriesViewer


class NotebookPanel(BasePanel):
    """ Base class for notebook panels. """

    def show(self, remake_layout=False):
        """ Show the plot in a jupyter notebook.

        Parameters
        ----------
        remake_layout : bool, default False
            If True, call ``make_layout`` even when the layout has already
            been created. Note that any changes made by ``modify_figures``
            will be omitted.
        """
        curdoc().theme = self.theme
        output_notebook(hide_banner=True)
        if self.layout is None or remake_layout:
            self.make_layout()
        show(self.layout)


class NotebookPlot(BasePlot, NotebookPanel):
    """ Base class for notebook plots. """


class NotebookServer(BasePlot):
    """ Base class for bokeh notebook servers. """

    def _modify_doc(self, doc):
        """ Make the app for displaying in a jupyter notebook. """
        self.doc = doc
        self.doc.theme = self.theme
        self.doc.add_root(row(self.layout))

    def show(self, notebook_url=None, port=0, remake_layout=False,
             verbose=False):
        """ Show the app in a jupyter notebook.

        Parameters
        ----------
        notebook_url : str, optional
            The URL of the notebook. Will be determined automatically if not
            specified.

        port : int, default 0
            The port over which the app will be served. Chosen randomly if
            set to 0.

        remake_layout : bool, default False
            If True, call ``make_layout`` even when the layout has already
            been created. Note that any changes made by ``modify_figures``
            will be omitted.

        verbose : bool, default False
            If True, create the document once again outside of show_app in
            order to show errors.
        """
        if notebook_url is None:
            notebook_url = get_notebook_url()

        output_notebook(hide_banner=True)

        if verbose:
            self.make_layout()
            Application(FunctionHandler(self._modify_doc)).create_document()

        if self.layout is None or remake_layout or verbose:
            self.make_layout()

        show(self._modify_doc, None, notebook_url=notebook_url, port=port)


class NotebookViewer(BaseViewer, NotebookServer):
    """ Base class for notebook viewers. """


class NotebookGridPlot(GridPlot, NotebookPanel):
    """ A notebook grid plot. """


class NotebookSpacer(SpacerPanel):
    """ A notebook spacer. """


class NotebookTimeseriesViewer(TimeseriesViewer, NotebookServer):
    """ A notebook time-series viewer. """
