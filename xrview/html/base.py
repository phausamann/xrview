""""""
from bokeh.io import output_file, show

from xrview.core import BasePanel, BasePlot, GridPlot, SpacerPanel


class HtmlPanel(BasePanel):
    """ Base class for HTML panels. """

    def show(self, filename, remake_layout=False):
        """ Show the plot in an HTML file.

        Parameters
        ----------
        remake_layout : bool, default False
            If True, call ``make_layout`` even when the layout has already
            been created. Note that any changes made by ``modify_figures``
            will be omitted.
        """
        output_file(filename)
        if self.layout is None or remake_layout:
            self.make_layout()
        show(self.layout)


class HtmlPlot(BasePlot, HtmlPanel):
    """ Base class for HTML plots. """


class HtmlGridPlot(GridPlot, HtmlPanel):
    """ An HTML grid plot. """


class HtmlSpacer(SpacerPanel):
    """ An HTML spacer. """
