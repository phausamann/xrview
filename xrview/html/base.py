""""""
from bokeh.plotting import output_file, show, curdoc

from xrview.core import BasePanel, BasePlot, GridPlot, SpacerPanel


class HtmlPanel(BasePanel):
    """ Base class for HTML panels. """

    def show(self, filename=None, remake_layout=False):
        """ Show the plot in an HTML file.

        Parameters
        ----------
        filename : str, optional
            If specified, save the plot to this HTML file.
        remake_layout : bool, default False
            If True, call ``make_layout`` even when the layout has already
            been created. Note that any changes made by ``modify_figures``
            will be omitted.
        """
        curdoc().theme = self.theme
        if filename is not None:
            output_file(filename)
        if self.layout is None or remake_layout:
            self.make_layout()
        show(self.layout)


class HtmlPlot(BasePlot, HtmlPanel):
    """ Base class for HTML plots.

    Examples
    --------
    .. bokeh-plot:: ../examples/html/minimal_example.py
        :source-position: none

    .. code-block:: python

        import numpy as np
        import xarray as xr
        from xrview.html import HtmlPlot

        x = np.linspace(0, 1, 100)
        y = np.sqrt(x)
        da = xr.DataArray(y, coords={'x': x}, dims='x')

        plot = HtmlPlot(da, x='x')
        plot.show()
    """


class HtmlGridPlot(GridPlot, HtmlPanel):
    """ An HTML grid plot. """


class HtmlSpacer(SpacerPanel):
    """ An HTML spacer. """
