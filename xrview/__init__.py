# -*- coding: utf-8 -*-

"""Top-level package for xrview."""

__author__ = """Peter Hausamann"""
__email__ = 'peter.hausamann@tum.de'
__version__ = '0.1.0'


from xrview.html import HtmlPlot
from xrview.notebook import \
    NotebookPlot, NotebookViewer, NotebookTimeseriesViewer


def plot(X, output='html', server=False, **kwargs):
    """ Create a plot from xarray data.

    Parameters
    ----------
    X: xarray.DataArray or Dataset
        The data to be plotted.

    output: 'html' or 'notebook', default 'html'
        Whether to show the plot in an HTML file or a Jupyter notebook
        output cell.

    server: bool, default False
        If True, create a bokeh server app that supports interactions and
        plotting of large datasets.

    kwargs:
        Keyword arguments to be passed to the plot instance.

    Returns
    -------
    plot: xrview.BasePanel
        A plot instance depending on the options.
    """
    if output == 'html':
        if server:
            raise NotImplementedError(
                'Server interface for HTML output is not yet implemented.')
        else:
            return HtmlPlot(X, **kwargs)
    elif output == 'notebook':
        if server:
            if 'resolution' in kwargs:
                return NotebookTimeseriesViewer(X, **kwargs)
            else:
                return NotebookViewer(X, **kwargs)
        else:
            return NotebookPlot(X, **kwargs)
    else:
        raise ValueError('Unrecognized output mode: ' + output)
