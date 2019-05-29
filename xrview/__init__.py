# -*- coding: utf-8 -*-

"""Top-level package for xrview."""

__author__ = """Peter Hausamann"""
__email__ = 'peter.hausamann@tum.de'
__version__ = '0.1.0'


from xrview.html import HtmlPlot
from xrview.notebook import NotebookPlot, NotebookViewer


def plot(X, output='html', interactive=False, **kwargs):
    """ Create a plot from xarray data.

    Parameters
    ----------
    X: xarray.DataArray or Dataset
        The data to be plotted.

    output: 'html' or 'notebook', default 'html'
        Whether to show the plot in an HTML file or a Jupyter notebook
        output cell.

    interactive: bool, default False
        If True, create an interactive viewer that supports interactions and
        plotting of large datasets.

    kwargs:
        Keyword arguments to be passed to the plot instance.

    Returns
    -------
    plot: xrview.BasePanel
        A plot instance depending on the options.
    """
    if output == 'html':
        if interactive:
            raise NotImplementedError(
                'Interactive plotting for HTML output is not implemented.')
        else:
            return HtmlPlot(X, **kwargs)
    elif output == 'notebook':
        if interactive:
            return NotebookViewer(X, **kwargs)
        else:
            return NotebookPlot(X, **kwargs)
    else:
        raise ValueError('Unrecognized output mode: ' + output)
