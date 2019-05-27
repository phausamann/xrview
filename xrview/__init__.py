# -*- coding: utf-8 -*-

"""Top-level package for xrview."""

__author__ = """Peter Hausamann"""
__email__ = 'peter.hausamann@tum.de'
__version__ = '0.1.0'


from xrview.html import HtmlPlot
from xrview.notebook import NotebookPlot, NotebookViewer


def plot(X, output='html', interactive=False, **kwargs):
    """"""
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
