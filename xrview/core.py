""" ``xrview.core`` """

import abc

import six

from bokeh.io import output_notebook
from bokeh.io.notebook import show_app
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import row

from xrview.utils import is_dataset, is_dataarray, get_notebook_url


@six.add_metaclass(abc.ABCMeta)
class Viewer(object):
    """ Base class for bokeh notebook apps.

    Parameters
    ----------
    data : xarray DataArray or Dataset
        The data to display.

    figsize : iterable
        The size of the figure in pixels.
    """

    def __init__(self, data, figsize):

        # check data
        if is_dataarray(data):
            if data.name is None:
                self.data = data.to_dataset(name='Data')
            else:
                self.data = data.to_dataset()
        elif is_dataset(data):
            self.data = data
        else:
            raise ValueError('data must be xarray DataArray or Dataset.')

        self.figsize = figsize

        self.doc = None
        self.layout = None

    @abc.abstractmethod
    def _make_layout(self):
        """ Make the app layout. """

    def _make_app(self, doc):
        """ Make the app for displaying in a jupyter notebook. """

        self.doc = doc
        self._make_layout()
        self.doc.add_root(row(self.layout))

    def _inplace_update(self):
        """ Update the current layout in place. """

        self.doc.roots[0].children[0] = self.layout

    def update_inplace(self, other):
        """ Update this instance with the properties of another viewer instance.

        Parameters
        ----------
        other : xrview.core.Viewer
            The instance that replaces the current instance.
        """

        doc = self.doc
        self.__dict__ = other.__dict__ # TODO: make this safer
        self._make_layout()
        self.doc = doc

        self.doc.add_next_tick_callback(self._inplace_update)

    def copy(self, with_data=False):
        """ Create a copy of this instance.

        Parameters
        ----------
        with_data : bool, default False
            If true, also copy the data.

        Returns
        -------
        new : xrview.core.Viewer
            The copied object.
        """

        from copy import copy

        new = self.__new__(type(self))

        new.__dict__ = {k: (copy(v) if (k != 'data' or with_data) else v)
                        for k, v in self.__dict__.items()}

        return new

    def show(self, notebook_url=None, port=0, verbose=True):
        """ Show the app in a jupyter notebook.

        Parameters
        ----------
        notebook_url : str, optional
            The URL of the notebook. Will be determined automatically if not
            specified.

        port : int, default 0
            The port over which the app will be served. Chosen randomly if
            set to 0.

        verbose : bool, default True
            If True, create the document once again outside of show_app in
            order to show errors.
        """

        if notebook_url is None:
            notebook_url = get_notebook_url()

        output_notebook(hide_banner=True)
        app = Application(FunctionHandler(self._make_app))

        if verbose:
            app.create_document()

        show_app(app, None, notebook_url=notebook_url, port=port)
