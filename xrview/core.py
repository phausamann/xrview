""" ``xrview.core`` """

import abc

import six

from bokeh.io import output_notebook
from bokeh.io.notebook import show_app
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application

from xrview.utils import is_dataset, is_dataarray, get_notebook_url


@six.add_metaclass(abc.ABCMeta)
class Viewer(object):
    """ Base class for bokeh notebook apps.

    Parameters
    ----------
    data : xarray DataArray or Dataset
        The data to display.
    """

    def __init__(self, data):

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

        self.doc = None
        self.layout = None

    @abc.abstractmethod
    def _make_layout(self):
        """ Make the app layout. """

    def _make_app(self, doc):
        """ Make the app for displaying in a jupyter notebook. """

        self.doc = doc
        self._make_layout()
        self.doc.add_root(self.layout)

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
