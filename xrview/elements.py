""" ``xrview.elements`` """
from xrview.utils import is_dataarray
from xrview.glyphs import get_glyph_list
from xrview.handlers import DataHandler, InteractiveDataHandler, \
    ResamplingDataHandler


class Element(object):

    def __init__(self, glyphs, data, coords=None, name=None):
        """ Base class for elements.

        Parameters
        ----------
        glyphs : xrview.elements.BaseGlyph or iterable thereof
            The glyph (or glyphs) to display.

        data : xarray.DataArray
            The data to display.

        coords : iterable of str, optional
            The coordinates of the DataArray to include. This is necessary
            for composite glyphs such as BoxWhisker.

        name : str, optional
            The name of the DataArray which will be used as the title of the
            figure. If not provided, the name of the DataArray will be used.
        """
        self.glyphs = get_glyph_list(glyphs)

        # TODO: check if it's better to store a DataArray by default
        if is_dataarray(data):
            self.data = data
        else:
            raise ValueError('data must be DataArray')

        self.coords = coords

        if name is None:
            self.name = self.data.name
        else:
            self.name = name
            self.data.name = name

        if self.name in self.data.coords:
            self.data = self.data.drop(self.name)
        self.data = self.data.to_dataset(name=self.data.name or 'Variable')

        self.context = None
        self.handler = None

    def _collect(self, hooks=None):
        """ Collect plottable data in a pandas DataFrame. """
        data = self.data
        if hooks is not None:
            for h in hooks:
                data = h(data)

        return self.context._collect_data(data, coords=self.coords)

    def attach(self, context):
        """ Attach element to context. """
        self.context = context
        self.handler = DataHandler(self._collect())


class InteractiveElement(Element):

    def attach(self, context):
        """ Attach element to context. """
        self.context = context
        self.handler = InteractiveDataHandler(self._collect(), context)


class ResamplingElement(Element):

    def __init__(self, glyphs, data, coords=None, name=None, resolution=None):

        super(ResamplingElement, self).__init__(glyphs, data, coords, name)
        self.resolution = resolution

    def attach(self, context):
        """ Attach element to context. """
        self.context = context
        if self.resolution is None:
            resolution = context.resolution
        else:
            resolution = self.resolution
        self.handler = ResamplingDataHandler(
            self._collect(), resolution * context.figsize[0],
            context=context, lowpass=context.lowpass)
