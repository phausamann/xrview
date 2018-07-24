""" ``xrview.elements`` """

from xrview.utils import is_dataarray, is_dataset
from xrview.timeseries.handlers import ResamplingDataHandler


class BaseGlyph(object):
    """ Base class for glyphs. """

    def __init__(self, **glyph_kwargs):

        self.glyph_kwargs = glyph_kwargs

        self.x_arg = 'x'
        self.y_arg = 'y'


class BaseElement(object):
    """ Base class for elements. """

    def __init__(self, data, coords=None, name=None, resolution=None):

        # TODO: check if it's better to store a DataArray by default

        if is_dataarray(data):
            self.data = data
        elif is_dataset(data) and len(data.data_vars) == 1:
            self.data = data[data.data_vars[0]]
        else:
            raise ValueError(
                'data must be DataArray or single-variable Dataset')

        self.coords = coords

        if name is None:
            self.name = self.data.name
        else:
            self.name = name
            self.data.name = name

        self.resolution = resolution

        if self.name in self.data.coords:
            self.data = self.data.drop(self.name)
        self.data = self.data.to_dataset()

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

        if self.resolution is None:
            resolution = context.resolution
        else:
            resolution = self.resolution

        self.handler = ResamplingDataHandler(
            self._collect(), resolution * context.figsize[0],
            context=context, lowpass=context.lowpass)


class CompositeElement(BaseElement):
    """ An element composed of multiple single elements. """

    def __init__(self, glyphs, data, name=None, resolution=None, **glyph_kwargs):

        super(CompositeElement, self).__init__(data, name, resolution)

        self.glyphs = [g(**glyph_kwargs) for g in glyphs]


class BaseGlyphElement(BaseGlyph, BaseElement):
    """"""

    def __init__(self, data, coords=None, name=None, resolution=None,
                 **glyph_kwargs):

        BaseElement.__init__(self, data, coords, name, resolution)
        BaseGlyph.__init__(self, **glyph_kwargs)


class LineGlyph(BaseGlyph):
    """ A line glyph. """

    glyph = 'line'


class CircleGlyph(BaseGlyph):
    """ A line glyph. """

    glyph = 'circle'


class RayGlyph(BaseGlyph):
    """ A ray glyph. """

    glyph = 'ray'


class Line(BaseGlyphElement):
    """ A line glyph. """

    glyph = 'line'


class Circle(BaseGlyphElement):
    """ A circle glyph. """

    glyph = 'circle'


class Ray(BaseGlyphElement):
    """ A ray glyph. """

    glyph = 'ray'


class VLine(CompositeElement):
    """ A collection of vertical lines. """

    def __init__(self, data, name=None, resolution=None, **glyph_kwargs):

        default_kwargs = dict(
            length=0, line_width=1, angle_units='deg', color='grey', alpha=0.5)

        default_kwargs.update(glyph_kwargs)

        super(VLine, self).__init__(
            [RayGlyph, RayGlyph], data, name, resolution, **default_kwargs)

        self.glyphs[0].glyph_kwargs['angle'] = 90
        self.glyphs[1].glyph_kwargs['angle'] = 270


class HBar(BaseGlyphElement):
    """ An HBar glyph. """

    glyph = 'hbar'

    def __init__(self, data, coords=None, name=None, resolution=None,
                 **glyph_kwargs):

        super(HBar, self).__init__(
            data, coords, name, resolution, **glyph_kwargs)

        self.x_arg = 'left'
