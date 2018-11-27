""" ``xrview.elements`` """

from xrview.utils import is_dataarray, is_dataset
from xrview.handlers import DataHandler, InteractiveDataHandler, \
    ResamplingDataHandler


# -- Glyphs -- #
class BaseGlyph(object):
    """

    Parameters
    ----------
    x_arg
    y_arg
    glyph_kwargs
    """
    method = None

    def __init__(self, x_arg='x', y_arg='y', **kwargs):
        self.x_arg = x_arg
        self.y_arg = y_arg
        self.glyph_kwargs = kwargs


class Line(BaseGlyph):
    """ A line glyph. """
    __doc__ = BaseGlyph.__doc__
    method = 'line'


class Circle(BaseGlyph):
    """ A circle glyph. """
    __doc__ = BaseGlyph.__doc__
    method = 'circle'


class Ray(BaseGlyph):
    """ A ray glyph. """
    __doc__ = BaseGlyph.__doc__
    method = 'ray'


class HBar(BaseGlyph):
    """

    Parameters
    ----------
    x_arg
    y_arg
    other
    glyph_kwargs
    """
    method = 'hbar'

    def __init__(self, height, x_arg='right', y_arg='y', other=0., **kwargs):
        if x_arg == 'left':
            kwargs.update({'right': other})
        elif x_arg == 'right':
            kwargs.update({'left': other})
        else:
            raise ValueError('Unrecognized x_arg')
        super(HBar, self).__init__(
            x_arg, y_arg, height=height, **kwargs)


class VBar(BaseGlyph):
    """

    Parameters
    ----------
    x_arg
    y_arg
    other
    glyph_kwargs
    """
    method = 'vbar'

    def __init__(self, width, x_arg='x', y_arg='top', other=0., **kwargs):
        if y_arg == 'top':
            kwargs.update({'bottom': other})
        elif y_arg == 'bottom':
            kwargs.update({'top': other})
        else:
            raise ValueError('Unrecognized x_arg')
        super(VBar, self).__init__(x_arg, y_arg, width=width, **kwargs)


class Rect(BaseGlyph):
    """

    Parameters
    ----------
    width
    height
    x_arg
    y_arg
    glyph_kwargs
    """
    method = 'rect'

    def __init__(self, width, height, x_arg='x', y_arg='y', **kwargs):
        super(Rect, self).__init__(
            x_arg, y_arg, width=width, height=height, **kwargs)


# -- Composite Glyphs -- #
class CompositeGlyph(object):
    """ A glyph composed of multiple glyphs. """

    def __init__(self, glyphs, x_arg='x', y_arg='y', **kwargs):
        self.glyphs = [g(x_arg=x_arg, y_arg=y_arg, **kwargs) for g in glyphs]

    def __iter__(self):
        return iter(self.glyphs)

    def __len__(self):
        return len(self.glyphs)


class VLine(CompositeGlyph):
    """ A collection of vertical lines. """

    def __init__(self, x_arg='x', y_arg='y', **kwargs):

        default_kwargs = dict(
            length=0, line_width=1, angle_units='deg', color='grey', alpha=0.5)
        default_kwargs.update(kwargs)

        super(VLine, self).__init__(
            [Ray, Ray], x_arg=x_arg, y_arg=y_arg, **default_kwargs)

        self.glyphs[0].glyph_kwargs['angle'] = 90
        self.glyphs[1].glyph_kwargs['angle'] = 270


def get_glyph(name, **kwargs):
    """

    Parameters
    ----------
    name : str
        The name of the glyph class.

    Returns
    -------
    glyph : BaseGlyph
        An instance of the corresponding glyph class.
    """
    if name == 'line':
        return Line(**kwargs)
    elif name == 'circle':
        return Circle(**kwargs)
    elif name == 'ray':
        return Ray(**kwargs)
    elif name == 'hbar':
        return HBar(**kwargs)
    elif name == 'vbar':
        return VBar(**kwargs)
    elif name == 'rect':
        return Rect(**kwargs)
    else:
        raise ValueError('Unrecognized or unsupported glyph: ' + name)


# -- Elements -- #
class Element(object):
    """ Base class for elements.

    Parameters
    ----------
    glyphs :
    data :
    coords :
    name :
    """

    def __init__(self, glyphs, data, coords=None, name=None):

        try:
            iter(glyphs)
        except TypeError:
            self.glyphs = [glyphs]
        else:
            self.glyphs = [g for g in glyphs]

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
        self.handler = DataHandler(self._collect())


class InteractiveElement(Element):

    def attach(self, context):
        """ Attach element to context. """
        self.context = context
        self.handler = InteractiveDataHandler(self._collect())


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
