""" xrview.glyphs """
from bokeh.models import Band as _Band
from bokeh.models import Whisker as _Whisker

from xrview.utils import MappingProxyType

__all__ = [
    "Line",
    "Circle",
    "Diamond",
    "Square",
    "Triangle",
    "Ray",
    "HBar",
    "VBar",
    "Rect",
    "Whisker",
    "Band",
    "VLine",
    "ErrorLine",
    "ErrorCircle",
    "BoxWhisker",
    "get_glyph",
    "get_glyph_list",
]


# -- Glyphs -- #
class BaseGlyph(object):
    """ Base class for glyphs. """

    method = None
    default_kwargs = MappingProxyType({})

    def __init__(self, x_arg="x", y_arg="y", **kwargs):
        """ Constructor.

        Parameters
        ----------
        x_arg: str, default 'x'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'y'
            The glyph argument associated with y-axis values in the data.

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        self.x_arg = x_arg
        self.y_arg = y_arg
        self.glyph_kwargs = dict(self.default_kwargs)
        self.glyph_kwargs.update(kwargs)


class Line(BaseGlyph):
    """ A line glyph. """

    method = "line"


class Circle(BaseGlyph):
    """ A circle glyph. """

    method = "circle"


class Diamond(BaseGlyph):
    """ A diamond glyph. """

    method = "diamond"


class Square(BaseGlyph):
    """ A square glyph. """

    method = "square"


class Triangle(BaseGlyph):
    """ A triangle glyph. """

    method = "triangle"


class Ray(BaseGlyph):
    """ A ray glyph. """

    method = "ray"
    default_kwargs = MappingProxyType({"length": 0, "angle": 0})


class HBar(BaseGlyph):
    """ A horizontal bar glyph. """

    method = "hbar"

    def __init__(self, height, x_arg="right", y_arg="y", other=0.0, **kwargs):
        """ Constructor.

        Parameters
        ----------
        height : str or float
            The name of a coordinate or a fixed value that will represent
            the height of the bar.

        x_arg: str, default 'right'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'y'
            The glyph argument associated with y-axis values in the data.

        other: str or float, default 0.
            The name of a coordinate or a fixed value that will represent
            the other side of the bar (i.e. the left side when x_arg='right').

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        if x_arg == "left":
            if "right" not in kwargs:
                kwargs.update({"right": other})
        elif x_arg == "right":
            if "left" not in kwargs:
                kwargs.update({"left": other})
        else:
            raise ValueError("Unrecognized x_arg")
        super(HBar, self).__init__(x_arg, y_arg, height=height, **kwargs)


class VBar(BaseGlyph):
    """ A vertical bar glyph. """

    method = "vbar"

    def __init__(self, width, x_arg="x", y_arg="top", other=0.0, **kwargs):
        """ Constructor.

        Parameters
        ----------
        width : str or float
            The name of a coordinate or a fixed value that will represent
            the width of the bar.

        x_arg: str, default 'x'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'top'
            The glyph argument associated with y-axis values in the data.

        other: str or float, default 0.
            The name of a coordinate or a fixed value that will represent
            the other side of the bar (i.e. the bottom side when y_arg='top').

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        if y_arg == "top":
            if "bottom" not in kwargs:
                kwargs.update({"bottom": other})
        elif y_arg == "bottom":
            if "top" not in kwargs:
                kwargs.update({"top": other})
        else:
            raise ValueError("Unrecognized x_arg")
        super(VBar, self).__init__(x_arg, y_arg, width=width, **kwargs)


class Rect(BaseGlyph):
    """ A rectangle glyph. """

    method = "rect"

    def __init__(self, width, height, x_arg="x", y_arg="y", **kwargs):
        """ Constructor.

        Parameters
        ----------
        width : str or float
            The name of a coordinate or a fixed value that will represent
            the width of the rectangle.

        height : str or float
            The name of a coordinate or a fixed value that will represent
            the height of the rectangle.

        x_arg: str, default 'x'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'y'
            The glyph argument associated with y-axis values in the data.

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        super(Rect, self).__init__(
            x_arg, y_arg, width=width, height=height, **kwargs
        )


# -- Compat Glyphs -- #
class BaseGlyphCompat(BaseGlyph):
    """ Base class for annotations that are currently treated like glyphs. """


class Whisker(BaseGlyphCompat):
    """ A whisker annotation. """

    method = _Whisker

    def __init__(self, x_arg="base", y_arg="upper", other=0.0, **kwargs):
        """ Constructor.

        Parameters
        ----------
        x_arg: str, default 'base'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'upper'
            The glyph argument associated with y-axis values in the data.

        other: str or float, default 0.
            The name of a coordinate or a fixed value that will represent
            the other end of the whisker (i.e. the lower end when
            y_arg='upper').

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        if y_arg == "upper":
            if "lower" not in kwargs:
                kwargs.update({"lower": other})
        elif y_arg == "lower":
            if "upper" not in kwargs:
                kwargs.update({"upper": other})
        else:
            raise ValueError("Unrecognized y_arg")
        super(Whisker, self).__init__(x_arg, y_arg, **kwargs)


class Band(BaseGlyphCompat):
    """ A band annotation. """

    method = _Band

    def __init__(self, x_arg="base", y_arg="upper", other=0.0, **kwargs):
        """ Constructor.

        Parameters
        ----------
        x_arg: str, default 'base'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'upper'
            The glyph argument associated with y-axis values in the data.

        other: str or float, default 0.
            The name of a coordinate or a fixed value that will represent
            the other end of the whisker (i.e. the lower end when
            y_arg='upper').

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        if y_arg == "upper":
            if "lower" not in kwargs:
                kwargs.update({"lower": other})
        elif y_arg == "lower":
            if "upper" not in kwargs:
                kwargs.update({"upper": other})
        else:
            raise ValueError("Unrecognized y_arg")
        super(Band, self).__init__(x_arg, y_arg, **kwargs)


# -- Composite Glyphs -- #
class CompositeGlyph(object):
    """ A glyph composed of multiple glyphs. """

    default_kwargs = MappingProxyType({})

    def __init__(self, glyphs, x_arg="x", y_arg="y", **kwargs):
        """ Constructor.

        Parameters
        ----------
        x_arg: str, default 'x'
            The glyph argument associated with x-axis values in the data.

        y_arg: str, default 'y'
            The glyph argument associated with y-axis values in the data.

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        glyph_kwargs = dict(self.default_kwargs)
        glyph_kwargs.update(kwargs)
        self.glyphs = [
            g(x_arg=x_arg, y_arg=y_arg, **glyph_kwargs) for g in glyphs
        ]

    def __iter__(self):
        return iter(self.glyphs)

    def __len__(self):
        return len(self.glyphs)


class VLine(CompositeGlyph):
    """ A collection of vertical lines. """

    default_kwargs = MappingProxyType(
        {
            "length": 0,
            "line_width": 1,
            "angle_units": "deg",
            "color": "grey",
            "alpha": 0.5,
        }
    )

    def __init__(self, x_arg="x", y_arg="y", **kwargs):
        super(VLine, self).__init__(
            [Ray, Ray], x_arg=x_arg, y_arg=y_arg, **kwargs
        )
        self.glyphs[0].glyph_kwargs["angle"] = 90
        self.glyphs[1].glyph_kwargs["angle"] = 270


class ErrorLine(CompositeGlyph):
    """  A line with an error bar. """

    def __init__(self, lower, upper, **kwargs):
        """ Constructor.

        Parameters
        ----------
        lower: str or float
            The name of a coordinate or a fixed value that will represent
            the lower end of the error bar.

        upper: str or float
            The name of a coordinate or a fixed value that will represent
            the upper end of the error bar.

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        self.glyphs = [Line(**kwargs)]
        kwargs.pop("color", None)
        kwargs.pop("legend", None)
        self.glyphs.append(Whisker(lower=lower, upper=upper, **kwargs))


class ErrorCircle(CompositeGlyph):
    """  A circle with an error bar. """

    def __init__(self, lower, upper, **kwargs):
        """ Constructor.

        Parameters
        ----------
        lower: str or float
            The name of a coordinate or a fixed value that will represent
            the lower end of the error bar.

        upper: str or float
            The name of a coordinate or a fixed value that will represent
            the upper end of the error bar.

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        self.glyphs = [Circle(**kwargs)]
        kwargs.pop("color", None)
        kwargs.pop("legend", None)
        self.glyphs.append(Whisker(lower=lower, upper=upper, **kwargs))


class BoxWhisker(CompositeGlyph):
    """ A box-whisker glyph. """

    default_kwargs = MappingProxyType({"line_color": "black"})

    def __init__(self, width, q_lower, w_lower, q_upper, w_upper, **kwargs):
        """ Constructor.

        Parameters
        ----------
        width : str or float
            The name of a coordinate or a fixed value that will represent
            the width of the box.

        q_lower: str or float
            The name of a coordinate or a fixed value that will represent
            the lower end of the box.

        w_lower: str or float
            The name of a coordinate or a fixed value that will represent
            the lower end of the error bar.

        q_upper: str or float
            The name of a coordinate or a fixed value that will represent
            the upper end of the box.

        w_upper: str or float
            The name of a coordinate or a fixed value that will represent
            the upper end of the error bar.

        kwargs:
            Additional keyword arguments to be passed to the underlying
            bokeh glyph(s).
        """
        glyph_kwargs = dict(self.default_kwargs)
        glyph_kwargs.update(kwargs)
        self.glyphs = [
            VBar(width, bottom=q_lower, **glyph_kwargs),
            VBar(width, y_arg="bottom", top=q_upper, **glyph_kwargs),
        ]
        glyph_kwargs.pop("color", None)
        glyph_kwargs.pop("legend", None)
        self.glyphs.append(
            Whisker(lower=w_lower, upper=w_upper, **glyph_kwargs)
        )


def get_glyph(name, *args, **kwargs):
    """ Get a glyph instance by name.

    Parameters
    ----------
    name : str
        The name of the glyph class.

    Returns
    -------
    glyph : BaseGlyph
        An instance of the corresponding glyph class.
    """
    glyphs = {
        "line": Line,
        "circle": Circle,
        "diamond": Diamond,
        "square": Square,
        "triangle": Triangle,
        "ray": Ray,
        "hbar": HBar,
        "vbar": VBar,
        "rect": Rect,
        "vline": VLine,
        "whisker": Whisker,
        "band": Band,
        "errorline": ErrorLine,
        "errorcircle": ErrorCircle,
        "boxwhisker": BoxWhisker,
    }

    try:
        return glyphs[name.lower()](*args, **kwargs)
    except KeyError:
        raise ValueError("Unrecognized or unsupported glyph: " + name)


def get_glyph_list(glyphs):
    """ Get a list of glyphs from str, single glyph or iterable. """
    if isinstance(glyphs, str):
        glyphs = [get_glyph(glyphs)]
    else:
        try:
            iter(glyphs)
        except TypeError:
            glyphs = [glyphs]
        else:
            glyphs = [
                get_glyph(g) if isinstance(g, str) else g for g in glyphs
            ]

    return glyphs
