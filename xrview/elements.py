""" ``xrview.elements`` """

import numpy as np
import pandas as pd

from xrview.utils import is_dataarray, is_dataset
from xrview.timeseries.handlers import ResamplingDataHandler


class BaseGlyph(object):
    """ Base class for glyphs. """

    def __init__(self, **glyph_kwargs):

        self.glyph_kwargs = glyph_kwargs


class BaseElement(object):
    """ Base class for elements. """

    def __init__(self, data, name=None, resolution=None):

        # TODO: check if it's better to store a DataArray by default

        if is_dataarray(data):
            self.data = data
        elif is_dataset(data) and len(data.data_vars) == 1:
            self.data = data[data.data_vars[0]]
        else:
            raise ValueError(
                'data must be DataArray or single-variable Dataset')

        if name is None:
            self.name = self.data.name
        else:
            self.name = name
            self.data.name = name

        self.resolution = resolution

        if self.name in self.data.coords:
            self.data = self.data.drop(self.name)
        self.data = self.data.to_dataset()

        self.x = None
        self.handler = None

    def _collect(self, data):
        """ Base method for collect. """

        plot_data = dict()

        for v in data.data_vars:
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                plot_data[v] = data[v].values
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for d in data[dim].values:
                    plot_data[v + '_' + str(d)] = data[v].sel(**{dim: d}).values
            else:
                raise ValueError(v + ' has too many dimensions')

        plot_data['selected'] = np.zeros(data.sizes[self.x], dtype=bool)

        return pd.DataFrame(plot_data, index=data[self.x])

    def collect(self, hooks=None):
        """ Collect plottable data in a pandas DataFrame. """

        data = self.data

        if hooks is not None:
            for h in hooks:
                data = h(data)

        return self._collect(data)

    def attach(self, context):
        """ Attach element to context. """

        self.x = context.x

        if self.resolution is None:
            resolution = context.resolution
        else:
            resolution = self.resolution

        self.handler = ResamplingDataHandler(
            self.collect(), resolution * context.figsize[0],
            context=context, lowpass=context.lowpass)


class CompositeElement(BaseElement):
    """ An element composed of multiple glyphs. """

    def __init__(self, glyphs, data, name=None, resolution=None, **glyph_kwargs):

        super(CompositeElement, self).__init__(data, name, resolution)

        self.glyphs = [g(**glyph_kwargs) for g in glyphs]


class BaseGlyphElement(BaseGlyph, BaseElement):
    """"""

    def __init__(self, data, name=None, resolution=None, **glyph_kwargs):

        BaseElement.__init__(self, data, name, resolution)
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
    """ A line glyph. """

    glyph = 'circle'


class Ray(BaseGlyphElement):
    """ A ray glyph. """

    glyph = 'ray'


class VLines(CompositeElement):
    """ A collection of vertical lines. """

    def __init__(self, data, name=None, resolution=None, **glyph_kwargs):

        default_kwargs = dict(
            length=0, line_width=1, angle_units='deg', color='grey', alpha=0.5)

        default_kwargs.update(glyph_kwargs)

        super(VLines, self).__init__(
            [RayGlyph, RayGlyph], data, name, resolution, **default_kwargs)

        self.glyphs[0].glyph_kwargs['angle'] = 90
        self.glyphs[1].glyph_kwargs['angle'] = 270
