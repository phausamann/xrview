from unittest import TestCase

import numpy as np

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from xrview.glyphs import Line, Circle, Ray, HBar, VBar, Rect, Whisker, Band, \
    VLine, get_glyph


class GlyphTests(TestCase):

    def setUp(self):
        """"""
        self.figure = figure()
        self.source = ColumnDataSource(
            {'x': np.arange(10), 'y': np.arange(10)})

    def _test_glyph(self, glyph):
        """"""
        kwargs = glyph.glyph_kwargs
        kwargs.update(
            {glyph.x_arg: 'x', glyph.y_arg: 'y', 'source': self.source})
        getattr(self.figure, glyph.method)(**kwargs)

    def _test_composite_glyph(self, composite):
        """"""
        for glyph in composite.glyphs:
            kwargs = glyph.glyph_kwargs
            kwargs.update(
                {glyph.x_arg: 'x', glyph.y_arg: 'y', 'source': self.source})
            getattr(self.figure, glyph.method)(**kwargs)

    def test_line(self):
        """"""
        self._test_glyph(Line())

    def test_circle(self):
        """"""
        self._test_glyph(Circle())

    def test_ray(self):
        """"""
        self._test_glyph(Ray())

    def test_vbar(self):
        """"""
        self._test_glyph(VBar(width=1.))
        self._test_glyph(VBar(width=1., y_arg='bottom'))

    def test_hbar(self):
        """"""
        self._test_glyph(HBar(height=1.))
        self._test_glyph(HBar(height=1., x_arg='left'))

    def test_rect(self):
        """"""
        self._test_glyph(Rect(width=1., height=1.))

    def test_vline(self):
        """"""
        self._test_composite_glyph(VLine())

    def test_get_glyph(self):
        """"""
        assert isinstance(get_glyph('line'), Line)
        assert isinstance(get_glyph('circle'), Circle)
        assert isinstance(get_glyph('ray'), Ray)
        assert isinstance(get_glyph('vbar', width=1.), VBar)
        assert isinstance(get_glyph('hbar', height=1.), HBar)
        assert isinstance(get_glyph('rect', width=1., height=1.), Rect)
        assert isinstance(get_glyph('vline'), VLine)
        with self.assertRaises(ValueError):
            get_glyph('not_a_glyph')


class GlyphCompatTests(TestCase):

    def setUp(self):
        """"""
        self.figure = figure()
        self.source = ColumnDataSource(
            {'x': np.arange(10), 'y': np.arange(10)})

    def _test_glyph(self, glyph):
        """"""
        kwargs = glyph.glyph_kwargs
        kwargs.update(
            {glyph.x_arg: 'x', glyph.y_arg: 'y', 'source': self.source})
        self.figure.add_layout(glyph.method(**kwargs))

    def test_whisker(self):
        """"""
        self._test_glyph(Whisker())

    def test_band(self):
        """"""
        self._test_glyph(Band())