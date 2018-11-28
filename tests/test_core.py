from unittest import TestCase

import numpy as np
import xarray as xr

from numpy import testing as npt

from xrview.core import BasePlot, BaseViewer
from xrview.timeseries import TimeseriesViewer
from xrview.elements import Line, Circle, VBar, VLine
from xrview.interactions import CoordValSelect


class BasePlotTests(TestCase):

    cls = BasePlot

    def setUp(self):
        n_samples = 1000
        n_axes = 3

        coord_1 = ['a'] * (n_samples // 2) + ['b'] * (n_samples // 2)
        coord_2 = np.zeros(n_samples)
        coord_2[5::10] = 1

        self.data = xr.Dataset(
            {'Var_1': (['sample', 'axis'], np.random.rand(n_samples, n_axes)),
             'Var_2': (['sample', 'axis'], np.random.rand(n_samples, n_axes))},
            coords={
                'sample': range(n_samples), 'axis': range(n_axes),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)},
        )

    def test_collect(self):
        v = self.cls(self.data, x='sample')
        data = v._collect()
        npt.assert_allclose(data.iloc[:, :3], v.data.Var_1)
        self.assertEqual(set(data.columns),
                         {'Var_1_0', 'Var_1_1', 'Var_1_2',
                          'Var_2_0', 'Var_2_1', 'Var_2_2',
                          'selected'})

    def test_make_handlers(self):
        v = self.cls(self.data, x='sample')
        v._make_handlers()
        assert len(v.handlers) == 1

    def test_make_maps(self):
        v1 = self.cls(self.data, x='sample')
        v1._make_handlers()
        v1._make_maps()

        self.assertTrue(all(v1.figure_map.index == range(2)))
        self.assertTrue(all(v1.glyph_map.index == range(6)))
        self.assertTrue(
            all(v1.glyph_map.figure[v1.glyph_map['var'] == 'Var_1'] == 0))
        self.assertTrue(
            all(v1.glyph_map.figure[v1.glyph_map['var'] == 'Var_2'] == 1))
        self.assertEqual(
            [a['y'] for a in v1.glyph_map.glyph_kwargs],
            ['Var_1_0', 'Var_1_1', 'Var_1_2', 'Var_2_0', 'Var_2_1', 'Var_2_2'])

        v2 = self.cls(self.data, x='sample', overlay='data_vars')
        v2._make_handlers()
        v2._make_maps()

        self.assertTrue(all(v2.figure_map.index == range(3)))
        self.assertTrue(all(v2.glyph_map.index == range(6)))
        self.assertTrue(all(v2.glyph_map.figure == v2.glyph_map.dim_val))
        self.assertEqual(
            [a['y'] for a in v2.glyph_map.glyph_kwargs],
            ['Var_1_0', 'Var_1_1', 'Var_1_2', 'Var_2_0', 'Var_2_1', 'Var_2_2'])

    def test_make_figures(self):
        v = self.cls(self.data, x='sample')
        v._make_handlers()
        v._make_maps()
        v._make_figures()

    def test_add_glyphs(self):
        v = self.cls(self.data, x='sample')
        v._make_handlers()
        v._make_maps()
        v._make_figures()
        v._add_glyphs()

    def test_add_tooltips(self):
        v = self.cls(self.data, x='sample', tooltips={'sample': '@index'})
        v._make_handlers()
        v._make_maps()
        v._make_figures()
        v._add_tooltips()

    def test_make_layout(self):
        v = self.cls(self.data, x='sample')
        v.make_layout()

    def test_custom_glyphs(self):
        v = self.cls(self.data, x='sample', glyphs=[Circle(), VBar(0.001)])
        v.make_layout()

    def test_add_figure(self):
        v = self.cls(self.data, x='sample')
        v.add_figure(Line(), self.data.Var_1, name='Test')
        v.make_layout()

    def test_add_overlay(self):
        v = self.cls(self.data, x='sample')
        v.add_overlay(Line(), self.data.coord_2, onto='Var_1')
        v.add_overlay(Line(), self.data.coord_2, name='Test', onto=1)
        v.add_overlay(VLine(), self.data.coord_2[self.data.coord_2 > 0])
        v.make_layout()

    def test_modify_figures(self):
        v = self.cls(self.data, x='sample')
        v.add_figure(Line(), self.data.Var_1, name='Test')
        v.make_layout()
        v.modify_figures({'xaxis.axis_label': 'test_label'}, 0)
        self.assertEqual(v.figures[0].xaxis[0].axis_label, 'test_label')


class BaseViewerTests(BasePlotTests):

    cls = BaseViewer

    def test_make_doc(self):
        v = self.cls(self.data, x='sample')
        v.make_layout()
        v.make_doc()

    def test_add_interaction(self):
        v = self.cls(self.data, x='sample')
        v.add_interaction(CoordValSelect('coord_1'))
        v.make_layout()

    def test_update_handlers(self):
        v = self.cls(self.data, x='sample')
        v.make_layout()
        v.make_doc()
        v.update_handlers()


class TimeseriesViewerTests(BaseViewerTests):

    cls = TimeseriesViewer

    def test_on_reset(self):
        v = self.cls(self.data, x='sample')
        v.make_layout()
        v.make_doc()
        v.on_reset(None)