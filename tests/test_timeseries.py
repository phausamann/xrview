from unittest import TestCase

import numpy as np
import xarray as xr

import numpy.testing as npt

from xrview.timeseries import TimeseriesViewer
from xrview.elements import Line, VLine
from xrview.interactions import CoordValSelect


class TimeseriesViewerTests(TestCase):

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

        v = TimeseriesViewer(self.data, x='sample')

        data = v._collect()

        self.assertEqual(set(data.columns),
                         {'Var_1_0', 'Var_1_1', 'Var_1_2',
                          'Var_2_0', 'Var_2_1', 'Var_2_2',
                          'selected'})

        npt.assert_allclose(data.iloc[:, :3], v.data.Var_1)

    def test_make_handlers(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1._make_handlers()

        assert len(v1.handlers) == 1

    def test_make_figures(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1._make_handlers()
        v1._make_maps()
        v1._make_figures()

    def test_add_glyphs(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1._make_handlers()
        v1._make_maps()
        v1._make_figures()
        v1._add_glyphs()

    def test_add_tooltips(self):

        v1 = TimeseriesViewer(
            self.data, x='sample', tooltips={'sample': '@index'})
        v1._make_handlers()
        v1._make_maps()
        v1._make_figures()
        v1._add_tooltips()

    def test_add_callbacks(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1._make_handlers()
        v1._make_maps()
        v1._make_figures()
        v1._add_callbacks()

    def test_add_figure(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1.add_figure(Line(self.data.Var_1, name='Test'))
        v1._make_layout()

    def test_add_overlay(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1.add_overlay(Line(self.data.coord_2, name='Test'))
        v1.add_overlay(VLine(self.data.coord_2[self.data.coord_2 > 0]))
        v1._make_layout()

    def test_add_interaction(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1.add_interaction(CoordValSelect('coord_1'))
        v1._make_layout()

    def test_modify_figure(self):

        v1 = TimeseriesViewer(self.data, x='sample')
        v1.add_figure(Line(self.data.Var_1, name='Test'))
        v1._make_layout()

        v1.modify_figure(0, {'xaxis.axis_label': 'test_label'})

        self.assertEqual(v1.figures[0].xaxis[0].axis_label, 'test_label')
