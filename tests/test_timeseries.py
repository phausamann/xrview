from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt

from xrview.timeseries.handlers import ResamplingDataHandler
from xrview.timeseries.base import Viewer
from xrview.elements import Line, VLines


class SamplingDataHandlerTests(TestCase):

    def setUp(self):

        self.n_samples = 1000

        data = pd.DataFrame({'y': np.random.random(self.n_samples),
                             'selected': np.zeros(self.n_samples, dtype=bool)})

        self.handler = ResamplingDataHandler(data, self.n_samples)

    def test_get_range(self):

        # integer index
        idx = range(self.n_samples)
        self.handler.data.index = idx
        self.handler.update_data()
        self.handler.update_source()

        assert self.handler.get_range() == (idx[0], idx[-1])
        assert self.handler.get_range(-1, self.n_samples) == (idx[0], idx[-1])
        assert self.handler.get_range(2, 7) == (2, 7)

        # float index
        idx = np.linspace(0, 1, self.n_samples)
        self.handler.data.index = idx
        self.handler.update_data()
        self.handler.update_source()

        assert self.handler.get_range() == (idx[0], idx[-1])
        assert self.handler.get_range(-1, 2) == (idx[0], idx[-1])
        assert self.handler.get_range(0.2, 0.7) == (0.2, 0.7)

        # datetime index
        idx = pd.date_range('2000-01-01', periods=self.n_samples, freq='H')
        t_outer = pd.to_datetime('2001-01-01')
        t_inner = pd.to_datetime('2000-01-02')
        self.handler.data.index = idx
        self.handler.update_data()
        self.handler.update_source()

        assert self.handler.get_range() == (idx[0], idx[-1])
        assert self.handler.get_range(
            0, t_outer.value / 1e6) == (idx[0], idx[-1])
        assert self.handler.get_range(
            t_inner.value / 1e6, t_inner.value / 1e6) == (t_inner, t_inner)


class ViewerTests(TestCase):

    def setUp(self):

        n_samples = 1000
        n_axes = 3

        coord_1 = ['a'] * (n_samples // 2) + ['b'] * (n_samples // 2)
        coord_2 = np.zeros(n_samples)

        self.data = xr.Dataset(
            {'Var_1': (['sample', 'axis'], np.random.rand(n_samples, n_axes)),
             'Var_2': (['sample', 'axis'], np.random.rand(n_samples, n_axes))},
            coords={
                'sample': range(n_samples), 'axis': range(n_axes),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)},
        )

    def test_collect(self):

        v = Viewer(self.data, x='sample')

        data = v.collect()

        self.assertEqual(set(data.columns),
                         {'Var_1_0', 'Var_1_1', 'Var_1_2',
                          'Var_2_0', 'Var_2_1', 'Var_2_2',
                          'selected'})

        npt.assert_allclose(data.iloc[:, :3], v.data.Var_1)

    def test_make_handlers(self):

        v1 = Viewer(self.data, x='sample')
        v1.make_handlers()

        assert len(v1.handlers) == 1

    def test_make_maps(self):

        v1 = Viewer(self.data, x='sample')
        v1.make_handlers()
        v1.make_maps()

        self.assertEqual(list(v1.figure_map.index), [0, 1])

        v2 = Viewer(self.data, x='sample', overlay='data_vars')
        v2.make_handlers()
        v2.make_maps()

        self.assertEqual(list(v2.figure_map.index), [0, 1, 2])

    def test_make_figures(self):

        v1 = Viewer(self.data, x='sample')
        v1.make_handlers()
        v1.make_maps()
        v1.make_figures()

    def test_add_glyphs(self):

        v1 = Viewer(self.data, x='sample')
        v1.make_handlers()
        v1.make_maps()
        v1.make_figures()
        v1.add_glyphs()

    def test_add_tooltips(self):

        v1 = Viewer(self.data, x='sample', tooltips={'sample': '@index'})
        v1.make_handlers()
        v1.make_maps()
        v1.make_figures()
        v1.add_tooltips()

    def test_add_callbacks(self):

        v1 = Viewer(self.data, x='sample')
        v1.make_handlers()
        v1.make_maps()
        v1.make_figures()
        v1.add_callbacks()

    def test_add_figure(self):

        v1 = Viewer(self.data, x='sample')
        v1.add_figure(Line(self.data.coord_2, name='Test'))
        v1.make_layout()

    def test_add_overlay(self):

        v1 = Viewer(self.data, x='sample')
        v1.add_overlay(Line(self.data.coord_2, name='Test'))
        v1.make_layout()

    def test_add_interaction(self):

        pass
