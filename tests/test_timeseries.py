from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt

from xrview.timeseries.handlers import ResamplingDataHandler
from xrview.timeseries.base import _map_vars_and_dims, Viewer


class SamplingDataHandlerTests(TestCase):

    def setUp(self):

        self.n_samples = 1000

        data = pd.DataFrame({'y': np.random.random(self.n_samples)})

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


class MiscTests(TestCase):

    def test_map_vars_and_dims(self):

        ds1 = xr.Dataset(
                {'Var_1': (['sample'], np.random.rand(10)),
                 'Var_2': (['sample', 'axis'], np.random.rand(10, 3)),
                 'Var_3': (['sample', 'feat'], np.random.rand(10, 10))},
            coords={'axis': range(3), 'feat': range(10)})

        ds2 = xr.Dataset(
                {'Var_1': (['sample', 'axis'], np.random.rand(10, 3)),
                 'Var_2': (['sample', 'axis'], np.random.rand(10, 3))},
            coords={'axis': range(3)})

        with self.assertRaises(ValueError):
            _map_vars_and_dims(ds1, 'time', 'dims')

        self.assertEqual(_map_vars_and_dims(ds1, 'sample', 'dims'),
                         {'Var_1': None,
                          'Var_2': tuple(range(3)),
                          'Var_3': tuple(range(10))})

        with self.assertRaises(ValueError):
            _map_vars_and_dims(ds1, 'sample', 'data_vars')

        self.assertEqual(_map_vars_and_dims(ds2, 'sample', 'data_vars'),
                         {0: ('Var_1', 'Var_2'),
                          1: ('Var_1', 'Var_2'),
                          2: ('Var_1', 'Var_2')})


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
        h1 = v1.make_handlers()

        assert None in h1

    def test_make_figures(self):

        v1 = Viewer(self.data, x='sample')
        v1.handlers = v1.make_handlers()
        f1 = v1.make_figures()

        assert len(np.unique([f._id for f in f1.figure])) == 2

        v2 = Viewer(self.data, x='sample', overlay='var')
        v2.handlers = v2.make_handlers()
        f2 = v2.make_figures()

        assert len(np.unique([f._id for f in f2.figure])) == 3

    def test_add_glyphs(self):

        v1 = Viewer(self.data, x='sample')
        v1.handlers = v1.make_handlers()
        v1.figure_map = v1.make_figures()
        v1.add_glyphs(v1.figure_map)

    def test_add_tooltips(self):

        pass

    def test_add_callbacks(self):

        pass

    def test_add_figure(self):

        pass

    def test_add_overlay(self):

        pass

    def test_add_interaction(self):

        pass
