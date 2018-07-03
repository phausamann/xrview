from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt

from xrview.timeseries.handlers import ResamplingDataHandler
from xrview.timeseries.base import Viewer


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

        v = Viewer(self.data, x='sample', overlay='axis', stack='data_vars')

        data = v.collect()

        assert set(data.columns) == {
            '0_Var_1', '1_Var_1', '2_Var_1',
            '0_Var_2', '1_Var_2', '2_Var_2',
            'selected'}

        npt.assert_allclose(data.iloc[:, :3], v.data.Data)

    def test_make_figures(self):

        v1 = Viewer(self.data, x='sample', overlay='axis', stack='data_vars')
        f1 = v1.make_figures()

        assert set(f1.index) == {'Var_1', 'Var_2'}

        v2 = Viewer(self.data, x='sample', overlay='data_vars', stack='axis')
        f2 = v2.make_figures()

        assert set(f2.index) == set(self.data.axis.values)

    def test_make_handlers(self):

        pass

    def test_add_glyphs(self):

        pass

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
