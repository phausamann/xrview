from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt

from xrview.timeseries.handlers import ResamplingDataHandler
from xrview.timeseries import TimeseriesViewer


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


class TimeseriesViewerTests(TestCase):

    def setUp(self):

        n_samples = 1000
        n_axes = 3

        coord_1 = ['a'] * (n_samples // 2) + ['b'] * (n_samples // 2)
        coord_2 = np.zeros(n_samples)

        data = xr.Dataset(
            {'Data': (['sample', 'axis'], np.random.rand(n_samples, n_axes))},
            coords={
                'sample': range(n_samples), 'axis': range(n_axes),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)},
        )

        self.viewer = TimeseriesViewer(data,
                                       sample_dim='sample',
                                       axis_dim='axis',
                                       select_coord='coord_1',
                                       figsize=(700, 500))

    def test_collect(self):

        # collect all
        data = self.viewer.collect()
        assert set(data.columns) == {'0_Data', '1_Data', '2_Data', 'selected'}
        npt.assert_allclose(data.iloc[:, :3], self.viewer.data.Data)

        # collect one coord
        data = self.viewer.collect(coord_vals=['a'])
        assert set(data.columns) == {'0_Data', '1_Data', '2_Data', 'selected'}
        npt.assert_allclose(
            data.iloc[:, :3],
            self.viewer.data.Data[self.viewer.data.coord_1 == 'a'])
