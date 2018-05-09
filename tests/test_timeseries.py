from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt

from xrview.timeseries.base import BaseViewer
from xrview.timeseries import TimeseriesViewer, FeatureMapViewer


class BaseViewerTests(TestCase):

    def setUp(self):

        self.n_samples = 1000

        data = pd.DataFrame({'y': np.random.random(self.n_samples)})

        self.viewer = BaseViewer(data)

    def test_get_range(self):

        # integer index
        idx = range(self.n_samples)
        self.viewer.plot_data.index = idx
        self.viewer._init_source()

        assert self.viewer.get_range() == (idx[0], idx[-1])
        assert self.viewer.get_range(-1, self.n_samples) == (idx[0], idx[-1])
        assert self.viewer.get_range(2, 7) == (2, 7)

        # float index
        idx = np.linspace(0, 1, self.n_samples)
        self.viewer.plot_data.index = idx
        self.viewer._init_source()

        assert self.viewer.get_range() == (idx[0], idx[-1])
        assert self.viewer.get_range(-1, 2) == (idx[0], idx[-1])
        assert self.viewer.get_range(0.2, 0.7) == (0.2, 0.7)

        # datetime index
        idx = pd.date_range('2000-01-01', periods=self.n_samples, freq='H')
        t_outer = pd.to_datetime('2001-01-01')
        t_inner = pd.to_datetime('2000-01-02')
        self.viewer.plot_data.index = idx
        self.viewer._init_source()

        assert self.viewer.get_range() == (idx[0], idx[-1])
        assert self.viewer.get_range(0, t_outer.value/1e6) == (idx[0], idx[-1])
        assert self.viewer.get_range(t_inner.value/1e6, t_inner.value/1e6) == \
            (t_inner, t_inner)


class TimeseriesViewerTests(TestCase):

    def setUp(self):

        n_samples = 1000
        n_axes = 3

        coord_1 = ['a']*(n_samples//2) + ['b']*(n_samples//2)
        coord_2 = np.zeros(n_samples)

        data = xr.DataArray(
            np.random.random((n_samples, n_axes)),
            coords={
                'sample': range(n_samples), 'axis': range(n_axes),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)},
            dims=['sample', 'axis']
        )

        self.viewer = TimeseriesViewer(data,
                                       sample_dim='sample',
                                       axis_dim='axis',
                                       select_coord='coord_1',
                                       vlines_coord='coord_2',
                                       figsize=(700, 500))

    def test_collect(self):

        # collect all
        self.viewer.collect()
        npt.assert_allclose(self.viewer.plot_data.columns,
                            self.viewer.data['axis'].values)
        npt.assert_allclose(self.viewer.plot_data,
                            self.viewer.data)

        # collect one coord
        self.viewer.collect(coord_vals=['a'])
        npt.assert_allclose(self.viewer.plot_data.columns,
                            self.viewer.data['axis'].values)
        npt.assert_allclose(self.viewer.plot_data,
                            self.viewer.data[self.viewer.data.coord_1 == 'a'])
