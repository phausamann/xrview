from unittest import TestCase

import numpy as np
import xarray as xr

import numpy.testing as npt

from xrview.legacy import TimeseriesViewer


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
