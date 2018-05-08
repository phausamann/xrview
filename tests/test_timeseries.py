from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt

from xrview.timeseries import TimeseriesViewer, FeatureMapViewer


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
                                       vlines_coord='coord_1',
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
