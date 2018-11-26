from unittest import TestCase

import numpy as np
import pandas as pd
from xrview.handlers import ResamplingDataHandler


class ResamplingDataHandlerTests(TestCase):

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
