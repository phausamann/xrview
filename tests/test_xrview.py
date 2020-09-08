from unittest import TestCase

import numpy as np
import xarray as xr

import xrview
from xrview.html import HtmlPlot
from xrview.notebook import (
    NotebookPlot,
    NotebookTimeseriesViewer,
    NotebookViewer,
)


class TestPlot(TestCase):
    def setUp(self):
        """"""
        n_samples = 1000
        n_axes = 3

        coord_1 = ["a"] * (n_samples // 2) + ["b"] * (n_samples // 2)
        coord_2 = np.zeros(n_samples)
        coord_2[5::10] = 1

        self.data = xr.Dataset(
            {
                "Var_1": (
                    ["sample", "axis"],
                    np.random.rand(n_samples, n_axes),
                ),
                "Var_2": (
                    ["sample", "axis"],
                    np.random.rand(n_samples, n_axes),
                ),
            },
            coords={
                "sample": range(n_samples),
                "axis": range(n_axes),
                "coord_1": (["sample"], coord_1),
                "coord_2": (["sample"], coord_2),
            },
        )

    def test_options(self):
        """"""
        plot = xrview.plot(self.data, x="sample")
        assert isinstance(plot, HtmlPlot)

        # TODO implement
        with self.assertRaises(NotImplementedError):
            xrview.plot(self.data, x="sample", server=True)

        plot = xrview.plot(self.data, x="sample", output="notebook")
        assert isinstance(plot, NotebookPlot)

        plot = xrview.plot(
            self.data, x="sample", output="notebook", server=True
        )
        assert isinstance(plot, NotebookViewer)

        plot = xrview.plot(
            self.data, x="sample", output="notebook", server=True, resolution=1
        )
        assert isinstance(plot, NotebookTimeseriesViewer)

    def test_html_plot(self):
        """"""
        plot = xrview.plot(self.data, x="sample")
        plot.show()
