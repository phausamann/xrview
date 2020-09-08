import os
import shutil
from unittest import TestCase

import numpy as np
import pytest
import xarray as xr
from bokeh.models import FactorRange, Span
from numpy import testing as npt

from xrview.core import BasePlot, BaseViewer
from xrview.glyphs import Circle, Line, VBar, VLine
from xrview.interactions import CoordValSelect
from xrview.timeseries import TimeseriesViewer

test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")


class BasePlotTests(TestCase):

    cls = BasePlot

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

    def test_constructor(self):
        """"""
        with self.assertRaises(ValueError):
            self.cls(self.data, x="not_a_dimension")

        with self.assertRaises(ValueError):
            self.cls(self.data.Var_1.values, x="sample")

        with self.assertRaises(ValueError):
            self.cls(self.data, x="sample", overlay="nothing")

    def test_collect(self):
        """"""
        v = self.cls(self.data, x="sample")
        data = v._collect()
        npt.assert_allclose(data.iloc[:, :3], v.data.Var_1)
        self.assertEqual(
            set(data.columns),
            {"Var_1_0", "Var_1_1", "Var_1_2", "Var_2_0", "Var_2_1", "Var_2_2"},
        )

    def test_make_handlers(self):
        """"""
        v = self.cls(self.data, x="sample")
        v._make_handlers()
        assert len(v.handlers) == 1
        assert isinstance(v.handlers[0], self.cls.handler_type)

    def test_make_maps(self):
        """"""
        v1 = self.cls(self.data, x="sample")
        v1._make_handlers()
        v1._make_maps()

        assert all(v1.figure_map.index == range(2))
        assert all(v1.glyph_map.index == range(6))
        assert all(v1.glyph_map.figure[v1.glyph_map["var"] == "Var_1"] == 0)
        assert all(v1.glyph_map.figure[v1.glyph_map["var"] == "Var_2"] == 1)
        self.assertEqual(
            {a["y"] for a in v1.glyph_map.glyph_kwargs},
            {"Var_1_0", "Var_1_1", "Var_1_2", "Var_2_0", "Var_2_1", "Var_2_2"},
        )

        v2 = self.cls(self.data, x="sample", overlay="data_vars")
        v2._make_handlers()
        v2._make_maps()

        assert all(v2.figure_map.index == range(3))
        assert all(v2.glyph_map.index == range(6))
        assert all(v2.glyph_map.figure == v2.glyph_map.dim_val)
        self.assertEqual(
            {a["y"] for a in v2.glyph_map.glyph_kwargs},
            {"Var_1_0", "Var_1_1", "Var_1_2", "Var_2_0", "Var_2_1", "Var_2_2"},
        )

    def test_make_figures(self):
        """"""
        v = self.cls(self.data, x="sample")
        v._make_handlers()
        v._make_maps()
        v._make_figures()

    def test_add_glyphs(self):
        """"""
        v = self.cls(self.data, x="sample")
        v._make_handlers()
        v._make_maps()
        v._make_figures()
        v._add_glyphs()

    def test_add_tooltips(self):
        """"""
        v = self.cls(self.data, x="sample", tooltips={"sample": "@index"})
        v._make_handlers()
        v._make_maps()
        v._make_figures()
        v._add_tooltips()

    def test_make_layout(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.make_layout()

    def test_coords(self):
        """"""
        v = self.cls(self.data, x="sample", coords=["coord_1"])
        v.make_layout()

    def test_custom_glyphs(self):
        """"""
        v = self.cls(self.data, x="sample", glyphs=[Circle(), VBar(0.001)])
        v.make_layout()

    def test_multiindex(self):
        """"""
        v = self.cls(self.data.stack(multi=("sample", "axis")), x="multi")
        v.make_layout()
        assert isinstance(v.figures[0].x_range, FactorRange)
        assert "sample" in v.handlers[0].source.column_names
        assert "axis" in v.handlers[0].source.column_names

    def test_add_figure(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.add_figure(self.data.Var_1, "line", name="Test")
        v.make_layout()

    def test_add_overlay(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.add_overlay(self.data.coord_2, "line", onto="Var_1")
        v.add_overlay(self.data.coord_2, Line(), name="Test", onto=1)
        v.add_overlay(self.data.coord_2[self.data.coord_2 > 0], VLine())
        v.make_layout()

    def test_add_annotation(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.add_annotation(Span(location=500, dimension="height"))
        v.make_layout()

    def test_modify_figures(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.add_figure(self.data.Var_1, Line(), name="Test")
        v.modify_figures({"xaxis.axis_label": "test_label"}, 0)
        v.make_layout()
        self.assertEqual(v.figures[0].xaxis[0].axis_label, "test_label")

    @pytest.mark.xfail(raises=RuntimeError)
    def test_export(self):
        """"""
        shutil.rmtree(os.path.join(test_data_dir, "out"), ignore_errors=True)
        os.makedirs(os.path.join(test_data_dir, "out"))

        v = self.cls(self.data, x="sample")
        v.export(os.path.join(test_data_dir, "out", "export_test.png"))
        v.export(os.path.join(test_data_dir, "out", "export_test.svg"))

        assert os.path.isfile(
            os.path.join(test_data_dir, "out", "export_test.png")
        )
        assert os.path.isfile(
            os.path.join(test_data_dir, "out", "export_test.svg")
        )
        assert os.path.isfile(
            os.path.join(test_data_dir, "out", "export_test_1.svg")
        )

        with self.assertRaises(ValueError):
            v.export("export_test.noext")
        with self.assertRaises(ValueError):
            v.export("export_test.noext", mode="nomode")


class BaseViewerTests(BasePlotTests):

    cls = BaseViewer

    def test_make_doc(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.make_layout()
        v.make_doc()

    def test_add_interaction(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.add_interaction(CoordValSelect("coord_1"))
        v.make_layout()

    def test_update_handlers(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.make_layout()
        v.make_doc()
        v.update_handlers()

    def test_on_selected_points_change(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.make_layout()
        v.make_doc()
        v.handlers[0].source.selected.indices = [5, 10, 15, 20]
        idx = v.handlers[0].source.selected.indices
        # TODO: execute next_tick_callback
        v.on_selected_points_change("indices", None, idx)

    def test_reset(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.make_layout()
        v.make_doc()
        # TODO: execute next_tick_callback
        v.on_reset(None)


class TimeseriesViewerTests(BaseViewerTests):

    cls = TimeseriesViewer

    def test_on_reset(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.make_layout()
        v.make_doc()
        v.on_reset(None)

    def test_on_xrange_change(self):
        """"""
        v = self.cls(self.data, x="sample")
        v.add_figure(self.data.Var_1, Line(), name="Test")
        v.make_layout()
        v.make_doc()
        v.on_xrange_change("start", None, 3.5)

    def test_multiindex(self):
        """"""
        # TODO: assertRaises...
        pass
