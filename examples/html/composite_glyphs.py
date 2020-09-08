import numpy as np
import xarray as xr
from bokeh.plotting import show

import xrview
from xrview.glyphs import BoxWhisker

x = np.arange(100) / 100
y = np.vstack([np.sqrt(x), x, x ** 2]).T
c = np.sin(10 * np.pi * x)

da = xr.DataArray(
    y,
    dims=(["x", "f"]),
    coords={
        "x": x,
        "f": ["sqrt", "lin", "square"],
        "sin": ("x", c),
        "lower": (["x", "f"], y - 0.5),
        "upper": (["x", "f"], y + 0.5),
        "q_lower": (["x", "f"], y - 0.25),
        "q_upper": (["x", "f"], y + 0.25),
    },
)

da = da[::10].stack(z=("x", "f"))

plot = xrview.plot(
    da,
    x="z",
    coords=True,
    y_range=(-0.5, 1.5),
    glyphs=BoxWhisker(0.8, "q_lower", "lower", "q_upper", "upper", color="f"),
)

plot.modify_figures(
    {"xgrid.visible": False, "xaxis.major_label_orientation": np.pi / 2}
)

# plot.show() doesn't work in sphinx
show(plot.make_layout())
