import numpy as np
import xarray as xr
from bokeh.plotting import show

import xrview

x = np.linspace(0, 1, 100)
y = np.vstack([np.sqrt(x), x, x ** 2]).T
ds = xr.Dataset(
    {
        "Clean": (["x", "f"], y),
        "Noisy": (["x", "f"], y + 0.01 * np.random.randn(100, 3)),
    },
    coords={"x": x, "f": ["sqrt(x)", "x", "x^2"]},
)

plot = xrview.plot(ds, x="x", ncols=2)

da = xr.DataArray(np.ones(20), {"x": np.linspace(0, 1, 20)}, "x")
plot.add_figure(da)

# plot.show() doesn't work in sphinx
show(plot.make_layout())
