import numpy as np
import xarray as xr
from bokeh.plotting import show

import xrview
from xrview.glyphs import Square

x = np.linspace(0, 1, 100)
y = np.sqrt(x)

da_sqrt = xr.DataArray(y, {"x": x}, "x")
da_const = xr.DataArray(np.ones(20), {"x": x[::5]}, "x")

plot = xrview.plot(da_sqrt, x="x", glyphs="circle")
plot.add_overlay(da_const, glyphs=["line", Square(color="red")])

# plot.show() doesn't work in sphinx
show(plot.make_layout())
