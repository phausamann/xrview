import numpy as np
import xarray as xr
from xrview.html import HtmlPlot

x = np.linspace(0, 1, 100)
y = np.sqrt(x)
da = xr.DataArray(y, coords={'x': x}, dims='x')

plot = HtmlPlot(da, x='x')

# plot.show() doesn't work in sphinx
from bokeh.plotting import show
show(plot.make_layout())
