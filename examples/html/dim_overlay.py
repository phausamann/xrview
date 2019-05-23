import numpy as np
import xarray as xr
from xrview.html import HtmlPlot

x = np.linspace(0, 1, 100)
y = np.vstack([np.sqrt(x), x, x ** 2]).T
da = xr.DataArray(y, dims=(['x', 'f']),
                  coords={'x': x, 'f': ['sqrt(x)', 'x', 'x^2']})

plot = HtmlPlot(da, 'x')

# plot.show() doesn't work in sphinx
from bokeh.plotting import show
show(plot.make_layout())
