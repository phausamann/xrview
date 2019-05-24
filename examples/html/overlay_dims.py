import numpy as np
import xarray as xr
from xrview.html import HtmlPlot

x = np.linspace(0, 1, 100)
y = np.vstack([np.sqrt(x), x, x ** 2]).T
ds = xr.Dataset({'Clean': (['x', 'f'], y),
                 'Noisy': (['x', 'f'], y + 0.01*np.random.randn(100, 3))},
                coords={'x': x, 'f': ['sqrt(x)', 'x', 'x^2']})

plot = HtmlPlot(ds, x='x', ncols=2)

# plot.show() doesn't work in sphinx
from bokeh.plotting import show
show(plot.make_layout())
