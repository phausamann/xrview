=====
Usage
=====

.. py:currentmodule:: xrview.html

xrview provides several utilities for automatically creating
interactive bokeh plots from xarray data types.

Basic plotting with HTML output
===============================

The basic tool for plotting is the :py:class:`HtmlPlot` class in the
:py:mod:`xrview.html` module. It is initialized with an ``xarray.DataArray``
or ``Dataset`` and the name of the dimension that will represent the x
coordinate in the plot.

All examples in this section assume the following imports:

.. code-block:: python

    import numpy as np
    import xarray as xr
    from xrview.html import HtmlPlot


Minimal example
~~~~~~~~~~~~~~~

The following code will open a browser tab with the figure shown below.

.. bokeh-plot:: ../examples/html/minimal_example.py
    :source-position: none

.. code-block:: python

    x = np.linspace(0, 1, 100)
    y = np.sqrt(x)
    da = xr.DataArray(y, coords={'x': x}, dims='x')

    plot = HtmlPlot(da, 'x')
    plot.show()


Overlaying and tiling plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the array has a second dimension, each element along this dimension will
be plotted as a separate line and a legend will be automatically created
based on the coordinates of this dimension.

.. bokeh-plot:: ../examples/html/dim_overlay.py
    :source-position: none

.. code-block:: python

    x = np.linspace(0, 1, 100)
    y = np.vstack([np.sqrt(x), x, x**2]).T
    da = xr.DataArray(y, dims=(['x', 'f']),
                      coords={'x': x, 'f': ['sqrt(x)', 'x', 'x^2']})

    plot = HtmlPlot(da, 'x')
    plot.show()

