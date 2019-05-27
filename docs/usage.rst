=====
Usage
=====

.. highlight:: python

.. currentmodule:: xrview

xrview provides several utilities for automatically creating
interactive bokeh plots from xarray data types.

Basic plotting with HTML output
===============================

``xrview.plot`` will create a plot of an ``xarray.DataArray`` or
``Dataset`` given the name of the dimension that represents the x
coordinate in the plot.

All examples in this section assume the following imports:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import xrview


Minimal example
~~~~~~~~~~~~~~~

The following code will open a browser tab with the figure shown below.

.. bokeh-plot:: ../examples/html/minimal_example.py
    :source-position: none

.. code-block:: python

    x = np.linspace(0, 1, 100)
    y = np.sqrt(x)
    da = xr.DataArray(y, coords={'x': x}, dims='x')

    plot = xrview.plot(da, x='x')
    plot.show()


Overlaying and tiling plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When passing a ``Dataset``, each variable will be plotted in a separate figure.
If the variable has a second dimension, each element along this dimension will
be plotted as a separate line and a legend will be automatically created
based on the coordinates of this dimension.

.. bokeh-plot:: ../examples/html/overlay_dims.py
    :source-position: none

.. code-block:: python

    x = np.linspace(0, 1, 100)
    y = np.vstack([np.sqrt(x), x, x ** 2]).T
    ds = xr.Dataset({'Clean': (['x', 'f'], y),
                     'Noisy': (['x', 'f'], y + 0.01*np.random.randn(100, 3))},
                    coords={'x': x, 'f': ['sqrt(x)', 'x', 'x^2']})

    plot = xrview.plot(ds, x='x', ncols=2)
    plot.show()

Alternatively, you can show the elements of the dimension in separate
figures and overlay the variables by specifying ``overlay='data_vars'``:

.. bokeh-plot:: ../examples/html/overlay_data_vars.py
    :source-position: none

.. code-block:: python

    plot = xrview.plot(ds, x='x', ncols=2, overlay='data_vars')
    plot.show()

