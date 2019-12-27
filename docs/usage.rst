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

.. doctest::

    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrview


Minimal example
~~~~~~~~~~~~~~~

The following code will open a browser tab with the figure shown below.

.. bokeh-plot:: ../examples/html/minimal_example.py
    :source-position: none

.. doctest::

    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sqrt(x)
    >>> da = xr.DataArray(y, coords={'x': x}, dims='x')
    >>> plot = xrview.plot(da, x='x')
    >>> plot.show() # doctest:+SKIP


Overlaying and tiling plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When passing a ``Dataset``, each variable will be plotted in a separate figure.
If the variable has a second dimension, each element along this dimension will
be plotted as a separate line and a legend will be automatically created
based on the coordinates of this dimension.

.. bokeh-plot:: ../examples/html/overlay_dims.py
    :source-position: none

.. doctest::

    >>> x = np.linspace(0, 1, 100)
    >>> y = np.vstack([np.sqrt(x), x, x ** 2]).T
    >>> ds = xr.Dataset({'Clean': (['x', 'f'], y),
    ...                  'Noisy': (['x', 'f'], y + 0.01*np.random.randn(100, 3))},
    ...                 coords={'x': x, 'f': ['sqrt(x)', 'x', 'x^2']})
    >>> plot = xrview.plot(ds, x='x', ncols=2)
    >>> plot.show() # doctest:+SKIP

Alternatively, you can show the elements of the dimension in separate
figures and overlay the variables by specifying ``overlay='data_vars'``:

.. bokeh-plot:: ../examples/html/overlay_data_vars.py
    :source-position: none

.. doctest::

    >>> plot = xrview.plot(ds, x='x', ncols=2, overlay='data_vars')
    >>> plot.show() # doctest:+SKIP

You can add additional figures to the plot with the :py:func:`add_figure`
method by providing data as a DataArray. The DataArray has to contain a
coordinate with the same name as the x coordinate, but they do not need to
have the same values.

.. bokeh-plot:: ../examples/html/add_figure.py
    :source-position: none

.. doctest::

    >>> da = xr.DataArray(np.ones(20), {'x': np.linspace(0, 1, 20)}, 'x', name='Const')
    >>> plot = xrview.plot(ds, x='x', ncols=2)
    >>> plot.add_figure(da)
    >>> plot.show() # doctest:+SKIP

Data can also be overlaid onto existing figures with the
:py:func:`add_overlay` method. With the ``onto`` parameter, you can select a
figure by index or title on which to overlay the data. By default, the data
will be overlaid onto all figures.

.. bokeh-plot:: ../examples/html/add_overlay.py
    :source-position: none

.. doctest::

    >>> plot = xrview.plot(ds, x='x', ncols=2)
    >>> plot.add_overlay(da, onto='Clean')
    >>> plot.show() # doctest:+SKIP
