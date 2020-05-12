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
    >>> da = xr.DataArray(y, {'x': x}, 'x')
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
    ...                 {'x': x, 'f': ['sqrt(x)', 'x', 'x^2']})
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

    >>> da = xr.DataArray(np.ones(20), {'x': np.linspace(0, 1, 20)}, 'x')
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


Customizing glyphs and annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xrview supports plotting with many of the standard bokeh glyphs as well as
some custom composite glyphs such as error bars and boxes for box plots.
Glyphs can be passed to :py:func:`plot`, :py:func:`add_overlay` and
:py:func:`add_figure` as strings, instances of a glyph class or iterables of
any combination of both. When you pass a glyph instance, you can specify
additional keyword arguments.

In this example, one array is plotted with circles and a second with a
blue line and red squares:

.. bokeh-plot:: ../examples/html/basic_glyphs.py
    :source-position: none

.. doctest::

    >>> from xrview.glyphs import Square
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sqrt(x)
    >>> da_sqrt = xr.DataArray(y, {'x': x}, 'x')
    >>> da_const = xr.DataArray(np.ones(20), {'x': x[::5]}, 'x')
    >>> plot = xrview.plot(da_sqrt, x='x', glyphs='circle')
    >>> plot.add_overlay(da_const, glyphs=['line', Square(color='red')])
    >>> plot.show() # doctest:+SKIP

xrview also provides a straightforward way of

.. note::

    See :ref:`API/Glyphs` for a list of available glyphs.


Categorical and time series data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Fine-tuning bokeh figures
=========================


Interactive plotting
====================

.. note::

    Interactive plotting is so far only supported in jupyter notebooks.

Adding interactions
~~~~~~~~~~~~~~~~~~~


Sub-sampled timeseries plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

