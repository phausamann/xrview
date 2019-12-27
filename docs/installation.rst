.. highlight:: shell

============
Installation
============


Latest release
--------------

xrview is currently unreleased, but can be installed from the GitHub
repository via ``pip``:

.. code-block:: console

    $ pip install git+https://github.com/phausamann/xrview.git

This will also install the minimal dependencies for creating HTML plots.

Optional dependencies
---------------------

For plotting in a jupyter notebook, you will also need the ``notebook`` and
``requests`` packages:

.. code-block:: console

    $ pip install notebook requests

Bokeh server support depends on a specific version of ``tornado``:

.. code-block:: console

    $ pip install tornado<=4.5.3
