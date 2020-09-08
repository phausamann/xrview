import functools

from bokeh.model import Model
from bokeh.models import (
    CategoricalColorMapper,
    LinearColorMapper,
    LogColorMapper,
)

try:
    from types import MappingProxyType
except ImportError:
    from collections import Mapping

    class MappingProxyType(Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def __str__(self):
            return self._data.__str__()

        def __repr__(self):
            return self._data.__repr__()


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def is_dataarray(X, require_attrs=None):
    """ Check whether an object is a DataArray.

    Parameters
    ----------
    X : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a DataArray.

    Returns
    -------
    bool
        Whether the object is a DataArray or not.
    """

    if require_attrs is None:
        require_attrs = ["values", "coords", "dims", "to_dataset"]

    return all([hasattr(X, name) for name in require_attrs])


def is_dataset(X, require_attrs=None):
    """ Check whether an object is a Dataset.

    Parameters
    ----------
    X : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a Dataset.

    Returns
    -------
    bool
        Whether the object is a Dataset or not.
    """

    if require_attrs is None:
        require_attrs = ["data_vars", "coords", "dims", "to_array"]

    return all([hasattr(X, name) for name in require_attrs])


def make_color_map(palette, n, field, mode="linear"):
    """

    Parameters
    ----------
    palette : bokeh palette

    n : int

    field : str

    mode : 'linear', 'log' or 'categorical', default 'linear'

    Returns
    -------
    cmap : dict

    """

    if callable(palette):
        palette = palette(n)
    else:
        palette = palette[n]

    if mode == "linear":
        mapper = LinearColorMapper(low=0, high=n, palette=palette)
    elif mode == "log":
        mapper = LogColorMapper(low=0, high=n, palette=palette)
    elif mode == "categorical":
        mapper = CategoricalColorMapper(
            factors=[str(i) for i in range(n)], palette=palette
        )
    else:
        raise ValueError("Unrecognized mode.")

    return {"field": field, "transform": mapper}


def clone_models(d):
    """ Clone bokeh models in a dict.

    Parameters
    ----------
    d : dict
        The input dict.

    Returns
    -------
    d : dict
        A copy of the input dict with all bokeh models replaced by a clone of
        themselves.
    """

    d = d.copy()

    for k, v in d.items():
        if isinstance(v, Model):
            d[k] = v._clone()
        elif isinstance(v, dict):
            d[k] = clone_models(v)

    return d
