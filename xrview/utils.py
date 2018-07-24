""" ``xrview.utils`` """

import json
import re
import ipykernel
import requests

from bokeh.models import \
    Model, LinearColorMapper, CategoricalColorMapper, LogColorMapper

from six.moves.urllib_parse import urljoin

try:  # Python 3
    from notebook.notebookapp import list_running_servers
except ImportError:  # Python 2
    import warnings
    from IPython.utils.shimmodule import ShimWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ShimWarning)
        from IPython.html.notebookapp import list_running_servers


def get_kernel_id():
    """ Get the kernel id of the current notebook.

    Returns
    -------
    kernel_id : str
        The id of the current kernel
    """

    return re.search(
        'kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)


def get_notebook_url():
    """ Get the url of the notebook server for the current notebook.

    Returns
    -------
    url : str
        The URL of the notebook server.
    """

    servers = list(list_running_servers())

    kernel_id = get_kernel_id()

    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        try:
            for nn in json.loads(response.text):
                if nn['kernel']['id'] == kernel_id:
                    return ss['url'][:-1]
        except:
            pass


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
        require_attrs = [
            'values',
            'coords',
            'dims',
            'to_dataset'
        ]

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
        require_attrs = [
            'data_vars',
            'coords',
            'dims',
            'to_array'
        ]

    return all([hasattr(X, name) for name in require_attrs])


def make_color_map(palette, n, field, mode='linear'):
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

    if mode == 'linear':
        mapper = LinearColorMapper(low=0, high=n, palette=palette)
    elif mode == 'log':
        mapper = LogColorMapper(low=0, high=n, palette=palette)
    elif mode == 'categorical':
        mapper = CategoricalColorMapper(
            factors=[str(i) for i in range(n)], palette=palette)
    else:
        raise ValueError('Unrecognized mode.')

    return {'field': field, 'transform': mapper}


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
