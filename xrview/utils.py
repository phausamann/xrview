""" ``xrview.utils`` """

import json
import re
import ipykernel
import requests
import functools

try:  # Python 3
    from urllib.parse import urljoin
except ImportError:  # Python 2
    from urlparse import urljoin

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
