""" ``xrview.utils`` """
import json
import re
from urllib.parse import urljoin

import ipykernel
import requests
from notebook.notebookapp import list_running_servers


def get_kernel_id():
    """ Get the kernel id of the current notebook.

    Returns
    -------
    kernel_id : str
        The id of the current kernel
    """

    return re.search(
        "kernel-(.*).json", ipykernel.connect.get_connection_file()
    ).group(1)


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
        try:
            response = requests.get(
                urljoin(ss["url"], "api/sessions"),
                params={"token": ss.get("token", "")},
            )
            for nn in json.loads(response.text):
                if nn["kernel"]["id"] == kernel_id:
                    return ss["url"][:-1]
        except Exception:
            pass
