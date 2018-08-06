""" ``xrview.jupyterlab`` """

from sidecar import Sidecar


class SidecarViewerManager(object):

    def __init__(self, anchor='tab-after'):

        self.anchor = anchor
        self.viewers = {}

    def show(self, viewer, title):

        if title not in self.viewers:
            self.viewers[title] = viewer.copy()
            with Sidecar(title=title, anchor=self.anchor):
                self.viewers[title].show()
        else:
            self.viewers[title].update_inplace(viewer)
