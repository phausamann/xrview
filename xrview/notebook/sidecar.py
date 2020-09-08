try:
    from sidecar import Sidecar

except ImportError:
    raise ValueError(
        "sidecar must be installed in order to use the "
        "SidecarViewerManager class."
    )


class SidecarViewerManager(object):
    def __init__(self, anchor="tab-after"):
        self.anchor = anchor
        self.sidecars = {}

    def show(self, viewer, title):
        if title not in self.sidecars:
            self.sidecars[title] = Sidecar(title=title, anchor=self.anchor)
        else:
            # TODO: find a better way to do this
            self.sidecars[title].close()
            self.sidecars[title].open()

        with self.sidecars[title]:
            viewer.show()
