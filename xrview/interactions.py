""" ``xrview.interactions`` """

import numpy as np

from bokeh.layouts import row
from bokeh.models import MultiSelect


class BaseInteraction(object):
    """ Base class for interactions. """


class CoordValSelect(BaseInteraction):
    """ """

    def __init__(self, coord):

        self.coord = coord

        self.coord_vals = None
        self.context = None

    def attach(self, context):
        """ Attach element to context. """

        self.context = context

    def on_selected_coord_change(self, attr, old, new):
        """ Callback for multi-select change event. """

        self.coord_vals = new
        self.context._update_handlers([self.collect_hook])

    def collect_hook(self, data):
        """ """

        if self.coord_vals is not None:
            idx = np.zeros(data.sizes[self.context.x], dtype=bool)
            for c in self.coord_vals:
                idx = idx | (data[self.coord].values == c)
            return data.isel(**{self.context.x: idx})
        else:
            return data

    def layout_hook(self):
        """ """

        options = [
            (v, v) for v in np.unique(self.context.data[self.coord])]

        multi_select = MultiSelect(
            title=self.coord, value=[options[0][0]], options=options)
        multi_select.size = len(options)
        multi_select.on_change('value', self.on_selected_coord_change)

        self.coord_vals = [options[0][0]]
        self.context._update_handlers()

        self.context.layout = row(self.context.layout, multi_select)
