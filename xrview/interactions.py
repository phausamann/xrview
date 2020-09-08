""" ``xrview.interactions`` """
import numpy as np
from bokeh.models import MultiSelect


class BaseInteraction(object):
    """ Base class for interactions. """


class CoordValSelect(BaseInteraction):
    """ A list widget for selecting unique values of a certain coordinate. """

    def __init__(self, coord, max_elements=30, location="right"):
        """ Constructor.

        Parameters
        ----------
        coord: str
            The name of the coordinate.

        max_elements: int, default 30
            The maximum size of the displayed list of unique values.

        location: str, default 'right'
            The location of the widget in the layout.
        """
        self.coord = coord
        self.max_elements = max_elements
        self.location = location

        self.coord_vals = None
        self.context = None

    def attach(self, context):
        """ Attach element to context. """
        self.context = context

    def on_selected_coord_change(self, attr, old, new):
        """ Callback for multi-select change event. """
        self.coord_vals = new
        self.context._update_handlers()

    def collect_hook(self, data):
        """ Hook for context.collect. """
        if self.coord_vals is not None:
            idx = np.zeros(data.sizes[self.context.x], dtype=bool)
            for c in self.coord_vals:
                idx = idx | (data[self.coord].values == c)
            return data.isel(**{self.context.x: idx})
        else:
            return data

    def layout_hook(self):
        """ Hook for layout creation. """
        options = [(v, v) for v in np.unique(self.context.data[self.coord])]

        layout = MultiSelect(
            title=self.coord, value=[options[0][0]], options=options
        )
        layout.size = min(len(options), self.max_elements)
        layout.on_change("value", self.on_selected_coord_change)

        self.coord_vals = [options[0][0]]
        self.context._update_handlers()

        return layout
