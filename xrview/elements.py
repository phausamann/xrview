""" ``xrview.elements`` """

import numpy as np
import pandas as pd

from bokeh.models import Line as _Line

from xrview.utils import is_dataarray, is_dataset
from xrview.timeseries.handlers import ResamplingDataHandler


class BaseElement(object):
    """ Base class for elements. """

    def __init__(self, data):

        if is_dataarray(data):
            if data.name is None:
                self.data = data.to_dataset(name='Data')
            else:
                self.data = data.to_dataset()
        elif is_dataset(data):
            self.data = data
        else:
            raise ValueError('data must be DataArray or Dataset')

        self.x = None
        self.handler = None

    def _collect(self, data):
        """ Base method for collect. """

        plot_data = dict()

        for v in data.data_vars:
            if self.x not in data[v].dims:
                raise ValueError(self.x + ' is not a dimension of ' + v)
            elif len(data[v].dims) == 1:
                plot_data[v] = data[v].values
            elif len(data[v].dims) == 2:
                dim = [d for d in data[v].dims if d != self.x][0]
                for d in data[dim].values:
                    plot_data[v + '_' + str(d)] = data[v].sel(**{dim: d}).values
            else:
                raise ValueError(v + ' has too many dimensions')

        plot_data['selected'] = np.zeros(data.sizes[self.x], dtype=bool)

        return pd.DataFrame(plot_data, index=data[self.x])

    def attach(self, context):
        """ Attach element to context. """

        self.x = context.x
        self.handler = ResamplingDataHandler(
            self._collect(self.data), context.resolution * context.figsize[0],
            context=context, lowpass=context.lowpass)


class Line(BaseElement):
    """ A line glyph. """

    glyph = _Line
