""" ``xrview.timeseries.sources`` """

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from pandas.core.indexes.base import InvalidIndexError

from bokeh.models import ColumnDataSource

from tornado import gen


class ResamplingDataHandler(object):
    """

    Parameters
    ----------
    data : pandas DataFrame


    factor : numeric


    lowpass : bool, default False


    context : BaseViewer, optional


    with_range : bool, default True

    """

    def __init__(self, data, factor, lowpass=False, context=None,
                 with_range=True):

        self.data = data
        self.factor = factor
        self.lowpass = lowpass
        self.context = context

        if with_range:
            self.source_data = self.get_dict_from_range(
                self.data.index[0], self.data.index[-1])
            self.source = ColumnDataSource(self.source_data)
        else:
            self.source = ColumnDataSource(self.data)
            self.source.add(self.data.index, 'index')
            self.source_data = self.source.data

        self.selection = []

        self.callbacks = {
            'update_data': [],
            'reset_data': [],
            'update_source': []
        }

    @staticmethod
    def from_range(data, max_samples, start, end, lowpass):
        """ Get sub-sampled pandas DataFrame from index range.

        Parameters
        ----------
        data : pandas DataFrame
            The data to be sub-sampled

        max_samples : numeric
            The subsampling factor.

        start : numeric
            The start of the range to be sub-sampled.

        end : numeric
            The end of the range to be sub-sampled.

        Returns
        -------
        data_new : pandas DataFrame
            A sub-sampled slice of the data.
        """

        if start is None:
            start = 0
        else:
            try:
                start = data.index.get_loc(start, method='nearest')
            except InvalidIndexError:
                # handle non-ordered/non-unique index
                start = np.argmin(np.abs(data.index - start))

        if end is None:
            end = data.shape[0]
        else:
            try:
                end = data.index.get_loc(end, method='nearest') + 1
            except InvalidIndexError:
                # handle non-ordered/non-unique index
                end = np.argmin(np.abs(data.index - end)) + 1

        step = int(np.ceil((end-start) / max_samples))

        # TODO: handle NaNs at start/end
        if step == 0:
            # hacky solution for range reset
            data_new = pd.concat((data.iloc[:1], data.iloc[-1:]))
        else:
            data_new = data.iloc[start:end]
            if step > 1 and lowpass:
                for c in data_new.columns:
                    if c != 'selected':
                        data_new[c] = filtfilt(
                            *butter(3, 1/step), data_new.loc[:, c])
            data_new = data_new.iloc[::step]
            # hacky solution for range reset
            if start > 0:
                data_new = pd.concat((data.iloc[:1], data_new))
            if end < data.shape[0]-1:
                data_new = data_new.append(data.iloc[-1])

        return data_new

    def get_range(self, start=None, end=None):
        """ Get the range of valid indexes for the data to be displayed.

        Parameters
        ----------
        start : numeric
            The start of the range to be displayed.

        end : numeric
            The end of the range to be displayed.

        Returns
        -------
        start : numeric
            The adjusted start.

        end : numeric
            The adjusted end.
        """

        first_source_idx = self.source.data['index'][0]
        last_source_idx = self.source.data['index'][-1]

        # convert to timestamp if necessary
        if isinstance(self.data.index, pd.DatetimeIndex):
            start = pd.to_datetime(start, unit='ms')
            end = pd.to_datetime(end, unit='ms')
            first_source_idx = pd.to_datetime(first_source_idx, unit='ms')
            last_source_idx = pd.to_datetime(last_source_idx, unit='ms')

            # get new start and end
        if start is not None:
            if start < first_source_idx:
                start = max(self.data.index[0], start)
            elif start > last_source_idx:
                start = min(self.data.index[-1], start)
            elif start < self.data.index[0]:
                start = self.data.index[0]
            elif start > self.data.index[-1]:
                start = self.data.index[-1]
        elif len(self.source.data['index']) > 0:
            start = first_source_idx
        else:
            start = self.data.index[0]

        if end is not None:
            if end < first_source_idx:
                end = max(self.data.index[0], end)
            elif end > last_source_idx:
                end = min(self.data.index[-1], end)
            elif end < self.data.index[0]:
                end = self.data.index[0]
            elif end > self.data.index[-1]:
                end = self.data.index[-1]
        elif len(self.source.data['index']) > 0:
            end = last_source_idx
        else:
            end = self.data.index[-1]

        return start, end

    def get_dict_from_range(self, start=None, end=None):
        """ Get sub-sampled source data from index range as a dict.

        Parameters
        ----------
        start : numeric
            The start of the range to be displayed.

        end : numeric
            The end of the range to be displayed.

        Returns
        -------
        new_source_data : dict
            The sub-sampled slice of the data to be displayed.
        """

        df = self.from_range(self.data, self.factor, start, end, self.lowpass)
        new_source_data = df.to_dict(orient='list')
        new_source_data['index'] = df.index

        for k in list(new_source_data):
            if isinstance(k, tuple):
                new_source_data['_'.join(k)] = new_source_data.pop(k)

        return new_source_data

    def update_data(self, start=None, end=None):
        """ Update data and selection to be displayed. """

        if self.context is not None and self.context.verbose:
            print('Updating data')

        start, end = self.get_range(start, end)

        self.source_data = self.get_dict_from_range(start, end)

        # update source selection
        if self.source.selected is not None \
                and np.sum(self.data.selected) > 0:
            self.selection = list(
                np.where(self.source_data['selected'])[0])
        else:
            self.selection = []

        # call attached callbacks
        for c in self.callbacks['update_data']:
            c()

    def reset_data(self):
        """ Reset data and selection to be displayed. """

        self.source_data = self.get_dict_from_range(None, None)

        self.data.selected = np.zeros(self.data.shape[0], dtype=bool)
        self.selection = []

        # call attached callbacks
        for c in self.callbacks['reset_data']:
            c()

    @gen.coroutine
    def update_source(self):
        """ Update data and selected.indices of self.source """

        if self.context is not None and self.context.verbose:
            print('Updating source')

        self.source.data = self.source_data

        if self.source.selected is not None:
            self.source.selected.indices = self.selection

        # call attached callbacks
        for c in self.callbacks['update_source']:
            c()

        # remove update lock
        # TODO: check if this can be a callback (and if it needs to be last)
        if self.context is not None:
            self.context.pending_xrange_update = False

    def add_callback(self, method, callback):
        """ Add a callback to one of this instance's methods.

        Parameters
        ----------
        method : str
            The name of the method this callback will be attached to.

        callback : callable
            The callback function.
        """

        if method not in self.callbacks:
            raise ValueError('Unrecognized method name: ' + str(method))

        if callback in self.callbacks[method]:
            raise ValueError(
                str(callback) + ' has already been attached to this instance.')

        self.callbacks[method].append(callback)
