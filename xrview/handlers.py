""" ``xrview.handlers`` """
from __future__ import division

import numpy as np
import pandas as pd

from pandas.core.indexes.base import InvalidIndexError

from bokeh.models import ColumnDataSource
from bokeh.document import without_document_lock

from tornado import gen


class DataHandler(object):
    """

    Parameters
    ----------
    data : pandas DataFrame

    """

    def __init__(self, data):

        self.source = ColumnDataSource(data)
        self.source.add(data.index, 'index')


class InteractiveDataHandler(DataHandler):

    def __init__(self, data, context=None, verbose=False):

        super(InteractiveDataHandler, self).__init__(data)

        self.data = data
        self.source_data = self.source.data

        self.context = context
        self.verbose = verbose

        self.selection_bounds = None
        self.selection = []

        self.pending_update = False
        self.update_buffer = None

        self.callbacks = {
            'update_data': [],
            'reset_data': [],
            'update_source': []
        }

    def get_dict(self):
        """ Get data as a dict. """
        new_source_data = self.data.to_dict(orient='list')
        new_source_data['index'] = self.data.index
        for k in list(new_source_data):
            if isinstance(k, tuple):
                new_source_data['_'.join(k)] = new_source_data.pop(k)

        return new_source_data

    @without_document_lock
    @gen.coroutine
    def update(self, **kwargs):
        """ Update callback for handler. """
        self.pending_update = True
        self.update_data(**kwargs)
        self.update_selection()
        if self.context is not None and self.context.doc is not None:
            self.context.doc.add_next_tick_callback(self.update_source)

    @without_document_lock
    @gen.coroutine
    def reset(self):
        """ Reset data and selection to be displayed. """
        self.selection_bounds = None
        self.selection = []
        for c in self.callbacks['reset_data']:
            c()
        if self.context is not None and self.context.doc is not None:
            self.context.doc.add_next_tick_callback(self.update_source)

    @without_document_lock
    @gen.coroutine
    def update_data(self, **kwargs):
        """ Update data and selection to be displayed. """
        self.source_data = self.get_dict()
        for c in self.callbacks['update_data']:
            c()

    @without_document_lock
    @gen.coroutine
    def update_selection(self):
        """ Update selection. """
        if self.source.selected is not None \
                and self.selection_bounds is not None:
            self.selection = list(np.where(
                (self.source_data['index'] >= self.selection_bounds[0])
                & (self.source_data['index'] <= self.selection_bounds[1]))[0])
        else:
            self.selection = []

    @gen.coroutine
    def update_source(self):
        """ Update data and selected.indices of self.source """
        if self.verbose:
            print('Updating source')
        self.source.data = self.source_data
        if self.source.selected is not None:
            self.source.selected.indices = self.selection
        for c in self.callbacks['update_source']:
            c()
        self.pending_update = False
        if self.update_buffer is not None:
            self.context.doc.add_next_tick_callback(self.update_buffer)
            self.update_buffer = None

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


class ResamplingDataHandler(InteractiveDataHandler):
    """

    Parameters
    ----------
    data : pandas DataFrame


    factor : numeric


    lowpass : bool, default False


    context : TimeseriesViewer, optional


    with_range : bool, default True

    """

    def __init__(self, data, factor, lowpass=False, context=None,
                 with_range=True, verbose=False):

        self.data = data

        self.factor = factor
        self.lowpass = lowpass
        self.context = context
        self.verbose = verbose

        if with_range:
            self.source_data = self.get_dict_from_range(
                self.data.index[0], self.data.index[-1])
            self.source = ColumnDataSource(self.source_data)
        else:
            self.source = ColumnDataSource(self.data)
            self.source.add(self.data.index, 'index')
            self.source_data = self.source.data

        self.selection_bounds = None
        self.selection = []

        self.pending_update = False
        self.update_buffer = None

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

        # handle the case of no data
        if data.shape[0] == 0:
            return data

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
                # TODO make this work
                from scipy.signal import butter, filtfilt
                for c in data_new.columns:
                    if c != 'selected':
                        coefs = butter(3, 1/step)
                        data_new[c] = filtfilt(
                            coefs[0], coefs[1], data_new.loc[:, c])
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

        # handle the case of no data
        if self.data.shape[0] == 0 or self.source.data['index'].shape[0] == 0:
            return None, None

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

    @without_document_lock
    @gen.coroutine
    def update_data(self, start=None, end=None):
        """ Update data and selection to be displayed. """
        if self.verbose:
            print('Updating data')
        start, end = self.get_range(start, end)
        self.source_data = self.get_dict_from_range(start, end)
        for c in self.callbacks['update_data']:
            c()

    @without_document_lock
    @gen.coroutine
    def reset(self):
        """ Reset data and selection to be displayed. """
        self.source_data = self.get_dict_from_range(None, None)
        self.selection_bounds = None
        self.selection = []
        for c in self.callbacks['reset_data']:
            c()
        if self.context is not None:
            self.context.doc.add_next_tick_callback(self.update_source)
