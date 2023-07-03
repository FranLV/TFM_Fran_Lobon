import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
from .timers import Timer
import h5py


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        
        data = h5py.File(path_to_event_file, "r")    

        self.dataset = dict()
        for columns in ['p', 'x', 'y', 't']:
            self.dataset[columns] = data['events']['table']['{}'.format(columns)]

        data.close()

        self.min_ts = self.dataset["t"].min()
        self.max_ts = self.dataset["t"].max()
        # self.data_duration = self.max_ts - self.min_ts
        self.t_start_idx = 0

        self._OFFSET = self.min_ts

        for offset, time in enumerate(self.dataset["t"]):
            if time >= self._OFFSET:
                self.t_start_idx = offset
                break

        self.t_end_idx = self.t_start_idx

        self.last_stamp = self.dataset["t"][self.t_start_idx]
        self.duration_s = duration_ms / 1000.0

        # Cambiar el número para aumentar los FPS
        self.timestamps_images = np.arange(self._OFFSET, self.max_ts, duration_ms * 1000)
        # Quitamos el último frame 
        self.image_indices = np.arange(len(self.timestamps_images)-1)

    def __len__(self):
        return len(self.left_event['x'])
    
    def get_max_image_indices(self) -> int:
        return len(self.image_indices)
    
    def get_start_index(self) -> int:
        return self.t_start_idx
    
    def get_events(self, idx: int) -> np.ndarray:

        t_end_us = self.timestamps_images[idx+1]

        self.t_start_idx = self.t_end_idx

        for offset, time in enumerate(self.dataset["t"][self.t_start_idx:]):
            if time >= t_end_us:
                self.t_end_idx = self.t_start_idx + offset - 1
                break

        return self.load_event(self.t_start_idx, self.t_end_idx)


    def load_event(self, start_index: int, end_index: int) -> np.ndarray:
        """Load events.
        The original hdf5 file contains (x, y, t, p),
        where x means in width direction, and y means in height direction. p is -1 or 1.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value, in sec. p is [-1, 1].
        """
        n_events = end_index - start_index
        events = np.zeros((n_events, 4), dtype=np.float64)


        events[:, 0] = self.dataset["t"][start_index:end_index]
        events[:, 1] = self.dataset["x"][start_index:end_index]
        events[:, 2] = self.dataset["y"][start_index:end_index]
        events[:, 3] = self.dataset["p"][start_index:end_index]

        # Change polarity value 0 => -1
        events[:, 3][events[:, 3] == 0] = -1

        # Change time microsec to sec
        events[:, 0] = events[:, 2] / 1e6
        
        return events
