import logging
import os
from typing import Tuple

import h5py
import numpy as np

from ..utils import estimate_corresponding_gt_flow, undistort_events
from . import DataLoaderBase

logger = logging.getLogger(__name__)


# hdf5 data loader
def h5py_loader(path: str):
    """Basic loader for .hdf5 files.
    Args:
        path (str) ... Path to the .hdf5 file.

    Returns:
        timestamp (dict) ... Doctionary of numpy arrays. Keys are "left" / "right".
        davis_left (dict) ... "event": np.ndarray.
        davis_right (dict) ... "event": np.ndarray.
    """
    data = h5py.File(path, "r")    

    dataset = dict()
    for columns in ['p', 'x', 'y', 't']:
        dataset[columns] = data['events']['table']['{}'.format(columns)]

    data.close()

    return dataset


class EnaviDataLoader(DataLoaderBase):
    """Dataloader class for MVSEC dataset."""

    NAME = "E-NAVI"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    # Override
    def set_sequence(self, sequence_name: str, undistort: bool = False) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        logger.info(f"Undistort events = {undistort}")

        # OWN

        self.dataset_files = self.get_sequence(sequence_name)
        dataset = h5py_loader(self.dataset_files["event"])

        #####

        # EVIMO2
        
        # dataset = dict()
        
        # dataset['p'] = np.load("{}\\dataset_events_p.npy".format(self.root_dir))
        # dataset['x'] = np.load("{}\\dataset_events_xy.npy".format(self.root_dir))[...,0]
        # dataset['y'] = np.load("{}\\dataset_events_xy.npy".format(self.root_dir))[...,1]
        # dataset['t'] = np.load("{}\\dataset_events_t.npy".format(self.root_dir)) * 1000000

        #####

        self.left_event = dataset # int16 .. for smaller memory consumption.
        self.left_ts = dataset["t"]  # float64

        # Setting up time suration statistics
        self.min_ts = self.left_ts.min()
        self.max_ts = self.left_ts.max()
        self.data_duration = self.max_ts - self.min_ts
        self.t_start_idx = 0

        self._OFFSET = self._OFFSET + self.min_ts

        for offset, time in enumerate(self.left_ts):
            if time >= self._OFFSET:
                self.t_start_idx = offset
                break

        self.t_end_idx = self.t_start_idx

        # Cambiar el número para aumentar los FPS
        self.timestamps_images = np.arange(self._OFFSET, self.max_ts, 100000)
        # Quitamos el último frame 
        self.image_indices = np.arange(len(self.timestamps_images)-1)

    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `outdoot_day2`.

        Returns:
            sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        data_path: str = os.path.join(self.root_dir, sequence_name)
        event_file = data_path + "/" + sequence_name + ".h5"

        sequence_file = {
            "event": event_file
        }

        return sequence_file

    def __len__(self):
        return len(self.left_event['x'])
    
    def get_max_image_indices(self) -> int:
        return len(self.image_indices)

    def load_event(self, start_index: int, end_index: int, cam: str = "") -> np.ndarray:
        """Load events.
        The original hdf5 file contains (x, y, t, p),
        where x means in width direction, and y means in height direction. p is -1 or 1.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value, in sec. p is [-1, 1].
        """
        n_events = end_index - start_index
        events = np.zeros((n_events, 4), dtype=np.float64)

        if len(self.left_event['x']) <= start_index:
            logger.error(
                f"Specified {start_index} to {end_index} index for {len(self.left_event)}."
            )
            raise IndexError
        events[:, 0] = self.left_event["y"][start_index:end_index]
        events[:, 1] = self.left_event["x"][start_index:end_index]
        events[:, 2] = self.left_ts[start_index:end_index]
        events[:, 3] = self.left_event["p"][start_index:end_index]

        # Change polarity value 0 => -1
        events[:, 3][events[:, 3] == 0] = -1

        # Change time microsec to sec
        events[:, 2] = events[:, 2] / 1e6
        return events

    def get_events(self, idx: int) -> np.ndarray:

        t_end_us = self.timestamps_images[idx+1]

        self.t_start_idx = self.t_end_idx

        for offset, time in enumerate(self.left_ts[self.t_start_idx:]):
            if time >= t_end_us:
                self.t_end_idx = self.t_start_idx + offset - 1
                break

        return self.load_event(self.t_start_idx, self.t_end_idx)