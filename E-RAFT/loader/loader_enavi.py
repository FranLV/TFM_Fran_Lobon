import math
from pathlib import Path
from typing import Dict, Tuple
import weakref

import cv2
import h5py
from numba import jit
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import visualization as visu
from matplotlib import pyplot as plt
from utils import transformers
import os
import imageio

import pandas as pd

from utils.dsec_utils import RepresentationType, VoxelGrid, flow_16bit_to_float

VISU_INDEX = 1

class EventSlicer:
    def __init__(self, h5f: h5py.File, logger=None):
        self.logger = logger
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events']['table']['{}'.format(dset_str)]


        
        self.t_offset = self.events['t'][0]

        # Para probar la generación de más fotogramas multiplicando el tiempo 
        # self.events['t'] = self.events['t'] * 2
        # self.t_offset = self.events['t'][0] * 2

        self.t_final = int(self.events['t'][-1])

        self.t_start_idx = [0,0]
        self.t_end_idx = [0,0]

    def get_final_time_us(self):
        return self.t_final

    def get_time_offset(self):
        return self.t_offset

    def get_events(self, t_start_us: int, t_end_us: int, t_index: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        self.t_start_idx[t_index] = self.t_end_idx[t_index] + 1        

        for offset, time in enumerate(self.events['t'][self.t_start_idx[t_index]:]):
            if time > t_end_us:
                self.t_end_idx[t_index] = self.t_start_idx[t_index] + offset
                break

        self.logger.write_line("Index_events. Start: {} ; End: {}".format(self.t_start_idx[t_index], self.t_end_idx[t_index]), False)
        self.logger.write_line("Time us. Start: {} ; End: {}".format(t_start_us, t_end_us), False)

        events = dict()

        t_start_offset = self.t_start_idx[t_index]
        t_end_offset = self.t_end_idx[t_index]
        n_events = t_end_offset - t_start_offset
        max_events = 30000
        
        if n_events > max_events:
            if t_index == 0:
                t_start_offset = self.t_start_idx[t_index] + n_events - max_events
            else:
                t_end_offset = self.t_end_idx[t_index] - (n_events - max_events) 

        self.logger.write_line("Index_events_real. Start: {} ; End: {}".format(t_start_offset, t_end_offset), False)
        self.logger.write_line("N events: {} ; After: {}".format(n_events, n_events - max_events), False)

        events['t'] = np.asarray(self.events['t'][t_start_offset:t_end_offset])

        self.logger.write_line("Event size: {} ".format(events['t'].shape[0]), False)

        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_offset:t_end_offset])
        return events


    def get_events_no_idx(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us
        
        t_start_idx_found = False
        t_end_idx= 0
        t_start_idx= 0

        for offset, time in enumerate(self.events['t']):
            if time > t_end_us:
                t_end_idx = offset
                break

            if time >= t_start_us and not t_start_idx_found:
                t_start_idx_found = True
                t_start_idx = offset


        events = dict()
        events['t'] = np.asarray(self.events['t'][t_start_idx:t_end_idx])
        
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_idx:t_end_idx])
        return events

class Sequence(Dataset):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str='test', delta_t_ms: int=100,
                 num_bins: int=15, transforms=None, name_idx=0, visualize=False, logger=None):
        assert num_bins >= 1
        assert delta_t_ms == 100
        assert seq_path.is_dir()
        assert mode in {'train', 'test'}

        self.logger = logger
        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Just for now, we always train with num_bins=15
        assert self.num_bins==15

        # Set event representation
        self.voxel_grid = None
        if representation_type == RepresentationType.VOXEL:
            self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)

        # Left events only
        ev_data_file = seq_path / (seq_path.name + '.h5')

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location, logger=logger)

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        #Load and compute timestamps and indices
        timestamps_images = np.arange(self.event_slicer.get_time_offset(), self.event_slicer.get_final_time_us(), 50000)
        image_indices = np.arange(len(timestamps_images))

        # Al estar utilizar E-RAFT pre-entrenado, hay que utilizar ventanas de tiempo de 100ms, además habrá que eliminar el primero y el último índice.
        self.timestamps_flow = timestamps_images[::2][1:-1]
        self.indices = image_indices[::2][1:-1]
        self.idx_to_visualize = self.indices

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, p, t, x, y, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)

    def get_data_sample(self, index, crop_window=None, flip=None):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        ts_start = [self.timestamps_flow[index] - self.delta_t_us, self.timestamps_flow[index]]
        ts_end = [self.timestamps_flow[index], self.timestamps_flow[index] + self.delta_t_us]

        file_index = self.indices[index]

        output = {
            'file_index': file_index,
            'timestamp': self.timestamps_flow[index]
        }
        # Save sample for benchmark submission
        output['save_submission'] = file_index in self.idx_to_visualize
        output['visualize'] = self.visualize_samples

        for i in range(len(names)):
            event_data = self.event_slicer.get_events(ts_start[i], ts_end[i], i)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x >= crop_window['start_x']-2) & (x < crop_window['start_x']+crop_window['crop_width']+2)
                y_mask = (y >= crop_window['start_y']-2) & (y < crop_window['start_y']+crop_window['crop_height']+2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x = x[mask_combined]
                y = y[mask_combined]

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_voxel_grid(p, t, x, y)
                output[names[i]] = event_representation
            output['name_map']=self.name_idx
        return output

    def __getitem__(self, idx):
        print("Index: ", idx)
        sample =  self.get_data_sample(idx)
        return sample

class EnaviProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_ms: int=100, num_bins=15,
                 type='standard', config=None, visualize=False, logger=None):
        assert dataset_path.is_dir(), str(dataset_path)
        assert delta_t_ms == 100
        self.config=config
        self.name_mapper_test = []
        self.logger = logger

        test_sequences = list()
        for child in dataset_path.iterdir():
            self.name_mapper_test.append("{}_eraft".format(str(child).split("/")[-1]))
            if type == 'standard':
                test_sequences.append(Sequence(child, representation_type, 'test', delta_t_ms, num_bins,
                                               transforms=[],
                                               name_idx=len(self.name_mapper_test)-1,
                                               visualize=visualize,
                                               logger=self.logger))            
            else:
                raise Exception('Please provide a valid subtype [standard/warm_start] in config file! For E-NAVI dataset only standard is supported.')

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_test_dataset(self):
        return self.test_dataset


    def get_name_mapping_test(self):
        return self.name_mapper_test

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(self.test_dataset.datasets[0].num_bins), True)
