import os

import dotenv
import pynwb
import numpy as np
import pytorch_lightning as pl
import torch 
from torch.utils.data import  DataLoader, TensorDataset

from nlb_tools.nwb_interface import NWBDataset
from scipy.interpolate import interp1d
from torch.nn.utils.rnn import pad_sequence

from support_functions import create_recon_data, to_tensor

dotenv.load_dotenv(override=True)
HOME_DIR = os.environ["HOME_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
EVAL_INPUT_FILE = os.environ["EVAL_INPUT_FILE"]
EVAL_TARGET_FILE = os.environ["EVAL_TARGET_FILE"]


import multiprocessing

# Global variable for multiprocessing fork-shared data to avoid IPC overhead
_MP_SPIKE_TIMES = None

def _bin_single_trial(args):
    trial_idx, start, stop, dt, num_neurons = args
    global _MP_SPIKE_TIMES
    
    time_bins = np.arange(start, stop, dt)
    num_bins = len(time_bins) - 1
    spike_matrix = np.zeros((num_bins, num_neurons), dtype=np.float32)
    
    for neuron, spike_times in enumerate(_MP_SPIKE_TIMES):
        # 1. Binary search (O(log N)) instead of scanning the full array
        left_idx = np.searchsorted(spike_times, start)
        right_idx = np.searchsorted(spike_times, stop)
        trial_spikes = spike_times[left_idx:right_idx]
        
        if len(trial_spikes) > 0:
            # 2. C-level np.bincount instead of np.histogram
            bin_indices = ((trial_spikes - start) / dt).astype(np.int32)
            valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
            bin_indices = bin_indices[valid_mask]
            if len(bin_indices) > 0:
                counts = np.bincount(bin_indices, minlength=num_bins)
                spike_matrix[:, neuron] = counts[:num_bins]
                
    return trial_idx, spike_matrix


class NLBDataModule(pl.LightningDataModule):
    """
    Loads data from nwb/HDF5 files and buids PyTorch
    TensorDataset and Dataloaders
    """
    def __init__(
            self,
            train_data_file,
            eval_data_file,
            phase = "val", # phase of running, "val" vs. "test"
            bin_width = 5,
            forward_ratio = 0.2,
            batch_size = 64,
            num_workers = 0,
        ): 
        super().__init__()
        self.save_hyperparameters()

    def _process_nwbfile(self, file_path, data_type, train_mask = None):

        io = pynwb.NWBHDF5IO(file_path, 'r')
        nwbfile = io.read()
        print(f"{data_type} dataset successfully read")
        
        units = nwbfile.units.to_dataframe()
        trials = nwbfile.trials.to_dataframe()
        is_blind_test = False

        if train_mask is None:
            is_heldout = units['heldout'].values.astype(bool)
            is_heldin = ~is_heldout
            is_blind_test = False
        else:
            num_train_heldin = (~train_mask).sum()           
            if len(units) == len(train_mask):
                is_heldout = train_mask
                is_heldin = ~is_heldout
                is_blind_test = False
            elif len(units) == num_train_heldin:
                print("detected to be blind test")
                is_heldout = None
                is_heldin = np.ones(len(units),dtype=bool)
                is_blind_test = True
            else:
                raise ValueError("eval units matches neither total train units nor train held in units")

        num_neurons = len(units)
        dt = self.hparams.bin_width / 1000.0

        has_behavior = 'behavior' in nwbfile.processing
        if has_behavior:
            hand_vel_interface = nwbfile.processing['behavior']['hand_vel']
            vel_raw = hand_vel_interface.data[:]
            vel_timestamps = hand_vel_interface.timestamps[:]

        dataset = {
            "heldin": [],
            "heldin_forward": [],
            "heldout": [],
            "heldout_forward": [],
            "behavior": []
        }
        
        #——————— obtain neuron spike information ———————
        spike_times_list = list(units['spike_times'])
        
        global _MP_SPIKE_TIMES
        _MP_SPIKE_TIMES = spike_times_list
        
        pool_inputs = []
        for idx, (_, trial) in enumerate(trials.iterrows()):
            pool_inputs.append((
                idx,
                trial['start_time'],
                trial['stop_time'],
                dt,
                num_neurons
            ))
            
        # Use multiprocessing Pool to process trials in parallel
        # Limit to 8 cores or cpu_count to be friendly to other users on the cluster
        num_cores = min(multiprocessing.cpu_count(), 8)
        with multiprocessing.Pool(processes=num_cores) as pool:
            mp_results = pool.map(_bin_single_trial, pool_inputs)
            
        # Clear global reference to free memory
        _MP_SPIKE_TIMES = None
        
        # Sort results by trial index to ensure exact ordering
        mp_results.sort(key=lambda x: x[0])
        spike_matrices = [res[1] for res in mp_results]

        #—————— get behavior(hand velocity) & update dataset ———————
        for i, (_, trial) in enumerate(trials.iterrows()):
            start = trial['start_time']
            stop = trial['stop_time']
            
            time_bins = np.arange(start, stop, dt)
            num_bins = len(time_bins) - 1
            
            spike_matrix = spike_matrices[i]

            if has_behavior:
                trial_vel_mask = (vel_timestamps > start) & (vel_timestamps < stop)
                trial_vel_timestamps = vel_timestamps[trial_vel_mask]
                trial_vel_raw = vel_raw[trial_vel_mask]

                bin_centers = time_bins[:-1] + (dt/2.0)
                interpolator = interp1d(
                    trial_vel_timestamps,
                    trial_vel_raw,
                    axis = 0,
                    kind = 'linear',
                    bounds_error = False,
                    fill_value = "extrapolate"
                )
                behavior = interpolator(bin_centers)
            else:
                behavior = np.zeros((num_bins, 2), dtype=np.float32)
                
            #update
            t_split = int(num_bins * (1-self.hparams.forward_ratio))

            heldin_matrix = spike_matrix[:,is_heldin]
            dataset["heldin"].append(torch.from_numpy(heldin_matrix[:t_split, :]))
            dataset["heldin_forward"].append(torch.from_numpy(heldin_matrix[t_split:, :]))

            if not is_blind_test:
                heldout_matrix = spike_matrix[:, is_heldout]
                dataset["heldout"].append(torch.from_numpy(heldout_matrix[:t_split, :]))
                dataset["heldout_forward"].append(torch.from_numpy(heldout_matrix[t_split:, :]))
            else:
                dataset["heldout"].append(torch.empty((t_split, 0)))
                dataset["heldout_forward"].append(torch.empty((num_bins - t_split, 0)))

            dataset["behavior"].append(torch.from_numpy(behavior).float())

            
        io.close()
        print(f"{data_type} dataset successfully closed")
        
        for key in dataset.keys():
            dataset[key] = pad_sequence(dataset[key], batch_first = True, padding_value = 0.0)

        return dataset, is_heldout

    def setup(self, stage = None):
        train_data_path = os.path.join(DATA_DIR, self.hparams.train_data_file)
        eval_data_path = os.path.join(DATA_DIR, self.hparams.eval_data_file)

        print('processing training data...')
        train_tensors, train_mask = self._process_nwbfile(train_data_path, "train")
        train_recon = create_recon_data(
            train_tensors['heldin'], train_tensors['heldin_forward'],
            train_tensors['heldout'], train_tensors['heldout_forward']
        )
        self.train_data = (train_tensors['heldin'], train_recon, train_tensors['behavior'])
        self.train_ds = TensorDataset(*self.train_data)


        print('processing eval data...')
        eval_tensors, _ = self._process_nwbfile(eval_data_path, "test", train_mask = train_mask)
        if len(eval_tensors['heldout'][0]) > 0:
            eval_recon = create_recon_data(
                eval_tensors['heldin'], eval_tensors['heldin_forward'],
                eval_tensors['heldout'], eval_tensors['heldout_forward']
            )
        else:
            eval_recon = torch.empty(0)
        self.valid_data = (eval_tensors['heldin'], eval_recon, eval_tensors['behavior'])
        self.valid_ds = TensorDataset(*self.valid_data)

        print("setup complete")

    def train_dataloader(self, shuffle = True): 
        """
        returns dataloader for the train data
        """
        train_dl = DataLoader(
            self.train_ds,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            persistent_workers = (self.hparams.num_workers > 0),
            pin_memory = True,
            shuffle = shuffle
        )
        return train_dl
    def val_dataloader(self): 
        """
        returns dataloader for the validation data
        """
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            persistent_workers = (self.hparams.num_workers > 0),
            pin_memory = True
        )
        return valid_dl 


#——— LORENTZ MODULE UESD FOR TESTING CODE INTEGRITY—————
class LorentzModule(pl.LightningDataModule):
    """Loads from preprocessed HDF5 files created using
    functions in `nlb_tools.make_tensors` and builds PyTorch
    `TensorDataset`s and `DataLoader`s that handle batching
    and shuffling.
    """

    def __init__(
            self,
            phase="val",
            batch_size=64,
            num_workers=4,
        ):
        super().__init__()
        self.save_hyperparameters()
        # Get the save path to the data
        self.save_path = os.path.join(
            DATA_DIR, f"lfads_lorenz.h5"
        )

    def setup(self, stage=None):
        """Loads the data from preprocessed HDF5 files.

        Parameters
        ----------
        stage : str, optional
            Ignored, by default None
        """
        # Load the training data arrays from file
        train_data_path = self.save_path
        with h5py.File(train_data_path, "r") as h5file:
            h5dict = {key: h5file[key][()] for key in h5file.keys()}
            #print(h5dict)
            train_data = to_tensor(h5file["train_data"][()])
            train_truth = to_tensor(h5file["train_truth"][()])
            train_inds = to_tensor(h5file["train_inds"][()])
            valid_data = to_tensor(h5file["valid_data"][()])
            valid_truth = to_tensor(h5file["valid_truth"][()])
            valid_inds = to_tensor(h5file["valid_inds"][()])
        self.train_data = (train_data, train_truth, train_inds)
        self.train_ds = TensorDataset(*self.train_data)
        self.valid_data = (valid_data, valid_truth, valid_inds)
        self.valid_ds = TensorDataset(*self.valid_data)

    def train_dataloader(self, shuffle=True):
        """Returns a dataloader for the training data.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data, by default True

        Returns
        -------
        torch.utils.data.DataLoader
            A dataloader that generates data during training.
        """
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return train_dl

    def valid_dataloader(self):
        """Returns a dataloader for the validation data.

        Returns
        -------
        torch.utils.data.DataLoader
            A dataloader that generates data during validation.
        """
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl