from typing import Dict, Tuple, Union
from scipy.interpolate import griddata
import numpy as np
from torcheeg.transforms import EEGTransform
import torch
import torch.nn.functional as F

class To2d(EEGTransform):
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg[np.newaxis, ...]


class ToGrid(EEGTransform):
    def __init__(self,
                 channel_location_dict: Dict[str, Tuple[int, int]],
                 apply_to_baseline: bool = False):
        super(ToGrid, self).__init__(apply_to_baseline=apply_to_baseline)
        self.channel_location_dict = channel_location_dict

        loc_x_list = []
        loc_y_list = []
        for _, locs in channel_location_dict.items():
            if locs is None:
                continue
            (loc_y, loc_x) = locs
            loc_x_list.append(loc_x)
            loc_y_list.append(loc_y)

        self.width = 9
        self.height = 9

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        num_bands = eeg.shape[0]
        num_electrodes = eeg.shape[1]
        timestep = eeg.shape[2]

        outputs = np.zeros((num_bands, self.height, self.width, timestep))
        for band_idx in range(num_bands):
            for i, locs in enumerate(self.channel_location_dict.values()):
                if locs is None:
                    continue
                (loc_y, loc_x) = locs
                outputs[band_idx, loc_y, loc_x, :] = eeg[band_idx, i, :] 
        outputs = outputs.transpose(0, 3, 1, 2)
        return outputs

    def reverse(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        eeg = eeg.transpose(1, 2, 0)
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        return {
            'eeg': outputs
        }

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'channel_location_dict': {...}})


class ToInterpolatedGrid(EEGTransform):
    def __init__(self,
                 channel_location_dict: Dict[str, Tuple[int, int]],
                 apply_to_baseline: bool = False):
        super(ToInterpolatedGrid,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.channel_location_dict = channel_location_dict
        self.location_array = np.array(list(channel_location_dict.values()))

        loc_x_list = []
        loc_y_list = []
        for _, (loc_x, loc_y) in channel_location_dict.items():
            loc_x_list.append(loc_x)
            loc_y_list.append(loc_y)

        self.width = max(loc_x_list) + 1
        self.height = max(loc_y_list) + 1

        self.grid_x, self.grid_y = np.mgrid[
            min(self.location_array[:, 0]):max(self.location_array[:, 0]
                                               ):self.width * 1j,
            min(self.location_array[:,
                                    1]):max(self.location_array[:,
                                                                1]):self.height *
            1j, ]


    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        eeg = eeg.transpose(1, 0)
        outputs = []

        for timestep_split_y in eeg:
            outputs.append(
                griddata(self.location_array,
                         timestep_split_y, (self.grid_x, self.grid_y),
                         method='cubic',
                         fill_value=0))
        outputs = np.array(outputs)
        return outputs

    def reverse(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        eeg = eeg.transpose(1, 2, 0)
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        return {
            'eeg': outputs
        }
        
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'channel_location_dict': {...}})
