from typing import Dict, Tuple, Union
from scipy.interpolate import griddata
import numpy as np
import torch
import torch.nn.functional as F
from torcheeg.transforms import EEGTransform


class Resize2d(EEGTransform):
    def __init__(self, size: Union[int, tuple], mode: str = 'bilinear', align_corners: bool = False):
        """
        Args:
            size (tuple or int): Target size (height, width).
            mode (str): Interpolation mode (e.g., 'bilinear', 'nearest', 'bicubic').
            align_corners (bool): Whether to align the corners of input and output. Recommended False for 'bilinear'.
        """
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            eeg (np.ndarray): Input EEG signal with shape [batch, channels, height, width].

        Returns:
            np.ndarray: Resized EEG with shape [batch, channels, new_height, new_width].
        """
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        eeg_resized = F.interpolate(eeg_tensor, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return eeg_resized.numpy()


class ToTensor(EEGTransform):
    """
    Convert a numpy.ndarray to torch.Tensor. Does not perform scaling.
    """
    def __init__(self, apply_to_baseline: bool = False):
        super().__init__(apply_to_baseline=apply_to_baseline)

    def __call__(self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg (np.ndarray): Input EEG signals.
            baseline (np.ndarray, optional): Baseline signals, if apply_to_baseline=True.

        Returns:
            dict: {'eeg': tensor} or {'eeg': tensor, 'baseline': tensor}.
        """
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> torch.Tensor:
        eeg = np.expand_dims(eeg, axis=0)
        return torch.from_numpy(eeg).float()


class To2d(EEGTransform):
    """
    Convert EEG to 2D representation with shape [1, num_electrodes, num_points].
    """
    def __call__(self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg[np.newaxis, ...]


class ToGrid(EEGTransform):
    """
    Project EEG channels to a 2D grid according to electrode positions.
    """
    def __init__(self, channel_location_dict: Dict[str, Tuple[int, int]], apply_to_baseline: bool = False):
        super().__init__(apply_to_baseline=apply_to_baseline)
        self.channel_location_dict = channel_location_dict
        self.width = 9
        self.height = 9

    def __call__(self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            eeg (np.ndarray): EEG of shape (num_bands, num_electrodes, timestep).

        Returns:
            np.ndarray: Grid EEG of shape (num_bands, timestep, height, width).
        """
        num_bands = eeg.shape[0]
        num_electrodes = eeg.shape[1]
        timestep = eeg.shape[2]

        outputs = np.zeros((num_bands, self.height, self.width, timestep))
        for band_idx in range(num_bands):
            for i, locs in enumerate(self.channel_location_dict.values()):
                if locs is None:
                    continue
                loc_y, loc_x = locs
                outputs[band_idx, loc_y, loc_x, :] = eeg[band_idx, i, :]

        outputs = outputs.transpose(0, 3, 1, 2)  # (num_bands, timestep, height, width)
        return outputs

    def reverse(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        """
        Inverse operation to extract original electrode signals from the grid.
        """
        eeg = eeg.transpose(1, 2, 0)
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        return {'eeg': outputs}

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

        self.width = max(loc_y_list) + 1
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
        eeg = np.squeeze(eeg, axis=0)
        eeg = eeg.transpose(1, 0)
        outputs = []

        for timestep_split_y in eeg:
            outputs.append(
                griddata(self.location_array,
                         timestep_split_y, (self.grid_x, self.grid_y),
                         method='cubic',
                         fill_value=0))
        
        outputs = np.array(outputs).astype(np.float32)
        return np.expand_dims(outputs, axis=0)

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
