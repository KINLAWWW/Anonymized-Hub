from typing import Dict, Tuple, Union, List
from scipy.interpolate import griddata
import numpy as np
from torcheeg.transforms import EEGTransform
import torch
import torch.nn.functional as F
import scipy

class Resize2d(EEGTransform):
    def __init__(self, size: Union[int, tuple], mode: str = 'bilinear', align_corners: bool = False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        eeg_resized = F.interpolate(eeg_tensor, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return eeg_resized.numpy()


class ToTensor(EEGTransform):
    def __init__(self, apply_to_baseline: bool = False):
        super(ToTensor, self).__init__(apply_to_baseline=apply_to_baseline)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> torch.Tensor:
        eeg = np.expand_dims(eeg, axis=0)
        return torch.from_numpy(eeg).float()

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


class Downsample(EEGTransform):
    def __init__(self,
                 num_points: int,
                 axis: Union[int, None] = -1,
                 apply_to_baseline: bool = False):
        super(Downsample, self).__init__(apply_to_baseline=apply_to_baseline)
        self.num_points = num_points
        self.axis = axis

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs):
        times_tamps = np.linspace(0,
                                  eeg.shape[self.axis] - 1,
                                  self.num_points,
                                  dtype=int)
        return eeg.take(times_tamps, axis=self.axis)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'num_points': self.num_points,
            'axis': self.axis
        })

class SetSamplingRate(EEGTransform):
    def __init__(self,origin_sampling_rate:int, target_sampling_rate:int, 
                 apply_to_baseline=False,
                 axis= -1,
                 scale:bool=False,
                 res_type:str='soxr_hq'):
        super(SetSamplingRate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.original_rate = origin_sampling_rate
        self.new_rate = target_sampling_rate
        self.axis = axis
        self.scale = scale
        self.res_type = res_type

    def apply(self,
        eeg: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        import lazy_loader as lazy
        samplerate = lazy.load("samplerate")
        resampy = lazy.load("resampy")
        soxr = lazy.load('soxr')

        eeg = eeg.astype(np.float32)

        if self.original_rate == self.new_rate:
            return eeg

        ratio = float(self.new_rate) / self.original_rate

        n_samples = int(np.ceil(eeg.shape[self.axis] * ratio))

        if self.res_type in ("scipy", "fft"):
            EEG_res = scipy.signal.resample(eeg, n_samples, axis=self.axis)
        elif self.res_type == "polyphase":
            self.original_rate = int(self.original_rate)
            self.new_rate = int(self.new_rate)
            gcd = np.gcd(self.original_rate, self.new_rate)
            EEG_res = scipy.signal.resample_poly(
                eeg, self.new_rate // gcd, self.original_rate // gcd, axis=self.axis
            )
        elif self.res_type in (
            "linear",
            "zero_order_hold",
            "sinc_best",
            "sinc_fastest",
            "sinc_medium",
        ):
            EEG_res = np.apply_along_axis(
                samplerate.resample, axis=self.axis, arr=eeg, ratio=ratio, converter_type=self.res_type
            )
        elif self.res_type.startswith("soxr"):
            EEG_res = np.apply_along_axis(
                soxr.resample,
                axis=self.axis,
                arr=eeg,
                in_rate=self.original_rate,
                out_rate=self.new_rate,
                quality=self.res_type,
            )
        else:
            EEG_res = resampy.resample(eeg, self.original_rate, self.new_rate, filter=self.res_type, axis=self.axis)

        if self.scale:
            EEG_res /= np.sqrt(ratio)

        return np.asarray(EEG_res, dtype=eeg.dtype)

    @property
    def __repr__(self)->any :
        return  f'''{
                'original_sampling_rate': self.original_rate,
                'target_sampling_rate': self.new_rate,
                'apply_to_baseline':self.apply_to_baseline
                'axis': self.axis,
                'scale': self.scale,
                'res_type': self.res_type
            }'''