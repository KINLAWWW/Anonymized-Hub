from typing import Union, Dict, List
import numpy as np
from torcheeg.transforms import EEGTransform
import scipy

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