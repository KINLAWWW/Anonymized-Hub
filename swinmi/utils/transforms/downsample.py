from typing import Union, Dict
import numpy as np
import scipy
from torcheeg.transforms import EEGTransform


class Downsample(EEGTransform):
    r"""
    Downsample the EEG signal to a specified number of data points.

    Args:
        num_points (int): The number of data points after downsampling.
        axis (int, optional): The dimension to downsample. Default is -1 (last axis).
        apply_to_baseline (bool): Whether to apply the transform to the baseline as well. Default is False.

    Example:
        >>> from torcheeg import transforms
        >>> t = transforms.Downsample(num_points=32, axis=-1)
        >>> t(eeg=np.random.randn(32, 128))['eeg'].shape
        (32, 32)
    """
    def __init__(self,
                 num_points: int,
                 axis: Union[int, None] = -1,
                 apply_to_baseline: bool = False):
        super(Downsample, self).__init__(apply_to_baseline=apply_to_baseline)
        self.num_points = num_points
        self.axis = axis

    def __call__(self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        timestamps = np.linspace(0, eeg.shape[self.axis] - 1, self.num_points, dtype=int)
        return eeg.take(timestamps, axis=self.axis)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'num_points': self.num_points,
            'axis': self.axis
        })


class SetSamplingRate(EEGTransform):
    r"""
    Resample an EEG series from origin_sampling_rate to target_sampling_rate.

    Args:
        origin_sampling_rate (int): Original sampling rate of EEG.
        target_sampling_rate (int): Target sampling rate of EEG.
        apply_to_baseline (bool): Apply to baseline if provided. Default False.
        axis (int): Axis along which to resample. Default -1.
        scale (bool): Scale output to preserve total energy. Default False.
        res_type (str): Resampling method. Options include:
            'soxr_hq', 'soxr_vhq', 'soxr_mq', 'soxr_lq', 'scipy', 'fft',
            'polyphase', 'linear', 'zero_order_hold', 'sinc_best', etc.

    Example:
        >>> t = SetSamplingRate(origin_sampling_rate=500, target_sampling_rate=128)
        >>> t(eeg=np.random.randn(32, 1000))['eeg'].shape
        (32, 256)
    """
    def __init__(self,
                 origin_sampling_rate: int,
                 target_sampling_rate: int,
                 apply_to_baseline: bool = False,
                 axis: int = -1,
                 scale: bool = False,
                 res_type: str = 'soxr_hq'):
        super(SetSamplingRate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.original_rate = origin_sampling_rate
        self.new_rate = target_sampling_rate
        self.axis = axis
        self.scale = scale
        self.res_type = res_type

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        import lazy_loader as lazy
        samplerate = lazy.load("samplerate")
        resampy = lazy.load("resampy")
        soxr = lazy.load("soxr")

        eeg = eeg.astype(np.float32)

        if self.original_rate == self.new_rate:
            return eeg

        ratio = float(self.new_rate) / self.original_rate
        n_samples = int(np.ceil(eeg.shape[self.axis] * ratio))

        if self.res_type in ("scipy", "fft"):
            eeg_res = scipy.signal.resample(eeg, n_samples, axis=self.axis)
        elif self.res_type == "polyphase":
            gcd = np.gcd(int(self.original_rate), int(self.new_rate))
            eeg_res = scipy.signal.resample_poly(eeg, self.new_rate // gcd, self.original_rate // gcd, axis=self.axis)
        elif self.res_type in ("linear", "zero_order_hold", "sinc_best", "sinc_medium", "sinc_fastest"):
            eeg_res = np.apply_along_axis(samplerate.resample, axis=self.axis, arr=eeg, ratio=ratio, converter_type=self.res_type)
        elif self.res_type.startswith("soxr"):
            eeg_res = np.apply_along_axis(
                soxr.resample,
                axis=self.axis,
                arr=eeg,
                in_rate=self.original_rate,
                out_rate=self.new_rate,
                quality=self.res_type
            )
        else:
            eeg_res = resampy.resample(eeg, self.original_rate, self.new_rate, filter=self.res_type, axis=self.axis)

        if self.scale:
            eeg_res /= np.sqrt(ratio)

        return np.asarray(eeg_res, dtype=eeg.dtype)

    @property
    def __repr__(self) -> str:
        return f"SetSamplingRate(original_sampling_rate={self.original_rate}, " \
               f"target_sampling_rate={self.new_rate}, apply_to_baseline={self.apply_to_baseline}, " \
               f"axis={self.axis}, scale={self.scale}, res_type='{self.res_type}')"
