from typing import Dict, Union, Any
from torcheeg.transforms import EEGTransform


class BaselineRemoval(EEGTransform):
    """Subtract the baseline signal from EEG."""

    def __init__(self):
        super(BaselineRemoval, self).__init__(apply_to_baseline=False)

    def __call__(self, *args, eeg: Any, baseline: Union[Any, None] = None, **kwargs) -> Dict[str, Any]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: Any, **kwargs) -> Any:
        baseline = kwargs.get('baseline', None)
        if baseline is None:
            return eeg

        assert baseline.shape == eeg.shape, (
            f'Baseline shape ({baseline.shape}) must match EEG shape ({eeg.shape}).'
        )
        return eeg - baseline

    @property
    def targets_as_params(self):
        return ['baseline']

    def get_params_dependent_on_targets(self, params):
        return {'baseline': params['baseline']}


class BaselineCorrection(EEGTransform):
    """Subtract the mean of baseline signal from EEG."""

    def __init__(self, axis: int = -1):
        super(BaselineCorrection, self).__init__(apply_to_baseline=False)
        self.axis = axis

    def __call__(self, *args, eeg: Any, baseline: Union[Any, None] = None, **kwargs):
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: Any, **kwargs) -> Any:
        baseline = kwargs.get('baseline', None)
        if baseline is None:
            return eeg
        return eeg - baseline.mean(self.axis, keepdims=True)

    @property
    def targets_as_params(self):
        return ['baseline']

    def get_params_dependent_on_targets(self, params):
        return {'baseline': params['baseline']}
