import os
import warnings
from typing import Any, Callable, Tuple, Union
import autoreject
import mne
import scipy.io as scio
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

mne.set_log_level('CRITICAL')
warnings.filterwarnings("ignore")
DEFAULT_CHANNEL_LIST = [
    'FC3', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP4'
]
DEFAULT_SAMPLING_RATE = 256

class EEGDataset(BaseDataset):

    def __init__(self,
                 root_path: str = './dataset',
                 duration: int = 1,
                 num_channel: int = 11,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {
            'root_path': root_path,
            'duration': duration,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def process_record(file: Any = None,
                       root_path: str = './dataset',
                       duration: int = 1,
                       sampling_rate: int = 128,
                       num_channel: int = 11,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        trial_folder = file
        trial_files = os.listdir(os.path.join(root_path, trial_folder))
        write_pointer = 0

        for trial_file in trial_files:
            trial_id = os.path.join(trial_folder, trial_file)
            try:
                trial_data = scio.loadmat(os.path.join(root_path, trial_folder,
                                                   trial_file),
                                      verify_compressed_data_integrity=False)
            except Exception as e:
                print(f'Error reading {trial_id}: {e}')
                continue
            run_samples = trial_data['EEGdata'].transpose(2, 0, 1)
            run_labels = trial_data['EEGdatalabel'][:, 0]

            ch_names = [
                ch[0].tolist()[0]
                for ch in trial_data['configuration_channel'][0] if ch[1].sum()
            ]

            assert ch_names == DEFAULT_CHANNEL_LIST, f'Channel_list {ch_names} is not correct.'

            ch_types = ['eeg'] * len(ch_names)
            ch_names_lower = [ch_name.lower() for ch_name in ch_names]

            info = mne.create_info(ch_names=ch_names_lower,
                                   sfreq=DEFAULT_SAMPLING_RATE,
                                   ch_types=ch_types)
            montage = mne.channels.make_standard_montage('standard_1020')
            montage.ch_names = [ch_name.lower() for ch_name in montage.ch_names]

            for run_id, (run_sample,
                         run_label) in enumerate(zip(run_samples, run_labels)):
                eeg_start, eeg_end = 9*DEFAULT_SAMPLING_RATE, 13*DEFAULT_SAMPLING_RATE
                run_sample = run_sample[:, eeg_start:eeg_end]
                run_info = {
                    'trial_id': trial_id,
                    'run_id': run_id,
                    'label': run_label
                }

                raw = mne.io.RawArray(run_sample, info)
                raw.set_montage(montage)

                raw = raw.filter(l_freq=8, h_freq=48)
                raw = raw.resample(sampling_rate)

                epochs = mne.make_fixed_length_epochs(raw,
                                                      duration=duration,
                                                      preload=True)
                rejector = autoreject.AutoReject(
                    cv=10 if len(epochs) > 10 else len(epochs), verbose=False)
                epochs = rejector.fit_transform(epochs)

                clip_samples = epochs.get_data()

                clip_samples = (clip_samples - clip_samples.min(axis=0)) / (
                    clip_samples.max(axis=0) - clip_samples.min(axis=0))

                if before_trial:
                    clip_samples = before_trial(clip_samples)

                for i, clip_sample in enumerate(clip_samples):
                    clip_id = f'{trial_folder}_{write_pointer}'

                    record_info = {
                        'clip_id': clip_id,
                        'start_at': i * duration * DEFAULT_SAMPLING_RATE,
                        'end_at': (i + 1) * duration * DEFAULT_SAMPLING_RATE
                    }
                    record_info.update(run_info)

                    if offline_transform:
                        clip_sample = offline_transform(
                            eeg=clip_sample[:num_channel])
                        clip_sample = clip_sample['eeg']
                    

                    yield {
                        'eeg': clip_sample,
                        'key': clip_id,
                        'info': record_info
                    }

                    write_pointer += 1

    def set_records(self,
                    root_path: str = './dataset',
                    **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        return [
            f for f in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, f))
        ]

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label