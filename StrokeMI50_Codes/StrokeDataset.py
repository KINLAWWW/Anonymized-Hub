import os
from typing import Any, Callable, Dict, Tuple, Union
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path
from scipy.io import loadmat
import re
import pandas as pd
import mne
from scipy.signal import stft
import numpy as np
from torcheeg.transforms import EEGTransform

class BaselineCorrection(EEGTransform):
    def __init__(self,axis=-1):
        super(BaselineCorrection, self).__init__(apply_to_baseline=False)
        self.axis=axis

    def __call__(self, *args, eeg: any, baseline= None, **kwargs) :
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)


    def apply(self, eeg, **kwargs) -> any:
        
         if kwargs['baseline'] is None:
            return eeg
         return eeg - kwargs['baseline'].mean(self.axis,keepdims= True)
    
    @property
    def targets_as_params(self):
        return ['baseline']
    
    def get_params_dependent_on_targets(self, params):
        return {'baseline': params['baseline']}


class StrokePatientsMIDataset(BaseDataset):
    def __init__(self,
                 root_path='./StrokePatientsMIDataset',
                 chunk_size: int = 500,
                 overlap: int = 250,
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
                 verbose: bool = True,
                ):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        self.subjects_info = pd.read_csv(os.path.join(root_path,
                                                      'participants.tsv'),
                                         sep='\t')
        self.electodes_info = pd.read_csv(os.path.join(
            root_path, "task-motor-imagery_electrodes.tsv"),
                                          sep='\t')
        electodes_info2 = pd.read_csv(os.path.join(
            root_path, "task-motor-imagery_channels.tsv"),
                                      sep='\t')
        self.electodes_info = pd.merge(self.electodes_info,
                                       electodes_info2,
                                       on='name',
                                       how='outer')
        refence = {
            'name': 'CPz',
            'type': 'EEG',
            'status': 'good',
            'status_description': 'refence'
        }

        insert_index = self.electodes_info.index[
            self.electodes_info.index.get_loc(17)]
        self.electodes_info = pd.concat([
            self.electodes_info.iloc[:insert_index],
            pd.DataFrame([refence], index=[insert_index]),
            self.electodes_info.iloc[insert_index:]
        ])
        self.electodes_info.index = range(len(self.electodes_info))

        self.events_info = pd.read_csv(os.path.join(
            root_path, 'task-motor-imagery_events.tsv'),
                                       sep='\t')

        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
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
            'verbose': verbose,
        }
        super().__init__(**params)
        self.__dict__.update(params)

    

    @staticmethod
    def process_record_edf(file,
                           chunk_size: int,
                           overlap: int,
                           offline_transform: Union[None, Callable] = None,
                           **kwargs):
        subject_id = int(
            re.findall("sub-(\d\d)_task-motor-imagery_eeg.edf", file)[0])
        edf_reader = mne.io.read_raw_edf(file, preload=True)
        epochs = mne.make_fixed_length_epochs(edf_reader,
                                              duration=8,
                                              preload=True)
        data = epochs.get_data(
        )  

        eeg = data[:, :30, :]

        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}"
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000
            
            write_pointer = 0
            baseline_id = f"{trial_id}_{write_pointer}"
            yield_dict = {'key': baseline_id,'eeg':eeg_baseline}
            yield yield_dict
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if (not offline_transform is None):
                    eeg_clip = offline_transform(eeg=eeg_clip,
                                                 baseline=eeg_baseline)['eeg']
                eeg_clip = eeg_clip.reshape(1,4, 30,128)

                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    'label': label,
                    'trial_id': trial_id,
                    'baseline_id': baseline_id,
                    'subject_id': subject_id
                }
                yield {'eeg':eeg_clip,'key': clip_id, "info": record_info}
                start, end = start + step, end + step
                write_pointer += 1

    @staticmethod
    def process_record(file,
                           chunk_size: int,
                           overlap: int,
                           offline_transform: Union[None, Callable] = None,
                           **kwargs):
        subject_id = int(
            re.findall("sub-(\d\d)_task-motor-imagery_eeg.mat", file)[0])
        fdata = loadmat(os.path.join(file))
        X, Y = fdata['eeg'][0][
            0]
        Y = Y[:, 0]
        eeg = X[:, :30, :]


        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}"
            start = 1250
            step = chunk_size - overlap
            end = start + chunk_size
            end_time_point = 3000
            write_pointer = 0
            baseline_id = f"{trial_id}_{write_pointer}"
            baseline_yield_dict = {'key': baseline_id,'eeg':eeg_baseline}
            yield baseline_yield_dict
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if (not offline_transform is None):
                    eeg_clip = offline_transform(eeg=eeg_clip,
                                                 baseline=eeg_baseline)['eeg']
                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    'label': label,
                    'trial_id': trial_id,
                    'baseline_id': baseline_id,
                    'subject_id': subject_id
                }
        
                yield {'eeg':eeg_clip,'key': clip_id, "info": record_info}
                start, end = start + step, end + step
                write_pointer += 1

    def set_records(self, root_path, **kwargs):
        subject_dir = os.path.join(root_path, 'sourcedata')
        return [
            os.path.join(os.path.join(subject_dir, sub),
                         os.listdir(os.path.join(subject_dir, sub))[0])
            for sub in os.listdir(subject_dir)
        ]

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        baseline_index = str(info['baseline_id'])
        signal = self.read_eeg(eeg_record, eeg_index)
        baseline = self.read_eeg(eeg_record, baseline_index)

        if self.online_transform:
            signal = self.online_transform(eeg=signal,
                                            baseline=baseline)['eeg']
        signal = signal.squeeze(0)
        if self.label_transform:
            info = self.label_transform(y=info)['y']
        
        return signal, info

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_session': self.after_session,
                'after_subject': self.after_subject,
                'io_path': self.io_path,
                'io_size': self.io_size,
                'io_mode': self.io_mode,
                'num_worker': self.num_worker,
                'verbose': self.verbose
            })