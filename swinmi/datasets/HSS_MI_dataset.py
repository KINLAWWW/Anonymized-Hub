import os
import warnings
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import autoreject
import mne
import numpy as np
from scipy.io import loadmat
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path


mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")


DEFAULT_CHANNEL_LIST = [
    "FC3", "FC4", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "CP3", "CP4"
]
DEFAULT_SAMPLING_RATE = 256


class HSSMIDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = "./data",
        duration: int = 1,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        num_channel: int = 11,
        online_transform: Optional[Callable] = None,
        offline_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        before_trial: Optional[Callable] = None,
        after_trial: Optional[Callable] = None,
        after_session: Optional[Callable] = None,
        after_subject: Optional[Callable] = None,
        io_path: Optional[str] = None,
        io_size: int = 1048576,
        io_mode: str = "lmdb",
        num_worker: int = 0,
        verbose: bool = True,
    ):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix="datasets")

        params = dict(
            root_path=root_path,
            duration=duration,
            sampling_rate=sampling_rate,
            num_channel=num_channel,
            online_transform=online_transform,
            offline_transform=offline_transform,
            label_transform=label_transform,
            before_trial=before_trial,
            after_trial=after_trial,
            after_session=after_session,
            after_subject=after_subject,
            io_path=io_path,
            io_size=io_size,
            io_mode=io_mode,
            num_worker=num_worker,
            verbose=verbose,
        )

        super().__init__(**params)
        self.__dict__.update(params)

    # ----------------------------------------------------------------------
    # Static generator to process .mat EEG records
    # ----------------------------------------------------------------------
    @staticmethod
    def process_record(file: Any = None,
                       root_path: str = '../data',
                       duration: int = 1,
                       sampling_rate: int = 128,
                       num_channel: int = 11,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        """
        Generator that processes a .mat EEG file and yields normalized epochs.

        Yields format:
            {
                "eeg": clip_sample,
                "key": clip_id,
                "info": { ... }
            }
        """
        # ---------------------- Load MATLAB file ----------------------
        try:
            mat = loadmat(file, verify_compressed_data_integrity=False)
        except Exception as e:
            raise RuntimeError(f"Error loading {file}: {e}")

        eeg_data = mat["EEGdata"].transpose(2, 0, 1)
        eeg_labels = mat["EEGdatalabel"][:, 0]

        # Channel list check
        ch_names = [
            ch[0].tolist()[0]
            for ch in mat["configuration_channel"][0] if ch[1].sum()
        ]
        assert ch_names == DEFAULT_CHANNEL_LIST, f"Incorrect channel list: {ch_names}"

        # ---------------------- Prepare MNE info ----------------------
        ch_types = ["eeg"] * len(ch_names)
        ch_names_lower = [c.lower() for c in ch_names]
        info = mne.create_info(
            ch_names=ch_names_lower,
            sfreq=DEFAULT_SAMPLING_RATE,
            ch_types=ch_types
        )
        montage = mne.channels.make_standard_montage("standard_1020")
        montage.ch_names = [ch_name.lower() for ch_name in montage.ch_names]

        # Extract subject/trial
        subject_id = os.path.basename(os.path.dirname(file))
        trial_id = os.path.basename(file)

        write_pointer = 0

        # ---------------------- Iterate over runs ----------------------
        for run_id, (run, label) in enumerate(zip(eeg_data, eeg_labels)):

            # crop 9–13 seconds
            start, end = 9 * DEFAULT_SAMPLING_RATE, 13 * DEFAULT_SAMPLING_RATE
            run = run[:, start:end]

            raw = mne.io.RawArray(run, info)
            raw.set_montage(montage)
            raw = raw.filter(l_freq=8, h_freq=48)
            raw = raw.resample(sampling_rate)

            # epoching
            epochs = mne.make_fixed_length_epochs(
                raw, duration=duration, preload=True
            )

            # autoreject
            rejector = autoreject.AutoReject(
                cv=min(10, len(epochs)),
                verbose=False
            )
            epochs = rejector.fit_transform(epochs)

            clips = epochs.get_data()
            # Min–max normalization
            clips = (clips - clips.min(axis=0)) / (clips.max(axis=0) - clips.min(axis=0))

            if before_trial:
                clips = before_trial(clips)

            # ---------------------- Yield clips ----------------------
            for i, clip in enumerate(clips):
                clip_id = f"{trial_id}_{write_pointer}"

                record_info = dict(
                    clip_id=clip_id,
                    subject_id=subject_id,
                    trial_id=trial_id,
                    run_id=run_id,
                    label=label,
                    start_at=i * duration * sampling_rate,
                    end_at=(i + 1) * duration * sampling_rate,
                )

                if offline_transform:
                    clip = offline_transform(eeg=clip[:num_channel])["eeg"]

                yield {
                    "eeg": clip,
                    "key": clip_id,
                    "info": record_info,
                }

                write_pointer += 1

    # ----------------------------------------------------------------------
    # Return list of record files (.mat)
    # ----------------------------------------------------------------------
    def set_records(self, root_path: str, **kwargs) -> list:
        records = []
        for subject in os.listdir(root_path):
            spath = os.path.join(root_path, subject)
            if not os.path.isdir(spath):
                continue

            for trial in os.listdir(spath):
                tpath = os.path.join(spath, trial)
                if os.path.isdir(tpath):
                    for f in os.listdir(tpath):
                        if f.endswith(".mat"):
                            records.append(os.path.join(tpath, f))
        return records


    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = self.read_info(index)
        eeg_index = str(info["clip_id"])
        record_id = str(info["_record_id"])

        eeg = self.read_eeg(record_id, eeg_index)
        label = info

        if self.online_transform:
            eeg = self.online_transform(eeg=eeg)["eeg"]

        if self.label_transform:
            label = self.label_transform(y=info)["y"]

        return eeg, label
