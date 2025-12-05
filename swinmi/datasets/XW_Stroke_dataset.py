import os
import re
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path


class XWStrokeDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = "./XWStrokeDataset",
        chunk_size: int = 500,
        overlap: int = 0,
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
    ) -> None:
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix="datasets")

        # Read metadata files
        self.root_path = root_path
        self.subjects_info = pd.read_csv(os.path.join(root_path, "participants.tsv"), sep="\t")

        elec1 = pd.read_csv(os.path.join(root_path, "task-motor-imagery_electrodes.tsv"), sep="\t")
        elec2 = pd.read_csv(os.path.join(root_path, "task-motor-imagery_channels.tsv"), sep="\t")
        self.electodes_info = pd.merge(elec1, elec2, on="name", how="outer")

        # Insert a reference channel CPz at a fixed index (kept from original script)
        reference = {"name": "CPz", "type": "EEG", "status": "good", "status_description": "reference"}
        # Safely compute insertion index (fallback to end if index not available)
        try:
            insert_index = self.electodes_info.index[self.electodes_info.index.get_loc(17)]
        except Exception:
            insert_index = len(self.electodes_info)

        self.electodes_info = pd.concat([
            self.electodes_info.iloc[:insert_index],
            pd.DataFrame([reference], index=[insert_index]),
            self.electodes_info.iloc[insert_index:],
        ]).reset_index(drop=True)

        self.events_info = pd.read_csv(os.path.join(root_path, "task-motor-imagery_events.tsv"), sep="\t")

        params = dict(
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
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
        # Update instance dict with params so BaseDataset has access
        self.__dict__.update(params)

    @staticmethod
    def _read_edf_fixed_length_epochs(file_path: str, duration: float = 8.0) -> np.ndarray:
        """Load EDF and split into fixed-length epochs using MNE."""
        raw = mne.io.read_raw_edf(file_path, preload=True)
        epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True)
        return epochs.get_data()

    @staticmethod
    def process_record_edf(
        file: str,
        chunk_size: int,
        overlap: int,
        offline_transform: Optional[Callable] = None,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """Generator to yield baseline and clips from an EDF record.

        Yields dictionaries containing keys 'key' and 'eeg' for baseline entries,
        and entries with 'eeg', 'key' and 'info' for clips.
        """
        m = re.findall(r"sub-(\d\d)_task-motor-imagery_eeg.edf", file)
        if not m:
            raise ValueError(f"EDF filename doesn't match expected pattern: {file}")
        subject_id = int(m[0])

        data = XWStrokeDataset._read_edf_fixed_length_epochs(file)
        eeg = data[:, :30, :]

        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, "chunk_size must be larger than overlap"
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000

            write_pointer = 0
            baseline_id = f"{trial_id}_{write_pointer}"
            yield {"key": baseline_id, "eeg": eeg_baseline}
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if offline_transform is not None:
                    eeg_clip = offline_transform(eeg=eeg_clip, baseline=eeg_baseline)["eeg"]

                # keep prior shape convention (1, 4, 30, 128) if that's intended by downstream code
                eeg_clip = eeg_clip.reshape(1, 4, 30, 128)

                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    "label": label,
                    "trial_id": trial_id,
                    "baseline_id": baseline_id,
                    "subject_id": subject_id,
                }
                yield {"eeg": eeg_clip, "key": clip_id, "info": record_info}

                start += step
                end += step
                write_pointer += 1

    @staticmethod
    def process_record(
        file: str,
        chunk_size: int,
        overlap: int,
        offline_transform: Optional[Callable] = None,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """Generator to yield baseline and clips from a .mat record."""
        m = re.findall(r"sub-(\d\d)_task-motor-imagery_eeg.mat", file)
        if not m:
            raise ValueError(f"MAT filename doesn't match expected pattern: {file}")
        subject_id = int(m[0])

        fdata = loadmat(os.path.join(file))
        X, Y = fdata["eeg"][0][0]
        # X.shape expected [n_trials, n_channels, n_timepoints]; Y contains labels
        eeg = X[:, :30, :]

        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, "chunk_size must be larger than overlap"
            start = 1250
            step = chunk_size - overlap
            end = start + chunk_size
            end_time_point = 3000

            write_pointer = 0
            baseline_id = f"{trial_id}_{write_pointer}"
            yield {"key": baseline_id, "eeg": eeg_baseline}
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if offline_transform is not None:
                    eeg_clip = offline_transform(eeg=eeg_clip, baseline=eeg_baseline)["eeg"]

                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    "label": label,
                    "trial_id": trial_id,
                    "baseline_id": baseline_id,
                    "subject_id": subject_id,
                }

                yield {"eeg": eeg_clip, "key": clip_id, "info": record_info}

                start += step
                end += step
                write_pointer += 1

    # -------------------------- dataset wiring --------------------------
    def set_records(self, root_path: str, **kwargs) -> list:
        """Return list of per-subject record file paths under root_path/sourcedata.

        It expects each subject folder under sourcedata to contain a single data file
        (either .mat or .edf) and returns the first file found in each subject folder.
        """
        subject_dir = os.path.join(root_path, "sourcedata")
        records = []
        for sub in sorted(os.listdir(subject_dir)):
            subdir = os.path.join(subject_dir, sub)
            files = os.listdir(subdir)
            if not files:
                continue
            records.append(os.path.join(subdir, files[0]))
        return records

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = self.read_info(index)
        eeg_index = str(info.get("clip_id"))
        eeg_record = str(info.get("_record_id"))
        baseline_index = str(info.get("baseline_id"))

        signal = self.read_eeg(eeg_record, eeg_index)
        baseline = self.read_eeg(eeg_record, baseline_index)

        if self.online_transform:
            signal = self.online_transform(eeg=signal, baseline=baseline)["eeg"]

        # Remove the leading singleton dim if present
        if hasattr(signal, "squeeze"):
            signal = signal.squeeze(0)

        if self.label_transform:
            info = self.label_transform(y=info)["y"]

        return signal, info

    @property
    def repr_body(self) -> Dict[str, Any]:
        base = super().repr_body
        base.update(
            {
                "root_path": self.root_path,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "online_transform": str(self.online_transform),
                "offline_transform": str(self.offline_transform),
                "label_transform": str(self.label_transform),
                "before_trial": str(self.before_trial),
                "after_trial": str(self.after_trial),
                "after_session": str(self.after_session),
                "after_subject": str(self.after_subject),
                "io_path": self.io_path,
                "io_size": self.io_size,
                "io_mode": self.io_mode,
                "num_worker": self.num_worker,
                "verbose": self.verbose,
            }
        )
        return base
