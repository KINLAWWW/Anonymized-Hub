import logging
import os
import re
from copy import copy
from typing import List, Tuple, Union, Dict

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

log = logging.getLogger('torcheeg')


class KFoldGroupbyTrial:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[float, None] = None,
        split_path: Union[None, str] = None
    ):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path
        self.k_fold = model_selection.KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        train_infos = {}
        test_infos = {}
        trial_ids = list(set(info['trial_id']))

        for trial_id in trial_ids:
            trial_info = info[info['trial_id'] == trial_id]
            for i, (train_index, test_index) in enumerate(self.k_fold.split(trial_info)):
                train_info = trial_info.iloc[train_index]
                test_info = trial_info.iloc[test_index]

                if i not in train_infos:
                    train_infos[i] = []
                if i not in test_infos:
                    test_infos[i] = []

                train_infos[i].append(train_info)
                test_infos[i].append(test_info)

        for i in train_infos.keys():
            train_info = pd.concat(train_infos[i], ignore_index=True)
            test_info = pd.concat(test_infos[i], ignore_index=True)
            train_info.to_csv(os.path.join(self.split_path, f'train_fold_{i}.csv'), index=False)
            test_info.to_csv(os.path.join(self.split_path, f'test_fold_{i}.csv'), index=False)

    @property
    def fold_ids(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info('📊 | Create the split of train and test set.')
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, '
                f'if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.')
            log.info('💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.')

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            train_info = pd.read_csv(os.path.join(self.split_path, f'train_fold_{fold_id}.csv'))
            test_info = pd.read_csv(os.path.join(self.split_path, f'test_fold_{fold_id}.csv'))

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset


class KFoldPerSubjectGroupbyTrial:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[float, None] = None,
        split_path: Union[None, str] = None
    ):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path
        self.k_fold = model_selection.KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]

            subject_train_infos = {}
            subject_test_infos = {}

            trial_ids = list(set(subject_info['trial_id']))
            for trial_id in trial_ids:
                trial_info = subject_info[subject_info['trial_id'] == trial_id]

                for i, (train_index, test_index) in enumerate(self.k_fold.split(trial_info)):
                    train_info = trial_info.iloc[train_index]
                    test_info = trial_info.iloc[test_index]

                    if i not in subject_train_infos:
                        subject_train_infos[i] = []
                    if i not in subject_test_infos:
                        subject_test_infos[i] = []

                    subject_train_infos[i].append(train_info)
                    subject_test_infos[i].append(test_info)

            for i in subject_train_infos.keys():
                subject_train_info = pd.concat(subject_train_infos[i], ignore_index=True)
                subject_test_info = pd.concat(subject_test_infos[i], ignore_index=True)

                subject_train_info.to_csv(os.path.join(self.split_path, f'train_subject_{subject}_fold_{i}.csv'), index=False)
                subject_test_info.to_csv(os.path.join(self.split_path, f'test_subject_{subject}_fold_{i}.csv'), index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    @property
    def fold_ids(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][1])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(
        self,
        dataset: BaseDataset,
        subject: Union[int, None] = None
    ) -> Tuple[BaseDataset, BaseDataset, int]:
        if not os.path.exists(self.split_path):
            log.info('📊 | Create the split of train and test set.')
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, '
                f'if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.')
            log.info('💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.')

        subjects = self.subjects
        fold_ids = self.fold_ids

        if subject is not None:
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if subject is not None and local_subject != subject:
                continue

            for fold_id in fold_ids:
                train_info = pd.read_csv(os.path.join(self.split_path, f'train_subject_{local_subject}_fold_{fold_id}.csv'))
                test_info = pd.read_csv(os.path.join(self.split_path, f'test_subject_{local_subject}_fold_{fold_id}.csv'))

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, test_dataset, local_subject

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            if i:
                format_string += ', '
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string
