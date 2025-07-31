import torch
import torch.nn as nn
from StrokeDataset import StrokePatientsMIDataset, BaselineCorrection
from strokesdict import STROKEPATIENTSMI_LOCATION_DICT
from torcheeg.transforms import Select,BandSignal,Compose
from to import ToGrid, ToTensor, SetSamplingRate
import pandas as pd
import os
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from classifier import ClassifierTrainer
from eegswintransformer import SwinTransformer


dataset = StrokePatientsMIDataset(root_path='./subdataset',
                                  io_path='.torcheeg/ALLdataset',
                        chunk_size=500,
                        offline_transform=Compose(
                                [BaselineCorrection(),
                                SetSamplingRate(origin_sampling_rate=500,target_sampling_rate=128),
                                BandSignal(sampling_rate=128,band_dict={'frequency_range':[8,40]})
                                ]),
                        online_transform=Compose(
                                [ToGrid(STROKEPATIENTSMI_LOCATION_DICT),ToTensor()]),
                        label_transform=Select('label'),
                        num_worker=8
)

HYPERPARAMETERS = {
    "seed": 42,
    "batch_size": 16,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_epochs": 50,
}

from torcheeg.model_selection import KFoldPerSubject
cv = KFoldPerSubject(n_splits=5, shuffle=True,split_path='.torcheeg/ALLdataset_KFoldPerSubject',random_state=42)

metrics = ['accuracy', 'recall', 'precision', 'f1score','kappa']
csv_path = 'logs/SwinCam_KFPS_results.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)


for i, (training_dataset, test_dataset) in enumerate(cv.split(dataset)):
    model = SwinTransformer(patch_size=(8,3,3),
                            depths=(2, 6, 4),
                            num_heads=(3,6,8),
                            window_size=(3,3,3)
                            )
    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=HYPERPARAMETERS['lr'],
                                weight_decay=HYPERPARAMETERS['weight_decay'],
                                metrics=['accuracy', 'recall', 'precision', 'f1score', 'kappa'],
                                accelerator="gpu")
    training_loader = DataLoader(training_dataset,
                            batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=False)
    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=25,
        mode='min',
        verbose=True
    )
    trainer.fit(training_loader,
                test_loader,
                max_epochs=HYPERPARAMETERS['num_epochs'],
                callbacks=[early_stopping_callback],
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=0.0)
    training_result = trainer.test(training_loader,
                                enable_progress_bar=False,
                                enable_model_summary=True)[0]
    test_result = trainer.test(test_loader,
                            enable_progress_bar=False,
                            enable_model_summary=True)[0]
    row = {metric: test_result[f"test_{metric}"] for metric in metrics}
    row["fold"] = i + 1
    columns_order = ["fold"] + metrics
    row_df = pd.DataFrame([row])[columns_order]

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df_metrics = pd.concat([existing, row_df], ignore_index=True)
    else:
        df_metrics = row_df

    df_metrics.to_csv(csv_path, index=False, float_format='%.4f')