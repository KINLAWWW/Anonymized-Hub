import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torcheeg import transforms
from classifier import ClassifierTrainer
from HSSDataset import EEGDataset
from train_test_split import KFoldGroupbyTrial
from to import ToInterpolatedGrid, To2d
from hssdict import HSS_LOCATION_DICT
import pandas as pd
import os
from model import SwinMI
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

HYPERPARAMETERS = {
    "seed": 42,
    "batch_size": 64,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_epochs": 50,
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(HYPERPARAMETERS['seed'])

subject_path = '../data/sub1'
subject_name = os.path.basename(os.path.normpath(subject_path))
save_path = os.path.join('.torcheeg/InterpolatedDataset', subject_name)

dataset = EEGDataset(io_path=save_path,
                     root_path=subject_path,
                     offline_transform=transforms.Compose([
                        To2d(),
                        ToInterpolatedGrid(HSS_LOCATION_DICT)
                     ]),
                     online_transform=transforms.ToTensor(),
                     label_transform=transforms.Compose([
                         transforms.Select('label'),
                         transforms.Lambda(lambda x: x - 1)
                     ]))
print(len(dataset))
print(dataset[0][0].shape)
cv_path = os.path.join('.torcheeg/Interpolatedselection', subject_name)

cv = KFoldGroupbyTrial(
    n_splits=5,
    shuffle=True,
    split_path=cv_path,
    random_state=42)

metrics = ['accuracy', 'recall', 'precision', 'f1score','kappa']
csv_path = 'logs/SwinMI_KFPSGT_results.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
for i, (training_dataset, test_dataset) in enumerate(cv.split(dataset)):
    model = SwinMI(patch_size=(8,2,2),
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
        patience=20,
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
    row["subject"] = subject_name
    columns_order = ["subject"] + ["fold"] + metrics
    row_df = pd.DataFrame([row])[columns_order]

    if not os.path.exists(csv_path):
        row_df.to_csv(csv_path, index=False, float_format='%.4f')
    else:
        row_df.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.4f')

