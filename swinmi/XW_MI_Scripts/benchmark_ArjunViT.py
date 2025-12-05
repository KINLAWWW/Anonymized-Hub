from torch.utils.data import DataLoader
from torcheeg.transforms import Select, BandSignal, Compose, ToTensor
from swinmi.utils.transforms import SetSamplingRate, BaselineCorrection

from swinmi.datasets import XWStrokeDataset, XWSTROKE_LOCATION_DICT
from torcheeg.model_selection import KFoldPerSubject, KFoldPerSubjectGroupbyTrial, KFoldGroupbyTrial, KFold
from torcheeg.models import ArjunViT
from swinmi.trainers import ClassifierTrainer

CONFIG = {
    "batch_size": 12,
    "lr": 1e-4,
    "weight_decay": 0,
    "num_epochs": 50,
    "n_splits": 4,
    "root_path": "./XW_Stroke",
    "io_path": ".torcheeg/XW_Stroke",
    "cv_strategy": "KFoldPerSubject",  # "KFoldPerSubjectGroupbyTrial", "KFoldGroupbyTrial",  "KFold"
}

cv_mapping = {
    "KFoldPerSubject": (KFoldPerSubject, ".torcheeg/XW_Stroke_P-Sub"),
    "KFoldPerSubjectGroupbyTrial": (KFoldPerSubjectGroupbyTrial, ".torcheeg/XW_Stroke_P-Sub-T"),
    "KFoldGroupbyTrial": (KFoldGroupbyTrial, ".torcheeg/XW_Stroke_KFold-T"),
    "KFold": (KFold, ".torcheeg/XW_Stroke_KFold")
}

def build_dataset():
    """Create dataset with offline & online preprocessing."""
    return XWStrokeDataset(
        root_path=CONFIG["root_path"],
        io_path=CONFIG["io_path"],
        chunk_size=500,
        overlap=250,
        offline_transform=Compose([
            BaselineCorrection(),
            SetSamplingRate(origin_sampling_rate=500,target_sampling_rate=128),
            BandSignal(sampling_rate=128,band_dict={'frequency_range':[8,40]})
        ]),
        online_transform=ToTensor(),
        label_transform=Select('label'),
        num_worker=8
    )


def build_model():
    return ArjunViT(chunk_size=128,
                    t_patch_size=32,
                    num_electrodes=30,
                    num_classes=2)


def train_and_evaluate_fold(fold_idx, train_set, test_set, metrics):
    model = build_model()
    trainer = ClassifierTrainer(
        model=model,
        num_classes=2,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        metrics=metrics,
        accelerator="gpu"
    )

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"\n===== Fold {fold_idx} =====")
    print(f"Train size = {len(train_set)}, Test size = {len(test_set)}")

    trainer.fit(
        train_loader,
        test_loader,
        max_epochs=CONFIG["num_epochs"],
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0.0
    )

    test_result = trainer.test(
        test_loader,
        enable_progress_bar=False,
        enable_model_summary=True
    )[0]

    fold_data = {metric: test_result[f"test_{metric}"] for metric in metrics}
    fold_data["fold"] = fold_idx

    print(f"Fold {fold_idx} results: {fold_data}")
    return fold_data


def main():
    metrics = ["accuracy", "recall", "precision", "f1score", "kappa"]

    print("Loading dataset ...")
    dataset = build_dataset()
    print("Dataset ready:")
    print("Sample input shape:", dataset[0][0].shape)
    print("Sample label:", dataset[0][1])
    print("Total samples:", len(dataset))

    if CONFIG["cv_strategy"] not in cv_mapping:
        raise ValueError(f"cv_strategy must be one of {list(cv_mapping.keys())}")

    cv_class, split_path = cv_mapping[CONFIG["cv_strategy"]]
    cv = cv_class(
        n_splits=CONFIG["n_splits"],
        shuffle=True,
        split_path=split_path
    )

    for fold_idx, (train_set, test_set) in enumerate(cv.split(dataset), start=1):
        train_and_evaluate_fold(fold_idx, train_set, test_set, metrics)


if __name__ == "__main__":
    main()
