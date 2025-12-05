from torch.utils.data import DataLoader
from torcheeg import transforms

from swinmi.datasets import HSSMIDataset
from torcheeg.model_selection import KFoldPerSubject, KFoldPerSubjectGroupbyTrial, KFoldGroupbyTrial, KFold
from swinmi.models import MSCFormer
from swinmi.trainers import ClassifierTrainer

CONFIG = {
    "batch_size": 12,
    "lr": 1e-4,
    "weight_decay": 0,
    "num_epochs": 200,
    "n_splits": 4,
    "root_path": "./HSS_MI",
    "io_path": ".torcheeg/HSS_MI_to2d",
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
    return HSSMIDataset(
        root_path=CONFIG["root_path"],
        io_path=CONFIG["io_path"],
        sampling_rate=128,  
        offline_transform=transforms.Compose([transforms.To2d(), transforms.ToTensor()]),
        label_transform=transforms.Compose([
                        transforms.Select('label'),
                        transforms.Lambda(lambda x: x - 1)
                        ])
    )


def build_model():
    return MSCFormer(number_channel=11)

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
