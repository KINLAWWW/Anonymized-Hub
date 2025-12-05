import logging
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from swinmi.utils.gradcam import compute_attention_map, get_prefrontal_mask, combined_loss

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader

log = logging.getLogger('torcheeg')


def classification_metrics(metric_list: List[str], num_classes: int):
    """Return a MetricCollection based on the provided metric list and number of classes."""
    allowed_metrics = [
        'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa'
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose from {allowed_metrics}."
            )

    metric_dict = {
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
        'precision': torchmetrics.Precision(task='multiclass', average='macro', num_classes=num_classes),
        'recall': torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes),
        'f1score': torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes),
        'matthews': torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=num_classes),
        'auroc': torchmetrics.AUROC(task='multiclass', num_classes=num_classes),
        'kappa': torchmetrics.CohenKappa(task='multiclass', num_classes=num_classes)
    }

    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class AttentionTrainer(pl.LightningModule):
    """
    A generic trainer class for EEG classification.

    Example usage:

        trainer = AttentionTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        model (nn.Module): Classification model. Output dimension should equal the number of categories.
        num_classes (int): Number of classes in the dataset.
        lr (float): Learning rate. Default: 0.001
        weight_decay (float): Weight decay. Default: 0.0
        devices (int): Number of devices to use. Default: 1
        accelerator (str): Accelerator type: 'cpu' or 'gpu'. Default: 'cpu'
        metrics (list of str): Metrics to use. Options: 'precision', 'recall', 'f1score', 
                               'accuracy', 'matthews', 'auroc', 'kappa'. Default: ['accuracy']
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        devices: int = 1,
        accelerator: str = "cpu",
        metrics: List[str] = ["accuracy"]
    ):
        super().__init__()
        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        # Loss function and attention mask
        self.ce_fn = nn.CrossEntropyLoss()
        self.mask = get_prefrontal_mask(grid_size=(3, 3))

        # Initialize metrics
        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        """Initialize train, validation, and test metrics."""
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    # ----------------------
    # Lightning fit/test wrapper
    # ----------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 300, *args, **kwargs) -> Any:
        """Train the model using PyTorch Lightning trainer."""
        trainer = pl.Trainer(
            devices=self.devices,
            accelerator=self.accelerator,
            enable_checkpointing=False,
            logger=False,
            max_epochs=max_epochs,
            *args,
            **kwargs
        )
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args, **kwargs) -> _EVALUATE_OUTPUT:
        """Test the model using PyTorch Lightning trainer."""
        trainer = pl.Trainer(
            devices=self.devices,
            accelerator=self.accelerator,
            enable_checkpointing=False,
            logger=False,
            *args,
            **kwargs
        )
        return trainer.test(self, test_loader)

    # ----------------------
    # Forward / Predict
    # ----------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def predict_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        """Predict step for LightningModule."""
        x, y = batch
        y_hat = self(x)
        return y_hat

    # ----------------------
    # Training / Validation / Test Steps
    # ----------------------
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with attention-based combined loss."""
        x, y = batch
        y_hat = self(x)

        # Compute attention-based activation maps
        activation_maps = compute_attention_map(self.model, x)

        # Compute combined loss
        loss = combined_loss(
            outputs=y_hat,
            labels=y,
            activation_maps=activation_maps,
            mask=self.mask,
            device=self.device
        )

        # Log loss and metrics
        self.log("train_loss", self.train_loss(loss), prog_bar=True, on_epoch=False, logger=False, on_step=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}", metric_value(y_hat, y), prog_bar=True, on_epoch=False, logger=False, on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Compute and log train metrics at the end of epoch."""
        self.log("train_loss", self.train_loss.compute(), prog_bar=False, on_epoch=True, on_step=False, logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}", metric_value.compute(), prog_bar=False, on_epoch=True, on_step=False, logger=True)

        # Print metrics
        str_log = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str_log += f"{key}: {value:.3f} "
        log.info(str_log + '\n')

        # Reset metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with attention-based combined loss."""
        x, y = batch
        y_hat = self(x)

        activation_maps = compute_attention_map(self.model, x)
        loss = combined_loss(
            outputs=y_hat,
            labels=y,
            activation_maps=activation_maps,
            mask=self.mask,
            device=self.device
        )

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of epoch."""
        self.log("val_loss", self.val_loss.compute(), prog_bar=False, on_epoch=True, on_step=False, logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}", metric_value.compute(), prog_bar=False, on_epoch=True, on_step=False, logger=True)

        # Print metrics
        str_log = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str_log += f"{key}: {value:.3f} "
        log.info(str_log + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step using CrossEntropyLoss."""
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at the end of epoch."""
        self.log("test_loss", self.test_loss.compute(), prog_bar=False, on_epoch=True, on_step=False, logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}", metric_value.compute(), prog_bar=False, on_epoch=True, on_step=False, logger=True)

        # Print metrics
        str_log = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str_log += f"{key}: {value:.3f} "
        log.info(str_log + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    # ----------------------
    # Optimizer
    # ----------------------
    def configure_optimizers(self):
        """Configure Adam optimizer."""
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
