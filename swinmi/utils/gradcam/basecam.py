from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import ttach as tta

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
        detach: bool = True,
    ) -> None:
        """
        Base class for Grad-CAM with optional TTA smoothing.
        """
        self.model = model.eval()
        self.target_layers = target_layers
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.detach = detach

        if tta_transforms is None:
            self.tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ])
        else:
            self.tta_transforms = tta_transforms

        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform, self.detach
        )

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        """Compute weight for each channel. Must be implemented in subclass."""
        raise NotImplementedError

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)

        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()

        if len(activations.shape) == 4:  # 2D conv
            weighted_activations = weights[:, :, None, None] * activations
        elif len(activations.shape) == 5:  # 3D conv
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape: {activations.shape}")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module] = None, eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(cat) for cat in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([t(output) for t, output in zip(targets, outputs)])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                torch.autograd.grad(loss, input_tensor, retain_graph=True, create_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, ...]:
        if len(input_tensor.shape) == 4:
            return input_tensor.size(-1), input_tensor.size(-2)
        elif len(input_tensor.shape) == 5:
            return input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
        else:
            raise ValueError("Invalid input_tensor shape.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> List[np.ndarray]:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        cam_per_layer = []

        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None
            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            cam_per_layer.append(np.round(np.mean(cam, axis=1)[:, None, :], 3))

        return cam_per_layer

    def aggregate_multi_layers(self, cam_per_layer: List[np.ndarray]) -> np.ndarray:
        cam = np.concatenate(cam_per_layer, axis=1)
        cam = np.maximum(cam, 0)
        result = np.mean(cam, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)
            cam = torch.from_numpy(cam[:, None, :, :])
            cam = transform.deaugment_mask(cam)
            cams.append(cam[:, 0, :, :].numpy())
        return np.mean(np.float32(cams), axis=0)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        if aug_smooth:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)
        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"Exception in CAM: {exc_type}, Message: {exc_value}")
            return True
