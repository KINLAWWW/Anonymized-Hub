import numpy as np
from .basecam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None):
        """
        Grad-CAM class for computing class activation maps.

        Args:
            model (nn.Module): The model to explain.
            target_layers (list[nn.Module]): The layers to compute CAMs for.
            reshape_transform (callable, optional): Function to reshape the activations if needed.
        """
        super(GradCAM, self).__init__(model, target_layers, reshape_transform)

    def get_cam_weights(self, input_tensor, target_layer, target_category, activations, grads):
        """
        Compute the weights of each channel for CAM calculation.

        Args:
            input_tensor (torch.Tensor): The input to the model.
            target_layer (nn.Module): The layer for which CAM is computed.
            target_category (int): The target class index.
            activations (torch.Tensor): Activations from the target layer.
            grads (torch.Tensor): Gradients of the target class w.r.t. activations.

        Returns:
            np.ndarray: Weights for each channel.

        Notes:
            For 2D images (Batch, Features, Width, Height), the weight is the spatial average of gradients.
            For 3D images (Batch, Features, Times, Width, Height), the weight is the average over all spatial and temporal dimensions.
        """
        # 2D image (Batch, Features, Width, Height)
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 3D image (Batch, Features, Times, Width, Height)
        elif len(grads.shape) == 5:
            # Compute mean over time and spatial dimensions
            # This represents the overall importance of each channel for the target class,
            # which is the classic Grad-CAM weight calculation method.
            return np.mean(grads, axis=(2, 3, 4))
        
        else:
            raise ValueError(
                "Invalid grads shape. Shape of grads should be 4 (2D image) or 5 (3D image)."
            )
