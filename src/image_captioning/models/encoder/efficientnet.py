from typing import Tuple

from torch import nn, Tensor
from torchvision.models import efficientnet
from image_captioning.models.model_registry import register_model


@register_model
class EfficientNet(nn.Module):
    def __init__(self, name: str, **kwargs):
        """
        EfficientNet model for image encoding.

        Args:
            name (str): Name of the EfficientNet variant.
            avg_pool_size (int): Size of the adaptive average pooling layer.
            **kwargs: Additional keyword arguments to be passed to the EfficientNet constructor.
        """
        super().__init__()
        self.model: efficientnet.EfficientNet = getattr(efficientnet, name)(**kwargs)
        # remove head
        self.model.avgpool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.model._forward_impl = lambda x: self.model.features(x)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass

        Args:
            x (Tensor): input tensor (B, 3, 224, 224)

        Returns:
            tuple[Tensor, Tensor]: encoded_image, global_features (B, 49, 2048), (B, 2048)
        """
        encoded_features: Tensor = self.model(x)
        batch_size = encoded_features.size(0)
        feature_size = encoded_features.size(1)
        num_pixels = encoded_features.size(2) * encoded_features.size(3)

        # get the global feature by using average pooling
        global_features = self.avgpool(encoded_features).view(batch_size, -1)

        encoded_image = encoded_features.permute(0, 2, 3, 1)
        encoded_image = encoded_image.view(batch_size, num_pixels, feature_size)

        return encoded_image, global_features
    
    def set_finetune_layer(self, num_layers: int = 8):
        """Set the number of layers to finetune

        Args:
            num_layers (int, optional): Number of layers to finetune. Defaults to 8.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.features[-num_layers:].parameters():
            param.requires_grad = True