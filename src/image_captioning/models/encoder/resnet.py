from typing import Tuple

from torch import nn, Tensor
from torchvision.models import resnet
from image_captioning.models.model_registry import register_model


@register_model
class ResNet(nn.Module):
    def __init__(self, name: str, avg_pool_size: int = 7, **kwargs):
        super().__init__()
        model = getattr(resnet, name)(**kwargs)
        # remove head
        self.resnet = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AvgPool2d(avg_pool_size)
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass

        Args:
            x (Tensor): input tensor (B, 3, 224, 224)

        Returns:
            tuple[Tensor, Tensor]: encoded_image, global_features (B, 49, 2048), (B, 2048)
        """
        encoded_features: Tensor = self.resnet(x) # (B, 2048, 7, 7)
        batch_size = encoded_features.size(0)
        feature_size = encoded_features.size(1)
        num_pixels = encoded_features.size(2) * encoded_features.size(3)

        # get the global feature by using average pooling
        global_features = self.avgpool(encoded_features).view(batch_size, -1) # (B, 2048)

        encoded_image = encoded_features.permute(0, 2, 3, 1) # (B, 7, 7, 2048)
        encoded_image = encoded_image.view(batch_size, num_pixels, feature_size) # (B, 49, 2048)
        
        return encoded_image, global_features
    
    def set_finetune_layer(self, num_layers: int = 8):
        """Set the number of layers to finetune

        Args:
            num_layers (int, optional): Number of layers to finetune. Defaults to 8.
        """
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet[-num_layers:].parameters():
            param.requires_grad = True