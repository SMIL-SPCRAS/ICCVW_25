# data_loading/feature_extractor.py

import torch
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import CLIPModel

class PretrainedImageEmbeddingExtractor:
    """
    Извлекает эмбеддинги из изображений
    """

    def __init__(self, config):
        """
        Параметры в config:
         - image_classifier_checkpoint (str)
         - emb_device (str)
         - cut_target_layer (str)
        """
        self.config = config
        self.device = config.emb_device
        self.image_model_type = config.image_model_type

        if self.image_model_type == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.model = resnet18(weights=weights).to(self.device)
            self.model.eval()
            self.features = nn.Sequential(*list(self.model.children())[:-config.cut_target_layer])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.image_model_type == "емоresnet50":
            self.model = torch.jit.load(config.image_classifier_checkpoint).to(
                self.device
            )
            self.model.eval()
            self.features = nn.Sequential(
                *list(self.model.children())[: -config.cut_target_layer]
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.image_model_type == "emo":
            self.model = torch.jit.load(config.image_classifier_checkpoint).to(
                self.device
            )
            self.model.eval()
            self.features = nn.Sequential(
                *list(self.model.children())[: -config.cut_target_layer]
            )
            self.feature_final = self.model.fc_feats  # Убираем avgpool и fc

        elif self.image_model_type == 'clip':
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        else:
            raise ValueError(
                f"❌ Неизвестный image_model_type: {self.image_model_type}"
            )

    def extract(self, x):
        """
        :return: тензор (B*seq_len, hidden_dim)
        """

        with torch.no_grad():
            if self.image_model_type == "resnet18" or self.image_model_type == "емоresnet50":
                x = self.features(x)  # [batch, 512, 7, 7] (для 224x224)
                x = self.avgpool(x) # [batch, 512, 1, 1] (для 224x224)
                x = x.view(x.shape[0], -1)  # [batch, 512]
            elif self.image_model_type == "emo":
                x = self.features(x)  # [batch, 512, 7, 7] (для 224x224)
                x = x.view(x.size(0), -1)  # flatten
                x = self.feature_final(x)
            elif self.image_model_type == "clip":
                x = self.model.get_image_features(x)
                # x /= x.norm(dim=-1, keepdim=True)
        return x
