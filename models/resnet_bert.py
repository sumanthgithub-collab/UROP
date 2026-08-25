import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50, ResNet50_Weights

class BERTResNet50EarlyFusion(nn.Module):
    def __init__(self, freeze_backbones=False):
        super().__init__()
        
        # Text Backbone: BERT-base-uncased (768-dim)
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Image Backbone: ResNet50 (2048-dim)
        weights = ResNet50_Weights.DEFAULT
        self.image_encoder = resnet50(weights=weights)
        self.image_encoder.fc = nn.Identity()
        self.image_projection = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        if freeze_backbones:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
        # Fusion + Classifier (256 + 256 = 512-dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0, :]
        text_feat = self.text_projection(text_cls)
        
        img_feat = self.image_encoder(image)
        img_feat = self.image_projection(img_feat)
        
        fused = torch.cat([text_feat, img_feat], dim=1)
        logits = self.classifier(fused)
        return logits


class BERTResNet50FeatureClassifier(nn.Module):
    """
    Improved Feature-based Classifier for ResNet + BERT Early Fusion.
    Integrates LayerNorm, GELU, Gated Fusion, and classifier dropout.
    """
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=256, dropout=0.4):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gated multimodal fusion mechanism
        self.gate_layer = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )
        
        # Joint classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, text_feat, image_feat):
        t_embed = self.text_proj(text_feat)
        i_embed = self.image_proj(image_feat)
        
        gate = self.gate_layer(torch.cat([t_embed, i_embed], dim=1))
        gated_embed = gate * t_embed + (1.0 - gate) * i_embed
        
        fused = torch.cat([gated_embed, t_embed, i_embed], dim=1)
        return self.classifier(fused)
