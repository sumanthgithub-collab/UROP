import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPMultimodalSentiment(nn.Module):
    def __init__(self, freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
                
        # CLIP text feature size = 512, image feature size = 512
        self.text_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract features from CLIP
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.clip.text_projection(text_outputs.pooler_output)
        
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(vision_outputs.pooler_output)
        
        t_feat = self.text_proj(text_embeds)
        i_feat = self.image_proj(image_embeds)
        
        fused = torch.cat([t_feat, i_feat], dim=1)
        logits = self.classifier(fused)
        return logits


class CLIPMultimodalImprovedClassifier(nn.Module):
    """
    Improved Classifier for CLIP Multimodal Sentiment Analysis.
    Features L2 normalization, LayerNorm, contrastive cross-product alignment, and regularized dropout.
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
        
        # Classifier operating on normalized embeddings + cross-modal interaction
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3 + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, text_feat, image_feat):
        t_embed = self.text_proj(text_feat)
        i_embed = self.image_proj(image_feat)
        
        # Unit sphere L2 normalization for contrastive alignment
        t_norm = t_embed / (t_embed.norm(dim=-1, keepdim=True) + 1e-8)
        i_norm = i_embed / (i_embed.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Element-wise interaction and scalar cosine similarity
        interaction = t_norm * i_norm
        cos_sim = (t_norm * i_norm).sum(dim=-1, keepdim=True)
        
        fused = torch.cat([t_norm, i_norm, interaction, cos_sim], dim=1)
        return self.classifier(fused)
