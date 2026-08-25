import torch
import torch.nn as nn
from transformers import ViltModel, ViltProcessor

class ViLTMultimodalSentiment(nn.Module):
    def __init__(self, freeze_vilt=False):
        super().__init__()
        # Use vilt-b32-mlm backbone
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        
        if freeze_vilt:
            for param in self.vilt.parameters():
                param.requires_grad = False
                
        # ViLT hidden size = 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None):
        vilt_outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )
        # Pooler output is the joint [CLS] representation (768-dim)
        cls_feat = vilt_outputs.pooler_output
        logits = self.classifier(cls_feat)
        return logits


class ViLTImprovedCrossModalClassifier(nn.Module):
    """
    Improved Bi-directional Cross-Attention Classifier for ViLT / Vision-Language Transformers.
    Incorporates multi-head cross-attention (text-query-image and image-query-text), residual skip connections, LayerNorm, and regularized dropout.
    """
    def __init__(self, text_dim=768, image_dim=2048, joint_dim=256, num_heads=4, dropout=0.4):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.GELU()
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.GELU()
        )
        
        self.cross_attn_text = nn.MultiheadAttention(embed_dim=joint_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_img = nn.MultiheadAttention(embed_dim=joint_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm_t = nn.LayerNorm(joint_dim)
        self.norm_i = nn.LayerNorm(joint_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, text_feat, image_feat):
        t_proj = self.text_proj(text_feat).unsqueeze(1) # (B, 1, joint_dim)
        i_proj = self.image_proj(image_feat).unsqueeze(1) # (B, 1, joint_dim)
        
        # Bi-directional Cross-Attention
        t_attn, _ = self.cross_attn_text(query=t_proj, key=i_proj, value=i_proj)
        i_attn, _ = self.cross_attn_img(query=i_proj, key=t_proj, value=t_proj)
        
        # Residual + Norm
        t_out = self.norm_t(t_proj + t_attn).squeeze(1)
        i_out = self.norm_i(i_proj + i_attn).squeeze(1)
        
        fused = torch.cat([t_out, i_out], dim=1)
        return self.classifier(fused)
