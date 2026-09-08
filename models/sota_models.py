import torch
import torch.nn as nn
import torch.nn.functional as F

class SOTA_TwitterRoBERTa_CLIP_Gated(nn.Module):
    """
    SOTA Model 1: Twitter-RoBERTa + CLIP-ViT-Large Gated Multimodal Fusion.
    Integrates dynamic modality gating, elementwise interaction, and strong LayerNorm/GELU regularization.
    """
    def __init__(self, text_dim=768, image_dim=768, proj_dim=256, hidden_dim=128, dropout=0.45):
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
        self.gate_layer = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        
        gate = self.gate_layer(torch.cat([t, i], dim=-1))
        gated = gate * t + (1.0 - gate) * i
        
        inter = t * i
        diff = torch.abs(t - i)
        
        fused = torch.cat([gated, t, i, inter], dim=-1)
        h = self.inter_proj(fused)
        return self.classifier(h)


class SOTA_DeBERTa_CLIP_Contrastive(nn.Module):
    """
    SOTA Model 2: DeBERTa-v3 + CLIP-ViT-Large Contrastive Interaction Classifier.
    Integrates L2 unit sphere normalization, elementwise product, difference vector, and cosine similarity scalar.
    """
    def __init__(self, text_dim=768, image_dim=768, proj_dim=256, hidden_dim=128, dropout=0.45):
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
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        
        t_norm = F.normalize(t, p=2, dim=-1)
        i_norm = F.normalize(i, p=2, dim=-1)
        
        inter = t_norm * i_norm
        diff = torch.abs(t_norm - i_norm)
        cos_sim = (t_norm * i_norm).sum(dim=-1, keepdim=True)
        
        fused = torch.cat([t_norm, i_norm, inter, diff, cos_sim], dim=-1)
        h = self.inter_proj(fused)
        return self.classifier(h)


class SOTA_BiDirectionalCrossAttention(nn.Module):
    """
    SOTA Model 3: Bi-directional Cross-Attention Transformer.
    Performs multi-head cross-attention across textual and visual representations with residual connections.
    """
    def __init__(self, text_dim=768, image_dim=768, proj_dim=256, hidden_dim=128, num_heads=8, dropout=0.45):
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
        self.attn_text = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_img = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)
        
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t_p = self.text_proj(text_feat).unsqueeze(1) # (B, 1, proj_dim)
        i_p = self.image_proj(image_feat).unsqueeze(1) # (B, 1, proj_dim)
        
        t_attn, _ = self.attn_text(query=t_p, key=i_p, value=i_p)
        i_attn, _ = self.attn_img(query=i_p, key=t_p, value=t_p)
        
        t_out = self.norm_t(t_p + t_attn).squeeze(1)
        i_out = self.norm_i(i_p + i_attn).squeeze(1)
        
        fused = torch.cat([t_out, i_out], dim=-1)
        h = self.inter_proj(fused)
        return self.classifier(h)
