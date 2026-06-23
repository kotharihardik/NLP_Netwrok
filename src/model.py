import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────
# Log-Time Positional Encoding
# Instead of encoding packet index (0,1,2,...),
# we encode actual cumulative time of each packet arrival.
# This gives the Transformer real temporal awareness.
# ──────────────────────────────────────────────────────────────
class LogTimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, ipt_channel):
        """
        x           : [B, seq_len, d_model]
        ipt_channel : [B, seq_len] — normalized inter-packet times
        """
        B, seq_len, d = x.shape

        # Cumulative time for each packet position
        cum_time = torch.cumsum(ipt_channel, dim=1)          # [B, seq_len]
        log_time = torch.log1p(cum_time * 1000.0)            # log scale

        # Sinusoidal encoding based on log time
        div_term = torch.exp(
            torch.arange(0, d, 2, device=x.device).float() *
            (-math.log(10000.0) / d)
        )                                                      # [d/2]

        pe = torch.zeros(B, seq_len, d, device=x.device)
        t  = log_time.unsqueeze(-1)                           # [B, seq_len, 1]
        pe[:, :, 0::2] = torch.sin(t * div_term)
        pe[:, :, 1::2] = torch.cos(t * div_term)

        return x + pe


# ──────────────────────────────────────────────────────────────
# Temporal Transformer Encoder
# Processes per-packet sequence to learn temporal behavior
# ──────────────────────────────────────────────────────────────
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=3, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = LogTimePositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True         # input shape: [B, seq, features]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x : [B, 32, 3]  — [batch, packets, (size, direction, ipt)]
        returns : [B, d_model]
        """
        ipt_channel = x[:, :, 2]           # inter-packet times for positional encoding
        h = self.input_proj(x)              # [B, 32, d_model]
        h = self.pos_enc(h, ipt_channel)   # add log-time positional encoding
        h = self.transformer(h)            # [B, 32, d_model]
        h = self.norm(h)

        # Global average pool over sequence → single flow representation
        return h.mean(dim=1)               # [B, d_model]


# ──────────────────────────────────────────────────────────────
# Statistical MLP Encoder
# Processes flow-level statistics and histogram features
# ──────────────────────────────────────────────────────────────
class StatEncoder(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        """
        x : [B, 36]
        returns : [B, hidden_dim]
        """
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────
# Cross-Modal Fusion
# Temporal features query statistical context via attention.
# This conditions packet-level understanding on network environment.
# ──────────────────────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, temporal_feat, stat_feat):
        """
        temporal_feat : [B, d_model]
        stat_feat     : [B, d_model]
        returns       : [B, d_model]
        """
        # Reshape for attention: [B, 1, d_model]
        q = temporal_feat.unsqueeze(1)
        k = stat_feat.unsqueeze(1)
        v = stat_feat.unsqueeze(1)

        attended, _ = self.attention(q, k, v)   # [B, 1, d_model]
        attended    = attended.squeeze(1)         # [B, d_model]

        # Residual concat fusion
        fused = torch.cat([temporal_feat + attended, stat_feat], dim=1)  # [B, 2*d_model]
        return self.ff(fused)                     # [B, d_model]


# ──────────────────────────────────────────────────────────────
# AEGISFlow — Full Model
# ──────────────────────────────────────────────────────────────
class AEGISFlow(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        d_model       = config["model"]["temporal_hidden"]
        stat_hidden   = config["model"]["stat_hidden"]
        embed_dim     = config["model"]["embedding_dim"]
        dropout       = config["model"]["dropout"]
        nhead         = config["model"]["temporal_heads"]
        num_layers    = config["model"]["temporal_layers"]

        # Encoders
        self.temporal_encoder = TemporalEncoder(
            input_dim=3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        self.stat_encoder = StatEncoder(
            input_dim=36,
            hidden_dim=stat_hidden,
            dropout=dropout
        )

        # Fusion
        self.fusion = CrossModalFusion(d_model=d_model, dropout=dropout)

        # Projection to embedding space (used for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim)
        )

        # Classification head (used for supervised cross-entropy loss)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, temporal_seq, stat_vec):
        """
        temporal_seq : [B, 32, 3]
        stat_vec     : [B, 36]
        returns:
          embedding  : [B, 128]  — L2-normalized for contrastive loss
          logits     : [B, num_classes] — for classification loss
        """
        t_feat    = self.temporal_encoder(temporal_seq)   # [B, 128]
        s_feat    = self.stat_encoder(stat_vec)            # [B, 128]
        fused     = self.fusion(t_feat, s_feat)            # [B, 128]
        embedding = self.projection(fused)                 # [B, 128]

        # L2 normalize embeddings — essential for contrastive learning
        embedding = F.normalize(embedding, dim=1)

        logits = self.classifier(embedding)                # [B, num_classes]

        return embedding, logits


if __name__ == "__main__":
    config = load_config()

    model = AEGISFlow(num_classes=92, config=config)
    model.train()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AEGISFlow total parameters: {total_params:,}")

    B = 256

    # Realistic dummy data matching actual feature ranges
    temporal_dummy        = torch.zeros(B, 32, 3)
    temporal_dummy[:,:,0] = torch.rand(B, 32)          # packet sizes [0,1]
    temporal_dummy[:,:,1] = torch.randint(0,2,(B,32)).float() * 2 - 1  # direction -1 or +1
    temporal_dummy[:,:,2] = torch.rand(B, 32) * 0.1    # IPT [0, 0.1] — realistic

    stat_dummy = torch.rand(B, 36)                      # stats all [0,1]

    # Debug each stage
    t_feat = model.temporal_encoder(temporal_dummy)
    print(f"temporal_encoder — nan: {t_feat.isnan().any()}, mean: {t_feat.mean().item():.4f}")

    s_feat = model.stat_encoder(stat_dummy)
    print(f"stat_encoder     — nan: {s_feat.isnan().any()}, mean: {s_feat.mean().item():.4f}")

    fused = model.fusion(t_feat, s_feat)
    print(f"fusion           — nan: {fused.isnan().any()}, mean: {fused.mean().item():.4f}")

    proj = model.projection(fused)
    print(f"projection       — nan: {proj.isnan().any()}, mean: {proj.mean().item():.4f}")

    embedding, logits = model(temporal_dummy, stat_dummy)
    print(f"\nEmbedding shape : {embedding.shape}")
    print(f"Logits shape    : {logits.shape}")
    print(f"Embedding norm  : {embedding.norm(dim=1).mean().item():.4f}  (should be ~1.0)")
    print("\nModel architecture ready.")