import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_min, scatter_max


def graph_minmax(x, batch):
    """
    Normalize node features per graph to [0, 1].
    
    Args:
        x (Tensor): Node features [N, D]
        batch (LongTensor): Graph index per node [N]
    
    Returns:
        Tensor: Normalized features [N, D]
    """
    x_min = scatter_min(x, batch, dim=0)[0]  # [B, D]
    x_max = scatter_max(x, batch, dim=0)[0]  # [B, D]
    scale = (x_max - x_min).clamp(min=1e-6)
    return (x - x_min[batch]) / scale[batch]


class RPEBuilder(nn.Module):
    """
    Relative Positional Encoding using Laplacian eigenvectors and sinusoids.
    Input: delta = lap_pos[u] - lap_pos[v] (per-graph normalized)
    Output: [E, n_heads] bias added to attention logits
    """
    def __init__(self, lap_dim, num_freqs=16, n_heads=8):
        super().__init__()
        self.num_freqs = num_freqs
        input_dim = lap_dim * (1 + 2 * num_freqs)
        self.rpe_proj = nn.Linear(input_dim, n_heads)

        # Small init + rescaling to std=1.0, bias=0 (safe)
        nn.init.xavier_uniform_(self.rpe_proj.weight, gain=0.1)
        nn.init.zeros_(self.rpe_proj.bias)

    def sinusoid(self, delta):
        """Sinusoidal encoding of relative Laplacian distance."""
        freqs = torch.arange(self.num_freqs, device=delta.device).float()
        freqs = 1.0 / (10000 ** (freqs / self.num_freqs))
        angles = delta.unsqueeze(-1) * freqs
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        return torch.cat([sin_enc, cos_enc], dim=-1).reshape(delta.size(0), -1)

    def forward(self, edge_index, lap_pos):
        u, v = edge_index
        delta = lap_pos[u] - lap_pos[v]  # [E, lap_dim], normalized per graph
        enc = self.sinusoid(delta)
        rpe_input = torch.cat([delta, enc], dim=-1)
        return self.rpe_proj(rpe_input)  # [E, n_heads]


class GraphInputEmbedding(nn.Module):
    """Embed node features and normalized Laplacian positions."""
    def __init__(self, d_node, d_model, lap_dim, dropout=0.0):
        super().__init__()
        self.node_proj = nn.Linear(d_node, d_model)
        self.lap_proj = nn.Linear(lap_dim, d_model)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.lap_proj.weight)
        nn.init.zeros_(self.node_proj.bias)
        nn.init.zeros_(self.lap_proj.bias)

    def forward(self, x, lap_pos, batch_nodes):
        """
        Args:
            x: Node features [N, d_node]
            lap_pos: Laplacian positions [N, lap_dim]
            batch_nodes: Graph index per node [N]
        """
        lap_pos = graph_minmax(lap_pos, batch_nodes)  # per-graph [0,1]
        h = self.node_proj(x) + self.lap_proj(lap_pos)
        return self.drop(h)


class ChemGraphormerAttention(nn.Module):
    """
    Graph attention with:
    - Scaled dot-product 
    - Edge type bias
    - Laplacian RPE
    - Gated message passing
    """
    def __init__(self, d_model, n_heads, d_edge, lap_dim,
                 num_freqs=16, dropout=0.0, bias_std=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.scale = math.sqrt(self.dk)

        # QKV
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Edge
        self.edge_bias = nn.Linear(d_edge, n_heads)
        self.edge_msg = nn.Linear(d_edge, self.dk)
        self.edge_gate = nn.Linear(d_edge, n_heads)

        # RPE
        self.rpe = RPEBuilder(lap_dim, num_freqs, n_heads)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_msg = nn.Dropout(dropout)

        # === INITIALIZATION ===
        for lin in (self.Wq, self.Wk, self.Wv):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        nn.init.normal_(self.edge_bias.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.edge_bias.bias)

        nn.init.xavier_uniform_(self.edge_msg.weight)
        nn.init.zeros_(self.edge_msg.bias)

        nn.init.xavier_uniform_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, 0)  

        # Safe rescaling: skip constants
        self.rescale_bias(self.edge_bias.weight, bias_std)
        self.rescale_bias(self.edge_bias.bias, bias_std)
        self.rescale_bias(self.rpe.rpe_proj.weight, bias_std)
        self.rescale_bias(self.rpe.rpe_proj.bias, bias_std)

    @staticmethod
    @torch.no_grad()
    def rescale_bias(param, target_std=1.0):
        """Rescale parameter to target std, skip if constant."""
        if param is None:
            return
        std = param.std(unbiased=False)
        if std < 1e-6:
            return
        param.data *= target_std / (std + 1e-6)

    def forward(self, h, edge_index, edge_attr, lap_pos, batch_nodes):
        u, v = edge_index
        N = h.size(0)

        Q = self.Wq(h).view(N, self.n_heads, self.dk)
        K = self.Wk(h).view(N, self.n_heads, self.dk)
        V = self.Wv(h).view(N, self.n_heads, self.dk)

        # Scaled dot-product
        logits = (Q[u] * K[v]).sum(-1) / self.scale
        logits = logits + self.edge_bias(edge_attr)  
        logits = logits + self.rpe(edge_index, lap_pos)

        # Attention weights
        attn = scatter_softmax(logits, v, dim=0)                   
        attn = self.drop_attn(attn)

        # Gated message
        msg = V[u] + self.edge_msg(edge_attr).unsqueeze(1)             
        gate = torch.sigmoid(self.edge_gate(edge_attr))             
        msg = self.drop_msg(msg * gate.unsqueeze(-1))

        out = scatter_add(attn.unsqueeze(-1) * msg, v, dim=0, dim_size=N)
        return out.reshape(N, -1)


class EncoderBlock(nn.Module):
    """Transformer encoder block with pre-norm and residual connections."""
    def __init__(self, d_model, n_heads, d_edge, lap_dim,
                 d_ff=1024, num_freqs=16, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ChemGraphormerAttention(
            d_model, n_heads, d_edge, lap_dim, num_freqs, dropout
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

        # FFN init
        nn.init.xavier_uniform_(self.ff[0].weight)
        nn.init.xavier_uniform_(self.ff[3].weight)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.zeros_(self.ff[3].bias)

    def forward(self, h, edge_index, edge_attr, lap_pos, batch_nodes):
        h = h + self.drop1(self.attn(self.norm1(h), edge_index, edge_attr, lap_pos, batch_nodes))
        h = h + self.drop2(self.ff(self.norm2(h)))
        return h


class ChemGraphormerEncoder(nn.Module):
    """Stack of encoder blocks with input embedding and final norm."""
    def __init__(self, d_node, d_edge, lap_dim,
                 d_model=256, n_heads=8, num_layers=6,
                 d_ff=1024, num_freqs=16, dropout=0.2):
        super().__init__()
        self.embed = GraphInputEmbedding(d_node, d_model, lap_dim, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_edge, lap_dim,
                         d_ff, num_freqs, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch):
        h = self.embed(batch.x, batch.lap_pos, batch.batch)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_attr,
                      batch.lap_pos, batch.batch)
        return self.norm(h)


class AttentionPooling(nn.Module):
    """Learnable graph-level pooling via attention scores."""
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, h, batch_nodes):
        scores = self.attn(h).squeeze(-1)
        scores = torch.exp(scores - scores.max())
        denom = scatter_add(scores, batch_nodes, dim=0).clamp(min=1e-6)
        pooled = scatter_add(h * scores.unsqueeze(-1), batch_nodes, dim=0)
        return pooled / denom.unsqueeze(-1)


class ChemGraphormerClassifier(nn.Module):
    """Final classifier: encoder >> pooling >> MLP head."""
    def __init__(self, d_node, d_edge, lap_dim, num_classes,
                 d_model=256, n_heads=8, num_layers=6,
                 d_ff=1024, num_freqs=16, dropout=0.2):
        super().__init__()
        self.encoder = ChemGraphormerEncoder(
            d_node, d_edge, lap_dim,
            d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, num_freqs=num_freqs, dropout=dropout
        )
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, batch):
        h = self.encoder(batch)
        g = self.pool(h, batch.batch)
        return self.head(g)