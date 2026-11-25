"""Architecture 2: CNN-BiLSTM with attention pooling (thesis Section 3.3.3).

Convolutional front-end, bidirectional LSTM over time, attention pooling with
one weight per time step, one output per class (logits). Explained with SHAP,
Grad-CAM (on the convolutional stage) and its attention weights.
"""
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Additive attention: one weight per time step, weights sum to one."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.last_weights = None

    def forward(self, h):                     # h: (batch, time, dim)
        a = torch.softmax(self.score(h).squeeze(-1), dim=1)   # (batch, time)
        self.last_weights = a.detach()
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_leads=12, n_outputs=5, conv_channels=(64, 128, 128),
                 lstm_hidden=128, dropout=0.5, use_attention=True):
        super().__init__()
        layers, c = [], n_leads
        for w in conv_channels:
            layers += [nn.Conv1d(c, w, 7, padding=3, bias=False), nn.BatchNorm1d(w),
                       nn.ReLU(), nn.MaxPool1d(2)]
            c = w
        self.conv = nn.Sequential(*layers)    # 1000 -> 125 time steps
        self.lstm = nn.LSTM(c, lstm_hidden, batch_first=True, bidirectional=True)
        self.use_attention = use_attention
        self.attention = AttentionPool(2 * lstm_hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * lstm_hidden, n_outputs))

    def forward(self, x, return_attention=False):
        f = self.conv(x)                       # (batch, channels, time)
        h, _ = self.lstm(f.transpose(1, 2))    # (batch, time, 2*hidden)
        if self.use_attention:
            pooled, a = self.attention(h)
        else:                                  # ablation: plain mean pooling
            pooled, a = h.mean(dim=1), None
        out = self.head(pooled)
        return (out, a) if return_attention else out

    @property
    def gradcam_layer(self):
        """Last convolutional stage (its final ReLU), used by xai/gradcam.py."""
        return self.conv[-2]


def build_cnn_bilstm_attention(n_leads=12, n_outputs=5, use_attention=True):
    """Return the CNN-BiLSTM with attention as a torch.nn.Module."""
    return CNNBiLSTMAttention(n_leads=n_leads, n_outputs=n_outputs, use_attention=use_attention)
