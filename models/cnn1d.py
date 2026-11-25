"""Architecture 1: residual one-dimensional CNN (thesis Section 3.3.2).

Input 12 x 1,000; residual convolutional blocks with batch normalisation and
pooling; global pooling; fully connected head with dropout; one output per
class (five for PTB-XL, one for EchoNext). Explained with SHAP and Grad-CAM.

The network returns logits (values before the sigmoid). Apply torch.sigmoid
to the output to get probabilities; the loss in models/losses.py applies the
sigmoid internally.
"""
import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    """Two convolutions with batch normalisation and a shortcut connection."""

    def __init__(self, c_in, c_out, kernel_size=7, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1d(nn.Module):
    def __init__(self, n_leads=12, n_outputs=5, widths=(64, 128, 128, 256), dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 15, 2, 7, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks += [ResBlock1d(c, w, stride=stride), ResBlock1d(w, w)]
            c = w
        self.features = nn.Sequential(*blocks)   # last conv stage = Grad-CAM layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_outputs))

    def forward(self, x):
        return self.head(self.pool(self.features(self.stem(x))))

    @property
    def gradcam_layer(self):
        """The last convolutional block, used by xai/gradcam.py."""
        return self.features[-1]


def build_cnn1d(n_leads=12, n_outputs=5):
    """Return the residual 1D-CNN as a torch.nn.Module."""
    return ResNet1d(n_leads=n_leads, n_outputs=n_outputs)
