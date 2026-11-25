"""Model architectures, loss and interpretable comparator (Objective a, thesis Section 3.3)."""
from .cnn1d import build_cnn1d
from .cnn_bilstm_attention import build_cnn_bilstm_attention


def build_model(architecture, n_leads=12, n_outputs=5):
    """Build 'cnn1d' or 'cnn_bilstm_attention' by name."""
    if architecture == "cnn1d":
        return build_cnn1d(n_leads, n_outputs)
    if architecture == "cnn_bilstm_attention":
        return build_cnn_bilstm_attention(n_leads, n_outputs)
    raise ValueError(f"Unknown architecture: {architecture}")
