"""Architecture 2: CNN-BiLSTM with attention pooling (thesis Section 3.3.3).

Convolutional front-end, bidirectional LSTM over time, attention pooling with
one weight per time step, sigmoid outputs. Explained with SHAP, Grad-CAM and
its attention weights.
"""


def build_cnn_bilstm_attention(n_leads=12, n_outputs=5):
    """Return the CNN-BiLSTM with attention as a torch.nn.Module."""
    raise NotImplementedError
