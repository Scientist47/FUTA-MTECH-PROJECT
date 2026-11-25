"""Attention weights of the CNN-BiLSTM attention-pooling layer
(thesis Section 3.3.5). One weight per time step, shared across leads.
"""


def extract_attention(model, inputs):
    """Return attention weights up-sampled to shape (n, 1000)."""
    raise NotImplementedError
