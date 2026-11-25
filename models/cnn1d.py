"""Architecture 1: residual one-dimensional CNN (thesis Section 3.3.2).

Input 12 x 1,000; residual convolutional blocks with batch normalisation and
pooling; global pooling; fully connected head with dropout; sigmoid outputs
(five for PTB-XL, one for EchoNext). Explained with SHAP and Grad-CAM.
"""


def build_cnn1d(n_leads=12, n_outputs=5):
    """Return the residual 1D-CNN as a torch.nn.Module."""
    raise NotImplementedError
