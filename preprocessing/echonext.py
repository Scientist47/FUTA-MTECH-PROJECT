"""Load EchoNext records and the binary structural heart disease (SHD) label.

EchoNext is used under PhysioNet credentialed access. Never commit its data.
Records are resampled to 100 Hz where necessary and use the predefined splits.
"""


def load_split(root, split):
    """Return signals and SHD labels for 'train', 'val' or 'test'."""
    raise NotImplementedError
