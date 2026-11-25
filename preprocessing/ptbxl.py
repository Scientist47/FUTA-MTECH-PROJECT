"""Load PTB-XL records and labels.

- Read the 100 Hz records (12 x 1,000 samples) with wfdb.
- Map scp_codes to the five diagnostic superclasses (NORM, MI, STTC, CD, HYP)
  using scl_statements.csv supplied with the dataset.
- Keep strat_fold and patient_id for partitioning and leakage checks.
"""


def load_metadata(root):
    """Return the PTB-XL metadata table with superclass labels."""
    raise NotImplementedError


def load_signals(root, metadata):
    """Return an array of shape (n_records, 12, 1000)."""
    raise NotImplementedError


def map_superclasses(scp_codes, statements):
    """Return the multi-hot superclass labels for one record."""
    raise NotImplementedError
