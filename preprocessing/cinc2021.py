"""Load the PhysioNet/CinC 2021 collection for external validation.

- Exclude the PTB and PTB-XL sources.
- Map SNOMED CT codes to the five superclasses.
- Test only superclasses with at least 100 positive records.
"""


def load_external(root, exclude_sources=('PTB', 'PTB-XL')):
    """Return signals and superclass labels for external validation."""
    raise NotImplementedError


def map_snomed_to_superclass(codes):
    """Return the multi-hot superclass labels for a list of SNOMED CT codes."""
    raise NotImplementedError
