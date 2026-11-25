"""Segment-by-lead alignment (thesis Equation 3.2).

A_{s,l} = sum over t in segment s of |phi_l(t)|, for five segments
(P, PR, QRS, ST, T) x 12 leads, normalised so the 60 cells sum to one.
Boundaries come from PTB-XL+ fiducial points, with NeuroKit2 as a fallback.
Maps without lead information are applied to every lead.
"""


def segment_boundaries(record_id, signal):
    """Return the P, PR, QRS, ST and T boundaries for one record."""
    raise NotImplementedError


def align_to_grid(attribution, boundaries):
    """Return a normalised 5 x 12 attribution matrix."""
    raise NotImplementedError
