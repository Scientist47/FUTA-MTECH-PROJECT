"""Clinical plausibility analysis (thesis Section 3.5.4, RQ3 and H3).

Delta = mean rating of real maps - mean rating of control maps (Equation 3.4),
with a 95% confidence interval; Jaccard overlap of rater-marked cells with
the model's top-k cells; inter-rater agreement (weighted Cohen's kappa or
Krippendorff's alpha).

Rater data are collected only after HREC approval and are never committed.
"""


def rating_difference(ratings):
    """Return Delta and its 95% confidence interval."""
    raise NotImplementedError


def jaccard_overlap(rater_cells, model_grid, k):
    """Return the Jaccard index for one case."""
    raise NotImplementedError


def inter_rater_agreement(ratings):
    """Return weighted kappa or Krippendorff's alpha."""
    raise NotImplementedError
