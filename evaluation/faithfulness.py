"""Faithfulness of the explanations (thesis Section 3.5.3, RQ2 and H2).

- Deletion and insertion curves over segment-by-lead cells (Equation 3.3),
  with removed cells replaced by zero (the training mean after normalisation).
- Randomisation checks (layer re-initialisation, shuffled labels).
- Stability under small perturbations.
- Cross-method agreement (Spearman correlation, top-k overlap).
- Known-answer tests (QRS-duration regression, planted waveform).
All compared against a random-attribution baseline, on 500 test records.
"""


def deletion_auc(model, inputs, grid, target_class):
    """Return the area under the deletion curve."""
    raise NotImplementedError


def insertion_auc(model, inputs, grid, target_class):
    """Return the area under the insertion curve."""
    raise NotImplementedError


def randomisation_check(model, explainer, inputs):
    """Return similarity between original and randomised-model maps."""
    raise NotImplementedError


def stability(model, explainer, inputs):
    """Return similarity of maps under small perturbations."""
    raise NotImplementedError


def method_agreement(grid_a, grid_b, k=10):
    """Return Spearman correlation and top-k overlap."""
    raise NotImplementedError


def known_answer_tests(explainer):
    """Run the QRS-duration and planted-waveform tests."""
    raise NotImplementedError
