import numpy as np
from voa_utils import load_simulation_grasp_scores, uniform_belief, expected_grasp_score,\
    expected_grasps_score_multiple_grasps, max_expected_grasp_score, belief_update


if __name__ == "__main__":
    # load simulation grasp scores, TODO: make it configurable with object/simulation-real/noise
    scores = load_simulation_grasp_scores('expo', '0.01')

    # start with uniform belief
    initial_belief = uniform_belief(scores.shape[1])

    # first part of VOA: the maximum expected score with new belief, expectation over poses distrbuted over belief:
    observation_probabilities = None  # TODO
    updated_belief = belief_update(initial_belief, observation_probabilities) # includes the expectation over b?

