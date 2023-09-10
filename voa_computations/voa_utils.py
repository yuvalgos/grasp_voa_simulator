import numpy as np
import pandas as pd


def load_simulation_grasp_scores(object_name: str, noise_std: str):
    """
    Load grasp scores from csv file.
    :param object_name: name of the object
    :param noise_std: standard deviation of the noise probably '0.00', '0.01', '0.03' or '0.05'
    :return: grasp scores as np array with size (n_grasps, n_poses)
    """
    scores_df = pd.read_csv(f'./data/{object_name}_simulation/grasp_score_{noise_std}.csv', index_col=0)
    scores = scores_df.values
    return scores


def uniform_belief(n_poses: int):
    return np.ones(n_poses) / n_poses


def expected_grasp_score(grasp_scores, pose_belief):
    """
    Compute expected grasp score given a pose belief and score for a grasp for each pose
    :param grasp_scores: grasp scores for each pose
    :param pose_belief: belief over poses
    """
    assert grasp_scores.shape == pose_belief.shape
    assert sum(pose_belief) - 1 < 1e-6
    return grasp_scores.dot(pose_belief)


def expected_grasps_score_multiple_grasps(grasp_scores_table, pose_belief):
    """
    Compute expected grasp score given a pose belief and score for a grasp for each pose for multiple grasps
    :param grasp_scores_table: grasp scores for each grasp and pose (grasp are the first dimention)
    :param pose_belief: belief over poses
    return: expected grasp score for each grasp
    """
    assert grasp_scores_table.shape[1] == pose_belief.shape[0]
    assert sum(pose_belief) - 1 < 1e-6
    return grasp_scores_table @ pose_belief.T


def max_expected_grasp_score(grasp_scores_table, pose_belief):
    """
    compute the max expected grasp score given a pose belief and a table of grasp scores for each pose and grasp
    :param grasp_scores_table: grasp scores for each grasp and pose (grasp are the first dimention)
    :param pose_belief: belief over poses
    return: max expected grasp score and the grasp index (0 is the first grasp)
    """
    scores = expected_grasps_score_multiple_grasps(grasp_scores_table, pose_belief)
    return np.max(scores), np.argmax(scores)


scores = load_simulation_grasp_scores('expo', '0.01')
# print( expected_grasp_score(scores[0], uniform_belief(scores.shape[1])) )
print( max_expected_grasp_score(scores, uniform_belief(scores.shape[1])) )

pass