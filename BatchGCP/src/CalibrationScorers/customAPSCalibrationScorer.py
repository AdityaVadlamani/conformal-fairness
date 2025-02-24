import numpy as np
import torch
from CalibrationScorers import calibrationScorer, predictionSet


class customAPSCalibrationPredictionSet(predictionSet.PredictionSet):
    def __init__(self, pred_set):
        self.pred_set = pred_set

    def cover(self, y):
        return y in self.pred_set

    def get_size(self):
        return len(self.pred_set)


class customAPSCalibrationScorer(calibrationScorer.CalibrationScorer):
    def __init__(self):
        self.f_pred = None

    def calc_score(self, x, y=None):
        # a vectorized implementation of APS score from
        # . https://github.com/soroushzargar/DAPS/blob/main/torch-conformal/gnn_cp/cp/transformations.py
        # sorted probs: n_samples x n_classes
        probs = torch.from_numpy(self.f_pred(x))
        probs_pi_rev_indices = torch.argsort(probs, dim=1, descending=True)
        sorted_probs_pi = torch.take_along_dim(probs, probs_pi_rev_indices, dim=1)
        # PI[i, j] = sum(pi_(1) + pi_(2) + ... + pi_(j-1))
        # PI[i, 0] = 0
        PI = torch.zeros(
            (sorted_probs_pi.shape[0], sorted_probs_pi.shape[1] + 1),
            device=probs.device,
        )
        PI[:, 1:] = torch.cumsum(sorted_probs_pi, dim=1)
        # we vectorize this loop
        # ranks = torch.zeros((n_samples, n_classes), dtype=torch.int32)
        # for i in range(n_samples):
        #    ranks[i, sorted_order[i]] = torch.arange(n_classes -1, -1, -1)
        ranks = probs_pi_rev_indices.argsort(dim=1)

        # cumulative score up to rank j
        # cls_scores[i, j] = NC score for class j for sample i
        # that is assuming that the true class is j
        # cls_score[i, j] = PI[i, rank[j]] + (1 - u) * probs[i, j]
        # note that PI starts at 0, so PI[i, rank[j]] = sum(probs[:rank[j] - 1])

        # whether to use uniform noise to adjust set size
        u_vec = torch.rand(
            probs.shape[0], 1, device=probs.device
        )  # u_vec[i, 0] = u for sample i
        cls_scores = PI.gather(1, ranks) + (1 - u_vec) * probs

        cls_scores = torch.min(cls_scores, torch.ones_like(cls_scores)).numpy()

        if y is not None:
            return cls_scores[y]
        return cls_scores

    def get_prediction_set(self, scores, calibration_threshold):
        pass

    def update(self, f_pred):
        self.f_pred = f_pred
