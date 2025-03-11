import gc
import math
from abc import ABC

import torch
from torch.masked import masked_tensor

from ..config import NeighborhoodConfig, PrimitiveScoreConfig
from ..constants import DEFAULT_DEVICE, Stage
from ..data import BaseDataModule


class CPScore(ABC):
    def __init__(self, **kwargs):
        self.defined_args = kwargs
        self.alpha = kwargs.get("alpha", None)

    def pipe_compute(self, probs):
        return self.compute(probs, **self.defined_args)

    def compute(self, probs, **kwargs):
        return probs

    def compute_quantile(self, scores, alpha=None, **kwargs):
        if self.alpha is not None:
            alpha = self.alpha
        else:
            assert alpha is not None, f"Missing alpha value for quantile computation"

        n = scores.shape[0]
        return torch.quantile(
            scores, min(1, math.ceil((n + 1) * (1 - alpha)) / n), interpolation="higher"
        )

    def compute_group_quantiles(self, scores, groups, alpha=None, **kwargs):
        unique_groups = torch.unique(groups)
        quantiles = []
        for g in unique_groups:
            mask = groups == g
            quantiles.append(self.compute_quantile(scores[mask], alpha, **kwargs))
        return torch.as_tensor(quantiles)


class TPSScore(CPScore):
    def __init__(self, config: PrimitiveScoreConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def compute(self, probs, **kwargs):
        return 1 - probs

    def compute_quantile(self, scores, alpha=None, **kwargs):
        if not self.config.use_tps_classwise:
            return super().compute_quantile(scores, alpha)

        # Use Classwise Conformal Predictions
        labels = kwargs.get("labels", None)
        num_classes = kwargs.get("num_classes", None)

        assert labels is not None, f"Missing labels for class-wise TPS Quantiles"
        assert (
            num_classes is not None
        ), f"Missing number of classes for class-wise TPS Quantiles"

        quantiles = torch.zeros(num_classes)
        for i in range(num_classes):
            # Class Based Quantile
            class_i_mask = labels == i
            if not class_i_mask.any():
                # If class not seen, then by exchangeability we
                # assume that it does not exist, thus we never predict it
                quantiles[i] = -1
            else:
                quantiles[i] = super().compute_quantile(scores[class_i_mask], alpha)

        return quantiles


class APSScore(CPScore):
    def __init__(self, config: PrimitiveScoreConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def compute(self, probs, **kwargs):
        # a vectorized implementation of APS score from
        # . https://github.com/soroushzargar/DAPS/blob/main/torch-conformal/gnn_cp/cp/transformations.py
        # sorted probs: n_samples x n_classes

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
        if self.config.use_aps_epsilon:
            # whether to use uniform noise to adjust set size
            u_vec = torch.rand(
                probs.shape[0], 1, device=probs.device
            )  # u_vec[i, 0] = u for sample i
            cls_scores = PI.gather(1, ranks) + (1 - u_vec) * probs
        else:
            cls_scores = PI.gather(1, ranks) + 1 * probs
        cls_scores = torch.min(cls_scores, torch.ones_like(cls_scores))
        return cls_scores


class NAPSScore(APSScore):
    def __init__(
        self, config: NeighborhoodConfig, datamodule: BaseDataModule, **kwargs
    ):
        super().__init__(
            PrimitiveScoreConfig(use_aps_epsilon=config.use_aps_epsilon), **kwargs
        )
        self.config = config
        self.datamodule = datamodule

    def compute_quantile(self, scores, alpha=None, **kwargs):
        if self.alpha is not None:
            alpha = self.alpha
        else:
            assert alpha is not None, f"Missing alpha value for quantile computation"

        # i.e based on batch size, k_hop input, and avg neighbors per node
        num_batches = self.config.num_batches

        # n = scores.shape[0]
        weight_function = self.config.weight_function
        k_hop = self.config.k_hop_neighborhood
        split_dict = self.datamodule.split_dict
        num_test_nodes = split_dict[Stage.TEST].shape[0]

        # Get K-Hop (This might also be function dependent)
        valid_k_hop = k_hop  # min with graph radius (maybe)

        # Get Weight Function information:
        function = self.__uniform
        if weight_function == "hyperbolic":
            function = self.__hyperbolic
        elif weight_function == "exponential":
            function = self.__exponential

        quantiles = torch.zeros(num_test_nodes).view((-1, 1))

        start_index = 0
        batch_size = int(num_test_nodes / num_batches)
        end_index = min(start_index + batch_size, num_test_nodes)

        # Runs Up To n+1 Batches
        while start_index < num_test_nodes:
            # Get the minimum distance from each test node to nodes in the calibration set, upto k hops
            k_hop_neighborhood = self.__k_hop_neighbors(
                valid_k_hop, start_index, end_index
            )

            if k_hop_neighborhood._nnz():
                # Compute The weights for the point masses, based on the specified function
                weights = function(neighbors=k_hop_neighborhood)

                # Force garbage collection on GPU
                del k_hop_neighborhood

                # get non-zero columns, can make quantile function more memory efficient if
                # columns with all zero weights are eliminated
                non_zero_col = torch.unique(weights.coalesce().indices()[1, :]).to(
                    DEFAULT_DEVICE
                )
                weights = torch.index_select(weights, 1, non_zero_col)

                # Get the quantiles for each test node in this batch
                quantiles[start_index:end_index, :] = self.__get_quantile(
                    scores[non_zero_col.to("cpu")],
                    alpha,
                    weights.to_dense().to("cpu"),
                )
                del non_zero_col
            else:
                # Get the quantiles for each test node in this batch
                quantiles[start_index:end_index, :] = float("inf")

            start_index += batch_size
            end_index = min(start_index + batch_size, num_test_nodes)

            gc.collect()

        return quantiles

    def __uniform(self, neighbors):
        # If positve will be 1, 0 for 0, and no negative value in tensor
        return neighbors.sign().to(dtype=torch.float32)

    def __hyperbolic(self, neighbors):
        # 1/k for each non-zero node in matrix
        coalesce = neighbors.coalesce()
        weights = torch.sparse.LongTensor(
            coalesce.indices(),
            torch.pow(coalesce.values().float(), -1).to(DEFAULT_DEVICE),
            coalesce.size(),
        )
        return weights

    def __exponential(self, neighbors):
        # Calculates e^(ln(2)x) -1 = 2^x -1
        weights = torch.special.expm1(torch.log(torch.tensor(2)) * (-neighbors)).to(
            dtype=torch.float32
        )
        # Add one to only the non-zero terms to get 2^x -1
        return weights + self.__uniform(neighbors)

    def __k_hop_neighbors(self, k_hop, start_index, end_index):
        split_dict = self.datamodule.split_dict

        # Create the Adjacency Matrix
        A = self.datamodule.adj_matrix.to(DEFAULT_DEVICE)

        # Batch of Test_Nodes
        # Get all calib nodes, and get test nodes in batch
        test_nodes = (split_dict[Stage.TEST].to(DEFAULT_DEVICE))[start_index:end_index]
        calib_nodes = split_dict[Stage.CALIBRATION].to(DEFAULT_DEVICE)

        n_hop = torch.index_select(A, 0, test_nodes)  # (size of batch) x (Num nodes)
        # Matrix - (size of batch) x (Num calib nodes), get 1 - hop neighbors for each of the test nodes in the batch
        k_hop_neighborhood = torch.index_select(n_hop, 1, calib_nodes).to(
            dtype=torch.int8
        )
        for n in range(2, k_hop + 1):
            # Get the nodes that are within n hops
            n_hop = n_hop.matmul(A)

            # n_hop_tensor values = -1 correspond to nodes that are at a minimum n_hop away from the test node
            # All other nodes are non-negative, so signbit will only be true for nodes exactly n_hop away
            n_hop_tensor = (
                k_hop_neighborhood - torch.index_select(n_hop, 1, calib_nodes).sign()
            )
            n_hop_tensor = n_hop_tensor.signbit() * n

            # Add the nodes that are n_hops away to the tensor containing neighbor information
            k_hop_neighborhood.add_(n_hop_tensor)

            del n_hop_tensor

        test_nodes.to("cpu")
        calib_nodes.to("cpu")

        del A
        del n_hop
        return k_hop_neighborhood

    def __get_quantile(
        self,
        calib_scores,
        alpha,
        weights,
    ):
        # Normalize the weights for each row, such that the weights in
        # each row sum to 1. The + 1 corresponds to the pointmass at ininity
        sum_weights = torch.sum(weights, dim=1).view((-1, 1)) + 1
        norm_weights = weights / sum_weights

        # Sorted calibration scores in ascending order, and apply the
        # same sorting to the weights to ensure that each score still corresponds to its weight
        dist_indexes = torch.argsort(calib_scores)
        calib_scores = calib_scores[dist_indexes]
        norm_weights = norm_weights[:, dist_indexes]

        # Take the cumulative weights to create a cdf (think y-axis of cdf plot)
        cum_weights = torch.cumsum(norm_weights, dim=1)

        # Get the first index in each row greater than 1-alpha, this index correponds to the
        # 1-alpha quantile of a given test node
        mask = cum_weights >= (1 - alpha)

        quant_indexes = torch.argmin(masked_tensor(cum_weights, mask), dim=1).to_tensor(
            value=-1
        )

        # If the index is -1, it means that the 1-alpha quantile is infinity, otherwise the quantile
        # corresponds to the given index in the calibration scores.
        quantiles = torch.full((weights.shape[0],), float("inf"))
        index_mask = quant_indexes >= 0
        quantiles[index_mask] = torch.gather(
            calib_scores, 0, quant_indexes[index_mask].to(dtype=torch.long)
        )
        return quantiles.view((-1, 1))
