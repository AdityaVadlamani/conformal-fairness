from abc import ABC
from typing import Optional, Tuple, Union

import torch

from .transformations import PredSetTransformation


class CRCLoss(ABC):
    def __init__(self, **kwargs):
        self.defined_args = kwargs

    def pipe_compute(self, scores, labels):
        return self.compute(scores, labels, **self.defined_args)

    def compute(
        self, scores, labels, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return torch.ones_like(labels, dtype=torch.float)

    def evaluate(
        self, pred_sets, labels, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return torch.ones_like(labels, dtype=torch.float)


class IncorrectLabelCoverageLoss(CRCLoss):
    """
    Incorrect Label Loss is concerned with the incorrect label coverage as the risk to control
    """

    def compute(self, scores, labels, lambdas, pos_label, **kwargs):
        losses = torch.empty((scores.shape[0], len(lambdas)), dtype=torch.float)
        for i, lmbda in enumerate(lambdas):
            pred_sets = PredSetTransformation(threshold=lmbda).pipe_transform(scores)
            pred_sets[:, pos_label] = (
                False  # Setting correct labels to false to represent removing it from prediction set
            )
            losses[:, i] = 1 - ((pred_sets.sum(dim=-1) > 0).to(torch.float))

        return losses

    def evaluate(self, pred_sets, labels, pos_label, **kwargs):
        # n_test x n_classes
        pred_sets[:, pos_label] = (
            False  # Setting correct labels to false to represent removing it from prediction set
        )
        return 1 - ((pred_sets.sum(dim=-1) > 0).to(float))


class CorrectLabelMiscoverageLoss(CRCLoss):
    """
    Correct Label Loss is concerned with the correct miscoverage as the risk to control.
    """

    def compute(self, scores, labels, lambdas, pos_label, **kwargs):
        losses = torch.empty((scores.shape[0], len(lambdas)), dtype=torch.float)
        for i, lmbda in enumerate(lambdas):
            pred_sets = PredSetTransformation(threshold=lmbda).pipe_transform(scores)

            losses[:, i] = pred_sets[:, pos_label] == False

        return losses

    def evaluate(self, pred_sets, labels, pos_label, **kwargs):
        # n_test x n_classes
        return (pred_sets[:, pos_label] == False).to(torch.float)


class DemographicParityLoss(CRCLoss):
    def compute(
        self,
        scores,
        labels,
        lambdas,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = torch.empty((scores.shape[0], len(lambdas)), dtype=torch.float)
        for i, lmbda in enumerate(lambdas):
            pred_sets = PredSetTransformation(threshold=lmbda).pipe_transform(scores)

            losses[:, i] = 1 - (pred_sets.sum(dim=-1) / num_classes)

        losses[groups != group_id, :] = 0
        return losses

    def evaluate(
        self,
        pred_sets,
        labels,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = 1 - (pred_sets.sum(dim=-1) / num_classes)
        losses[groups != group_id] = 0
        return losses


class EqualOpportunityLoss(CRCLoss):
    def compute(
        self,
        scores,
        labels,
        lambdas,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = torch.empty((scores.shape[0], len(lambdas)), dtype=torch.float)
        for i, lmbda in enumerate(lambdas):
            pred_sets = PredSetTransformation(threshold=lmbda).pipe_transform(scores)
            losses[:, i] = pred_sets[:, pos_label] == False

        losses[labels != pos_label, :] = 0
        losses[groups != group_id, :] = 0
        return losses

    def evaluate(
        self,
        pred_sets,
        labels,
        groups,
        pos_label,
        group_id,
        num_classes=None,
        **kwargs,
    ):
        losses = (pred_sets[:, pos_label] == False).to(torch.float)
        losses[labels != pos_label] = 0
        losses[groups != group_id] = 0
        return losses


class PredictiveEqualityLoss(CRCLoss):
    def compute(
        self,
        scores,
        labels,
        lambdas,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = torch.empty((scores.shape[0], len(lambdas)), dtype=torch.float)
        for i, lmbda in enumerate(lambdas):
            pred_sets = PredSetTransformation(threshold=lmbda).pipe_transform(scores)
            losses[:, i] = (
                (num_classes - 1) - pred_sets.sum(dim=-1) + pred_sets[:, pos_label]
            ) / (num_classes - 1)
        losses[labels == pos_label, :] = 0
        losses[groups != group_id, :] = 0
        return losses

    def evaluate(
        self,
        pred_sets,
        labels,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = (num_classes - 1 - pred_sets.sum(dim=-1) + pred_sets[:, pos_label]) / (
            num_classes - 1
        )
        losses[labels != pos_label] = 0
        losses[groups != group_id] = 0
        return losses


class EqualizedOddsLoss(CRCLoss):
    def __init__(
        self,
        base_loss_module: Optional[Union[CRCLoss, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.equal_opp_loss_fn = EqualOpportunityLoss(
            base_loss_module=base_loss_module, **kwargs
        )
        self.pred_equal_loss_fn = PredictiveEqualityLoss(
            base_loss_module=base_loss_module, **kwargs
        )

        self.equal_opp_losses = None
        self.pred_equal_losses = None

    def compute(
        self,
        scores,
        labels,
        lambdas,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        if self.equal_opp_losses is None or kwargs.get("force_recompute", False):
            self.equal_opp_losses = self.equal_opp_loss_fn.compute(
                scores,
                labels,
                lambdas,
                groups,
                pos_label,
                group_id,
                num_classes,
                **kwargs,
            )

        if self.pred_equal_losses is None or kwargs.get("force_recompute", False):
            self.pred_equal_losses = self.pred_equal_loss_fn.compute(
                scores,
                labels,
                lambdas,
                groups,
                pos_label,
                group_id,
                num_classes,
                **kwargs,
            )

        return self.equal_opp_losses, self.pred_equal_losses

    def evaluate(
        self,
        pred_sets,
        labels,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        equal_op_loss = self.equal_opp_loss_fn.evaluate(
            pred_sets, labels, groups, pos_label, group_id, num_classes, **kwargs
        )

        pred_equal_loss = self.pred_equal_loss_fn.evaluate(
            pred_sets, labels, groups, pos_label, group_id, num_classes, **kwargs
        )

        return equal_op_loss, pred_equal_loss


class PredictiveParityLoss(CRCLoss):
    def compute(
        self,
        scores,
        labels,
        lambdas,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = torch.empty((scores.shape[0], len(lambdas)), dtype=torch.float)
        for i, lmbda in enumerate(lambdas):
            pred_sets = PredSetTransformation(threshold=lmbda).pipe_transform(scores)

            losses[:, i] = torch.where(
                pred_sets[:, pos_label],
                1 / ((pred_sets.sum(dim=-1)).to(float)) - 1 / num_classes,
                0,
            )

        # Monotonize losses to be non-increasing
        losses = torch.flip(
            torch.cummax(torch.flip(losses, dims=[-1]), dim=-1).values, dims=[-1]
        )
        losses[labels != pos_label, :] = 0
        losses[groups != group_id, :] = 0
        return losses

    def evaluate(
        self,
        pred_sets,
        labels,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        losses = torch.where(
            pred_sets[:, pos_label],
            1 / ((pred_sets.sum(dim=-1)).to(float)) - 1 / num_classes,
            0,
        ).reshape(len(labels), -1)

        # Monotonize losses to be non-increasing
        losses = torch.flip(
            torch.cummax(torch.flip(losses, dims=[-1]), dim=-1).values, dims=[-1]
        )
        losses[labels != pos_label] = 0
        losses[groups != group_id] = 0
        return losses


class ConditionalUseAccuracyLoss(CRCLoss):
    def __init__(
        self,
        base_loss_module: Optional[Union[CRCLoss, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pred_parity_loss_fn = PredictiveParityLoss(
            base_loss_module=base_loss_module, **kwargs
        )
        self.pred_equal_loss_fn = PredictiveEqualityLoss()(
            base_loss_module=base_loss_module, **kwargs
        )

        self.pred_parity_losses = None
        self.pred_equal_losses = None

    def compute(
        self,
        scores,
        labels,
        lambdas,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        if self.pred_parity_losses is None or kwargs.get("force_recompute", False):
            self.pred_parity_losses = self.pred_parity_loss_fn.compute(
                scores,
                labels,
                lambdas,
                groups,
                pos_label,
                group_id,
                num_classes,
                **kwargs,
            )

        if self.pred_equal_losses is None or kwargs.get("force_recompute", False):
            self.pred_equal_losses = self.pred_equal_loss_fn.compute(
                scores,
                labels,
                lambdas,
                groups,
                pos_label,
                group_id,
                num_classes,
                **kwargs,
            )

        return self.pred_parity_losses, self.pred_equal_losses

    def evaluate(
        self,
        pred_sets,
        labels,
        groups,
        pos_label,
        group_id,
        num_classes,
        **kwargs,
    ):
        pred_parity_loss = self.pred_parity_loss_fn.evaluate(
            pred_sets, labels, groups, pos_label, group_id, num_classes, **kwargs
        )

        pred_equal_loss = self.pred_equal_loss_fn.evaluate(
            pred_sets, labels, groups, pos_label, group_id, num_classes, **kwargs
        )

        return (
            pred_parity_loss,
            pred_equal_loss,
        )
