import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import lightning.pytorch as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ..config import ConfExptConfig, SharedBaseConfig
from ..constants import *


class BaseDataset(ABC, Dataset):
    def __init__(self, name, *args, **kwargs):
        self._name = name
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self):
        return self._name

    def _setup_masks(self, n_points, extra_calib_test_seed: Optional[int] = None):
        train_mask = torch.zeros(n_points, dtype=torch.bool)
        val_mask = torch.zeros(n_points, dtype=torch.bool)
        calib_mask = torch.zeros(n_points, dtype=torch.bool)
        test_mask = torch.zeros(n_points, dtype=torch.bool)

        gen = torch.Generator().manual_seed(self.seed)

        assert self.split_config is not None, f"Split config must be provided"

        if self.name in PARTIALLY_LABELED:
            n_points = int(sum(self.y >= 0))  # -1 points are unlabeled

        n_train = int(n_points * self.split_config.train)
        n_val = int(n_points * self.split_config.valid)
        n_calib = int(n_points * self.split_config.calib)

        if self.name in FAIRNESS_DATASETS:
            labeled_points = self.y >= 0
            groups = self.sens[labeled_points]
            labels = self.y[labeled_points]

            group_label_pairs = list(zip(groups, labels))
            ids = labeled_points.nonzero()

            train_ids, ids, _, group_label_pairs = train_test_split(
                labeled_points.nonzero(),
                group_label_pairs,
                train_size=n_train,
                stratify=group_label_pairs,
                random_state=self.seed,
            )

            val_ids, ids, _, group_label_pairs = train_test_split(
                ids,
                group_label_pairs,
                train_size=n_val,
                stratify=group_label_pairs,
                random_state=self.seed,
            )

            calib_ids, test_ids, _, _ = train_test_split(
                ids,
                group_label_pairs,
                train_size=n_calib,
                stratify=group_label_pairs,
                random_state=extra_calib_test_seed or self.seed,
            )

            train_mask[train_ids] = True
            val_mask[val_ids] = True
            calib_mask[calib_ids] = True
            test_mask[test_ids] = True

        elif not (self.name in PREDEF_SPLIT_DATASETS):
            # note: seed set in L.seed_everything
            r_order = np.random.permutation(n_points)  # Randomize order of points
            if self.name in PARTIALLY_LABELED:
                labeled_points = self.y >= 0
                r_order = np.random.permutation(labeled_points.nonzero())

            train_mask[r_order[:n_train]] = True
            val_mask[r_order[n_train : n_train + n_val]] = True

            # reset order
            if extra_calib_test_seed is not None:
                reshuffle_inds = r_order[n_train + n_val :]
                new_order = np.random.default_rng(
                    seed=extra_calib_test_seed
                ).permutation(reshuffle_inds)
                calib_mask[new_order[:n_calib]] = True
                test_mask[new_order[n_calib:]] = True
            else:
                calib_mask[r_order[n_train + n_val : n_train + n_val + n_calib]] = True
                test_mask[r_order[n_train + n_val + n_calib :]] = True

        else:
            train_mask = self.predef_splits == PreDefSplit.TRAIN

            if train_mask.sum() > n_train:
                overage = train_mask.sum() - n_train
                train_mask_indexes = train_mask.nonzero(as_tuple=True)[0]
                overage_idx = train_mask_indexes[
                    torch.randperm(len(train_mask_indexes), generator=gen)
                ][:overage]

                train_mask[overage_idx] = False

                logging.warning(
                    f"Predefined Training Split has {overage} more points than requested. These will be removed."
                )

            val_mask = self.predef_splits == PreDefSplit.VALIDATION

            if val_mask.sum() > n_val:
                overage = val_mask.sum() - n_val
                val_mask_indexes = val_mask.nonzero(as_tuple=True)[0]
                overage_idx = val_mask_indexes[
                    torch.randperm(len(val_mask_indexes), generator=gen)
                ][:overage]

                val_mask[overage_idx] = False

                logging.warning(
                    f"Predefined Validation Split has {overage} more points than requested. These will be removed."
                )

            calib_test_points = torch.nonzero(
                self.predef_splits == PreDefSplit.TESTCALIB,
                as_tuple=True,
            )[0]

            if extra_calib_test_seed is not None:
                calib_test_points = np.random.default_rng(
                    seed=extra_calib_test_seed
                ).permutation(calib_test_points)

            calib_mask[calib_test_points[:n_calib]] = True
            test_mask[calib_test_points[n_calib:]] = True

        return train_mask, val_mask, calib_mask, test_mask

    def _setup_calib_tune_qscore(self, n_points, mask_dict, tune_frac):
        assert Stage.CALIBRATION.mask_dstr in mask_dict
        calib_mask = mask_dict[Stage.CALIBRATION.mask_dstr]
        calib_points = calib_mask.nonzero(as_tuple=True)[0]
        N = len(calib_points)

        tune_calib_points = torch.zeros(n_points, dtype=torch.bool)
        qscore_calib_points = torch.zeros(n_points, dtype=torch.bool)

        if self.name in FAIRNESS_DATASETS and tune_frac > 0:
            groups = self.sens[calib_points]
            labels = self.y[calib_points]
            group_label_pairs = list(zip(groups, labels))

            tune_calib_ids, qscore_calib_ids, _, _ = train_test_split(
                calib_points,
                group_label_pairs,
                train_size=tune_frac,
                stratify=group_label_pairs,
                random_state=self.seed,
            )
        else:
            tune_ct = int(tune_frac * N)
            tune_calib_ids = calib_points[:tune_ct]
            qscore_calib_ids = calib_points[tune_ct:]

        tune_calib_points[tune_calib_ids] = True
        qscore_calib_points[qscore_calib_ids] = True

        return tune_calib_points, qscore_calib_points

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def resplit_calib_test(self, seed: int):
        pass

    @abstractmethod
    def split_calib_tune_qscore(self, tune_frac: float):
        pass

    @abstractmethod
    def get_mask_inds(self, mask_key: str):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def update_features(self, new_feats):
        pass


class BaseDataModule(ABC, L.LightningDataModule):
    def __init__(self, config: SharedBaseConfig) -> None:
        super().__init__()
        self.config = config

        self.name = config.dataset.name
        self.seed = config.seed
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.has_setup = False
        self.dataset_dir = config.dataset_dir
        self.split_dict: Dict[Stage, torch.Tensor] = {}

        self._base_dataset: Optional[BaseDataset] = None

    @property
    @abstractmethod
    def X(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def y(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def sens(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def labeled_points(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_points(self) -> int:
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_sensitive_groups(self) -> int:
        pass

    @abstractmethod
    def _create_dataset(
        self,
        name: str,
        dataset_dir: Optional[str] = None,
        /,
        *,
        pred_attrs: List[str] = [],
        discard_attrs: List[str] = [],
        sens_attrs: List[str] = [],
        dataset_args=None,
        force_reprep=False,
    ):
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        self._create_dataset(
            self.name,
            self.dataset_dir,
            pred_attrs=self.config.dataset.pred_attrs,
            discard_attrs=self.config.dataset.discard_attrs,
            sens_attrs=self.config.dataset.sens_attrs,
            force_reprep=self.config.dataset.force_reprep,
            dataset_args=self.config.dataset,
        )

    @abstractmethod
    def setup(self, args: SharedBaseConfig) -> None:
        pass

    @abstractmethod
    def _init_with_dataset(self, dataset: BaseDataset):
        pass

    def update_features(self, new_feats):
        assert self._base_dataset is not None
        self._base_dataset.update_features(new_feats)
        self._init_with_dataset(self._base_dataset)

    def resplit_calib_test(self, args: ConfExptConfig):
        # calib + test should be re split for a different conformal seed
        assert self._base_dataset is not None
        if args.conformal_seed is not None:
            dataset = self._base_dataset.resplit_calib_test(args.conformal_seed)
            self._init_with_dataset(dataset)

    def split_calib_tune_qscore(self, tune_frac: float):
        # resplit calib into tune and qscore sets
        dataset = self._base_dataset.split_calib_tune_qscore(tune_frac)
        self._init_with_dataset(dataset)

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def val_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def all_dataloader(self):
        pass

    @abstractmethod
    def custom_dataloader(
        self,
        points,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        pass
