from typing import List, Optional

import lightning.pytorch as L
import torch
from conformal_fairness.config import SharedBaseConfig
from torch.utils.data import DataLoader, Subset

from ..constants import *
from ..utils import get_custom_dataset
from .base_datamodule import BaseDataset


class TabularDataset(BaseDataset):
    def __init__(self, name, X, y, sens, args: SharedBaseConfig):
        self.X = X
        self.y = y
        self.sens = sens
        self.masks = {}

        self.split_config = args.dataset_split_fractions
        self.seed = args.seed
        self.scaled = not (name in NEEDS_FEAT_SCALING)
        BaseDataset.__init__()

    def process(self):
        self._setup_masks(self.X.shape[0])

    def resplit_calib_test(self, seed: int):
        self._setup_masks(self.X.shape[0], seed)
        return self

    def split_calib_tune_qscore(self, tune_frac: float):
        tune_calib_nodes, qscore_calib_nodes = self._setup_calib_tune_qscore(
            self.X.shape[0], self.masks, tune_frac
        )
        self.masks[Stage.CALIBRATION_TUNE.mask_dstr] = tune_calib_nodes
        self.masks[Stage.CALIBRATION_QSCORE.mask_dstr] = qscore_calib_nodes
        return self

    def get_mask_inds(self, mask_key: str):
        mask = torch.Tensor(self.masks[mask_key])
        return torch.nonzero(mask, as_tuple=True)[0]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index], self.sens[index])

    def update_features(self, new_feats):
        self.X = new_feats


class TabularDataModule(L.LightningDataModule):
    def __init__(self, config: SharedBaseConfig) -> None:
        super(TabularDataModule, self).__init__(config)

    @property
    def labeled_points(self) -> torch.Tensor:
        return torch.nonzero(self.y >= 0, as_tuple=True)[0]

    @property
    def num_points(self) -> int:
        return self.X.shape[0]

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        return torch.unique(self.y[self.labeled_points])

    @property
    def num_sensitive_groups(self) -> int:
        return torch.unique(self.sens[self.labeled_points])

    def _create_dataset(
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

        def make_custom_lambda(
            dataset: str,
            pred_attrs,
            discard_attrs,
            sens_attrs,
            force_reprep,
            dataset_args,
        ):
            return lambda dir: get_custom_dataset(
                ds_name=dataset,
                ds_dir=dir,
                pred_attrs=pred_attrs,
                discard_attrs=discard_attrs,
                sens_attrs=sens_attrs,
                force_reprep=force_reprep,
                dataset_args=dataset_args,
            )

        dataset_init_funcs = {
            dataset: make_custom_lambda(
                dataset,
                pred_attrs,
                discard_attrs,
                sens_attrs,
                force_reprep,
                dataset_args,
            )
            for dataset in TABULAR_DATASETS
        }

        if name not in dataset_init_funcs:
            raise NotImplementedError(f"{name} not supported")
        return dataset_init_funcs[name](dataset_dir)

    def prepare_data(self) -> None:
        assert self.name is not None and self.name in TABULAR_DATASETS
        super(TabularDataModule, self).prepare_data()

    def _init_with_dataset(self, dataset: TabularDataset):
        self._base_dataset = dataset
        # init all available splits
        self.split_dict = {
            stage: dataset.get_mask_inds(stage.mask_dstr)
            for stage in Stage
            if stage.mask_dstr in self._base_dataset.masks
        }

        self.has_setup = True

    def setup(self, args: SharedBaseConfig) -> None:
        assert self.name is not None and self.name in TABULAR_DATASETS
        if not self.has_setup:
            X, y, sens = self._create_dataset(
                self.name,
                self.dataset_dir,
                pred_attrs=self.config.dataset.pred_attrs,
                discard_attrs=self.config.dataset.discard_attrs,
                sens_attrs=self.config.dataset.sens_attrs,
                dataset_args=self.config.dataset,
            )

            dataset = TabularDataset(self.name, X, y, sens, args)
            self._init_with_dataset(dataset)

    def train_dataloader(self):
        return DataLoader(
            Subset(self._base_dataset, self.split_dict[Stage.TRAIN]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self._base_dataset, self.split_dict[Stage.VALIDATION]),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            Subset(self._base_dataset, self.split_dict[Stage.TEST]),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def all_dataloader(self):
        return DataLoader(
            Subset(self._base_dataset, self.labeled_points),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
