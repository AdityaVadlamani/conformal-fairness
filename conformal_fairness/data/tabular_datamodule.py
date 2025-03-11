from typing import List, Optional
import torch
from torch.utils.data import DataLoader, Subset
from conformal_fairness.config import SharedBaseConfig

from ..constants import *
from .base_datamodule import BaseDataModule, BaseDataset
from ..utils.data_utils import get_custom_dataset


class TabularDataset(BaseDataset):
    def __init__(self, name, X, y, sens, args: SharedBaseConfig):
        self.X = X
        self.y = y
        self.sens = sens
        self.masks = {}

        self.split_config = args.dataset_split_fractions
        self.seed = args.seed
        self.scaled = not (name in NEEDS_FEAT_SCALING)
        super(TabularDataset, self).__init__(name=name)

    def process(self):
        return self._setup_masks(self.X.shape[0])

    def resplit_calib_test(self, seed: int):
        self._setup_masks(self.X.shape[0], seed)
        return self

    def split_calib_tune_qscore(self, tune_frac: float):
        tune_calib_points, qscore_calib_points = self._setup_calib_tune_qscore(
            self.X.shape[0], self.masks, tune_frac
        )
        self.masks[Stage.CALIBRATION_TUNE.mask_dstr] = tune_calib_points
        self.masks[Stage.CALIBRATION_QSCORE.mask_dstr] = qscore_calib_points
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


class TabularDataModule(BaseDataModule):
    def __init__(self, config: SharedBaseConfig) -> None:
        super(TabularDataModule, self).__init__(config)

    @property
    def X(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.as_tensor(self._base_dataset.X)

    @property
    def y(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.as_tensor(self._base_dataset.y)

    @property
    def sens(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        if self.name in FAIRNESS_DATASETS:
            return torch.as_tensor(self._base_dataset.sens)
        raise NotImplementedError(
            f"No sensitive groups in {self.name} to be considered"
        )

    @property
    def labeled_points(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.nonzero(self.y >= 0, as_tuple=True)[0]

    @property
    def num_points(self) -> int:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return self.X.shape[0]

    @property
    def num_features(self) -> int:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return self.y[self.labeled_points].unique().shape[0]

    @property
    def num_sensitive_groups(self) -> int:
        assert self.has_setup, f"Need to call setup before accessing properties"
        if self.name in FAIRNESS_DATASETS:
            return self.sens.unique().shape[0]
        raise NotImplementedError(
            f"No sensitive groups in {self.name} to be considered"
        )

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
        train_mask, val_mask, calib_mask, test_mask = dataset.process()
        dataset.masks[Stage.TRAIN.mask_dstr] = train_mask
        dataset.masks[Stage.VALIDATION.mask_dstr] = val_mask
        dataset.masks[Stage.CALIBRATION.mask_dstr] = calib_mask
        dataset.masks[Stage.TEST.mask_dstr] = test_mask

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

            # TODO: Add predefined splits case for tabular. Low priority since none of our datasets have that current
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

    def custom_dataloader(
        self,
        points,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            Subset(self._base_dataset, points),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )
