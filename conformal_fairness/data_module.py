import logging
from typing import Dict, List, Optional

import dgl
import lightning.pytorch as L
import numpy as np
import torch
from conformal_fairness.config import ConfExptConfig, SharedBaseConfig
from dgl.data import (
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CiteseerGraphDataset,
    CoauthorCSDataset,
    CoauthorPhysicsDataset,
    CoraFullDataset,
    DGLDataset,
    FlickrDataset,
    PubmedGraphDataset,
)
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.model_selection import train_test_split

from .constants import *
from .data_utils import get_custom_dataset


class ClassificationDataset(DGLDataset):
    def __init__(self, name, graph: dgl.DGLGraph, args: SharedBaseConfig):
        self.loading_style = args.dataset_loading_style
        self.split_config = args.dataset_split_fractions
        self.n_samples_per_class = args.dataset_n_samples_per_class
        self.graph = graph
        self.seed = args.seed
        self.scaled = not (name in NEEDS_FEAT_SCALING)
        super().__init__(name=name)

    def _setup_masks(self, extra_calib_test_seed: Optional[int] = None):
        self.graph = dgl.add_self_loop(self.graph)
        n_nodes = self.graph.ndata[LABEL_FIELD].shape[0]
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        calib_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        gen = torch.Generator().manual_seed(self.seed)
        match self.loading_style:
            case sample_type.split.name:
                assert (
                    self.split_config is not None
                ), f"Split config must be provided for loading style {sample_type.split.name}"

                if self.name in PARTIALLY_LABELED:
                    n_nodes = int(
                        sum(self.graph.ndata[LABEL_FIELD] >= 0)
                    )  # -1 Nodes are unlabeled

                n_train = int(n_nodes * self.split_config.train)
                n_val = int(n_nodes * self.split_config.valid)
                n_calib = int(n_nodes * self.split_config.calib)

                if self.name in FAIRNESS_DATASETS:
                    labeled_nodes = self.graph.ndata[LABEL_FIELD] >= 0
                    groups = self.graph.ndata[SENS_FIELD][labeled_nodes]
                    labels = self.graph.ndata[LABEL_FIELD][labeled_nodes]

                    group_label_pairs = list(zip(groups, labels))
                    node_ids = labeled_nodes.nonzero()

                    train_ids, node_ids, _, group_label_pairs = train_test_split(
                        labeled_nodes.nonzero(),
                        group_label_pairs,
                        train_size=n_train,
                        stratify=group_label_pairs,
                        random_state=self.seed,
                    )

                    val_ids, node_ids, _, group_label_pairs = train_test_split(
                        node_ids,
                        group_label_pairs,
                        train_size=n_val,
                        stratify=group_label_pairs,
                        random_state=self.seed,
                    )

                    calib_ids, test_ids, _, _ = train_test_split(
                        node_ids,
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
                    r_order = np.random.permutation(n_nodes)  # Randomize order of nodes
                    if self.name in PARTIALLY_LABELED:
                        labeled_nodes = self.graph.ndata[LABEL_FIELD] >= 0
                        r_order = np.random.permutation(labeled_nodes.nonzero())

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
                        calib_mask[
                            r_order[n_train + n_val : n_train + n_val + n_calib]
                        ] = True
                        test_mask[r_order[n_train + n_val + n_calib :]] = True

                else:
                    train_mask = self.graph.ndata[PREDDEF_FIELD] == PreDefSplit.TRAIN

                    if train_mask.sum() > n_train:
                        overage = train_mask.sum() - n_train
                        train_mask_indexes = train_mask.nonzero(as_tuple=True)[0]
                        overage_idx = train_mask_indexes[
                            torch.randperm(len(train_mask_indexes), generator=gen)
                        ][:overage]

                        train_mask[overage_idx] = False

                        logging.warning(
                            f"Predefined Training Split has {overage} more nodes than requested. These will be removed."
                        )

                    val_mask = self.graph.ndata[PREDDEF_FIELD] == PreDefSplit.VALIDATION

                    if val_mask.sum() > n_val:
                        overage = val_mask.sum() - n_val
                        val_mask_indexes = val_mask.nonzero(as_tuple=True)[0]
                        overage_idx = val_mask_indexes[
                            torch.randperm(len(val_mask_indexes), generator=gen)
                        ][:overage]

                        val_mask[overage_idx] = False

                        logging.warning(
                            f"Predefined Validation Split has {overage} more nodes than requested. These will be removed."
                        )

                    calib_test_nodes = torch.nonzero(
                        self.graph.ndata[PREDDEF_FIELD] == PreDefSplit.TESTCALIB,
                        as_tuple=True,
                    )[0]

                    if extra_calib_test_seed is not None:
                        calib_test_nodes = np.random.default_rng(
                            seed=extra_calib_test_seed
                        ).permutation(calib_test_nodes)

                    calib_mask[calib_test_nodes[:n_calib]] = True
                    test_mask[calib_test_nodes[n_calib:]] = True

            case sample_type.n_samples_per_class.name:
                assert (
                    self.n_samples_per_class is not None
                ), f"n_samples_per_class must be provided for loading style {sample_type.n_samples_per_class.name}"
                if not (self.name in PREDEF_SPLIT_DATASETS):
                    for i in range(self.graph.ndata[LABEL_FIELD].max().item() + 1):
                        # get random permutation of nodes from this class
                        idx = torch.where(self.graph.ndata[LABEL_FIELD] == i)[0]
                        idx = idx[torch.randperm(idx.shape[0])]
                        # set the masks as per samples per class
                        train_mask[idx[: self.n_samples_per_class]] = True
                        val_mask[
                            idx[self.n_samples_per_class : 2 * self.n_samples_per_class]
                        ] = True
                        if extra_calib_test_seed is not None:
                            reshuffle_inds = idx[2 * self.n_samples_per_class :]
                            new_order = np.random.default_rng(
                                seed=(extra_calib_test_seed + i)
                            ).permutation(reshuffle_inds)
                            calib_mask[
                                new_order[
                                    2
                                    * self.n_samples_per_class : 3
                                    * self.n_samples_per_class
                                ]
                            ] = True
                            test_mask[new_order[3 * self.n_samples_per_class :]] = True
                        else:
                            calib_mask[
                                idx[
                                    2
                                    * self.n_samples_per_class : 3
                                    * self.n_samples_per_class
                                ]
                            ] = True
                            test_mask[idx[3 * self.n_samples_per_class :]] = True
                        if 3 * self.n_samples_per_class >= idx.shape[0]:
                            logging.warning(
                                f"Class {i} has insufficient samples to evaluate."
                            )

                else:
                    # Predefined Splits Are Provided
                    for i in range(self.graph.ndata[LABEL_FIELD].max().item() + 1):
                        idx = torch.where(self.graph.ndata[LABEL_FIELD] == i)[0]
                        pre_def_mask = self.graph.ndata[PREDDEF_FIELD][
                            idx
                        ]  # Masks Of Splits For ith class group
                        idx_perm = torch.randperm(idx.shape[0])

                        # permute the masks and indices
                        idx = idx[idx_perm]
                        pre_def_mask = pre_def_mask[idx_perm]

                        # get random permutation of nodes from this class
                        train_idx = idx[pre_def_mask == PreDefSplit.TRAIN]
                        train_mask[train_idx[: self.n_samples_per_class]] = True
                        if self.n_samples_per_class >= train_idx.shape[0]:
                            logging.warning(
                                f"Class {i} has insufficient samples to Train."
                            )

                        # Same As Above But For Validation nodes
                        val_idx = idx[pre_def_mask == PreDefSplit.VALIDATION]
                        val_mask[val_idx[: self.n_samples_per_class]] = True
                        if self.n_samples_per_class >= val_idx.shape[0]:
                            logging.warning(
                                f"Class {i} has insufficient samples to Validate."
                            )

                        # Compute Test and Calib Nodes
                        test_cal_idx = idx[pre_def_mask == PreDefSplit.TESTCALIB]

                        if extra_calib_test_seed is not None:
                            test_cal_idx = np.random.default_rng(
                                seed=(extra_calib_test_seed + i)
                            ).permutation(test_cal_idx)

                        calib_mask[test_cal_idx[: self.n_samples_per_class]] = True
                        test_mask[test_cal_idx[self.n_samples_per_class :]] = True

                        if self.n_samples_per_class >= test_cal_idx.shape[0]:
                            logging.warning(
                                f"Class {i} has insufficient samples to evaluate."
                            )
            case _:
                raise ValueError(f"Invalid loading style {self.loading_style}")
        self.graph.ndata[Stage.TRAIN.mask_dstr] = train_mask
        self.graph.ndata[Stage.VALIDATION.mask_dstr] = val_mask
        self.graph.ndata[Stage.CALIBRATION.mask_dstr] = calib_mask
        self.graph.ndata[Stage.TEST.mask_dstr] = test_mask

        # Scale Dataset if needed
        if not self.scaled:
            standard_scaler = TensorNormalizer().fit(
                (self.graph.ndata[FEATURE_FIELD])[train_mask, :]
            )
            self.update_features(
                standard_scaler.transform(self.graph.ndata[FEATURE_FIELD])
            )
            self.scaled = True

    def process(self):
        self._setup_masks()

    def resplit_calib_test(self, seed: int):
        self._setup_masks(seed)
        return self

    def split_graph_calib_tune_qscore(self, tune_frac: float):
        # split the calib into non overlapping tune and qscore
        assert Stage.CALIBRATION.mask_dstr in self.graph.ndata
        n_nodes = self.graph.num_nodes()
        calib_mask = self.graph.ndata[Stage.CALIBRATION.mask_dstr]
        calib_nodes = calib_mask.nonzero(as_tuple=True)[0]
        N = len(calib_nodes)

        tune_calib_nodes = torch.zeros(n_nodes, dtype=torch.bool)
        qscore_calib_nodes = torch.zeros(n_nodes, dtype=torch.bool)
        match self.loading_style:
            case sample_type.split.name:
                if self.name in FAIRNESS_DATASETS and tune_frac > 0:
                    groups = self.graph.ndata[SENS_FIELD][calib_nodes]
                    labels = self.graph.ndata[LABEL_FIELD][calib_nodes]
                    group_label_pairs = list(zip(groups, labels))

                    tune_calib_ids, qscore_calib_ids, _, _ = train_test_split(
                        calib_nodes,
                        group_label_pairs,
                        train_size=tune_frac,
                        stratify=group_label_pairs,
                        random_state=self.seed,
                    )
                else:
                    tune_ct = int(tune_frac * N)
                    tune_calib_ids = calib_nodes[:tune_ct]
                    qscore_calib_ids = calib_nodes[tune_ct:]

                tune_calib_nodes[tune_calib_ids] = True
                qscore_calib_nodes[qscore_calib_ids] = True
            case sample_type.n_samples_per_class.name:
                assert (
                    self.n_samples_per_class is not None
                ), f"n_samples_per_class must be provided for loading style {sample_type.n_samples_per_class.name}"

                for i in range(self.graph.ndata[LABEL_FIELD].max().item() + 1):
                    calib_class_i = (
                        self.graph.ndata[LABEL_FIELD][calib_nodes] == i
                    ).nonzero(as_tuple=True)[0]
                    n_class_i = calib_class_i.shape[0]
                    tune_ct_i = int(tune_frac * n_class_i)
                    if n_class_i > 0:
                        tune_i = calib_class_i[:tune_ct_i]
                        qscore_i = calib_class_i[tune_ct_i:]
                        tune_calib_nodes[tune_i] = True
                        qscore_calib_nodes[qscore_i] = True

        self.graph.ndata[Stage.CALIBRATION_TUNE.mask_dstr] = tune_calib_nodes
        self.graph.ndata[Stage.CALIBRATION_QSCORE.mask_dstr] = qscore_calib_nodes
        return self

    def get_mask_inds(self, mask_key: str):
        mask = torch.Tensor(self.graph.ndata[mask_key])
        return torch.nonzero(mask, as_tuple=True)[0]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        return self.graph

    def update_features(self, new_feats):
        self.graph.ndata[FEATURE_FIELD] = new_feats


def init_graph_dataset(
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

    def make_ogb_lambda(dataset: str):
        return lambda dir: DglNodePropPredDataset(name=dataset, root=dir)

    def make_custom_lambda(
        dataset: str, pred_attrs, discard_attrs, sens_attrs, force_reprep, dataset_args
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

    dataset_init_funcs = (
        {
            CORA: CoraFullDataset,
            CITESEER: CiteseerGraphDataset,
            PUBMED: PubmedGraphDataset,
            COAUTHOR_CS: CoauthorCSDataset,
            COAUTHOR_PHYSICS: CoauthorPhysicsDataset,
            AMAZON_PHOTOS: AmazonCoBuyPhotoDataset,
            AMAZON_COMPUTERS: AmazonCoBuyComputerDataset,
            FLICKR: FlickrDataset,
        }
        | {dataset: make_ogb_lambda(dataset) for dataset in OGBN_DATASETS}
        | {
            dataset: make_custom_lambda(
                dataset,
                pred_attrs,
                discard_attrs,
                sens_attrs,
                force_reprep,
                dataset_args,
            )
            for dataset in CUSTOM_DATASETS
        }
    )
    if name not in dataset_init_funcs:
        raise NotImplementedError(f"{name} not supported")
    return dataset_init_funcs[name](dataset_dir)


class DataModule(L.LightningDataModule):
    def __init__(self, config: SharedBaseConfig) -> None:
        super().__init__()
        self.config = config
        self.name = config.dataset.name
        self.seed = config.seed
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.has_setup = (
            False  # stateful, ensures that setup runs exactly once per expt
        )
        self.dataset_dir = config.dataset_dir
        self.split_dict: Dict[Stage, torch.Tensor] = {}

    @property
    def labeled_nodes(self) -> torch.Tensor:
        return (self.graph.ndata[LABEL_FIELD] >= 0).nonzero().squeeze()

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes()

    @property
    def num_features(self) -> int:
        features = torch.Tensor(self.graph.ndata[FEATURE_FIELD])
        return features.shape[1]

    @property
    def num_classes(self) -> int:
        labels = torch.Tensor(self.graph.ndata[LABEL_FIELD])
        return labels[self.labeled_nodes].unique().shape[0]

    @property
    def edge_index(self) -> torch.Tensor:
        return torch.stack(self.graph.edges(), dim=0)

    @property
    def adj_matrix(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.edge_index,
            torch.ones(self.edge_index.shape[1]),
            (self.num_nodes, self.num_nodes),
            dtype=torch.float,
        )

    @property
    def num_sensitive_groups(self) -> int:
        if self.name in FAIRNESS_DATASETS:
            return len(torch.unique(self.graph.ndata[SENS_FIELD]))
        raise NotImplementedError(
            f"No sensitive groups in {self.name} to be considered"
        )

    def prepare_data(self) -> None:
        assert self.name is not None and self.name in CLASSIFICATION_DATASETS
        init_graph_dataset(
            self.name,
            self.dataset_dir,
            pred_attrs=self.config.dataset.pred_attrs,
            discard_attrs=self.config.dataset.discard_attrs,
            sens_attrs=self.config.dataset.sens_attrs,
            force_reprep=self.config.dataset.force_reprep,
            dataset_args=self.config.dataset,
        )

    def _init_with_dataset(self, dataset: ClassificationDataset):
        self._base_dataset = dataset
        self.graph = dataset[0]
        # init all available splits
        self.split_dict = {
            stage: dataset.get_mask_inds(stage.mask_dstr)
            for stage in Stage
            if stage.mask_dstr in self.graph.ndata
        }

        self.has_setup = True

    def setup(self, args: SharedBaseConfig) -> None:
        assert self.name is not None
        if not self.has_setup:

            # Use DGL Dataset when possible
            dataset = init_graph_dataset(
                self.name,
                self.dataset_dir,
                pred_attrs=self.config.dataset.pred_attrs,
                discard_attrs=self.config.dataset.discard_attrs,
                sens_attrs=self.config.dataset.sens_attrs,
                dataset_args=self.config.dataset,
            )

            if self.name in CLASSIFICATION_DATASETS:
                graph = dataset[0]

                if self.name not in CUSTOM_DATASETS:
                    node_degree_exact = True
                    node_degree_percentile = 0.25

                    node_degrees = graph.in_degrees().float()
                    if node_degree_exact:
                        node_degrees.add_(torch.rand_like(node_degrees))
                    n_percentile = torch.quantile(
                        node_degrees, node_degree_percentile, interpolation="higher"
                    )

                    sens_attr = node_degrees < n_percentile
                    graph.ndata[SENS_FIELD] = sens_attr

                if self.name in OGBN_DATASETS:
                    graph, labels = dataset[0]
                    graph.ndata[LABEL_FIELD] = labels.reshape(-1)

                    # Set up predefined splits for OGB
                    split_idx = dataset.get_idx_split()
                    graph.ndata[PREDDEF_FIELD] = torch.zeros(len(labels))

                    graph.ndata[PREDDEF_FIELD][split_idx["train"]] = PreDefSplit.TRAIN
                    graph.ndata[PREDDEF_FIELD][
                        split_idx["valid"]
                    ] = PreDefSplit.VALIDATION
                    graph.ndata[PREDDEF_FIELD][
                        split_idx["test"]
                    ] = PreDefSplit.TESTCALIB

                elif self.name in PREDEF_SPLIT_DATASETS:
                    # Ensure Consistent Naming (since constants can change)
                    num_nodes = graph.ndata["train_mask"].shape[0]
                    graph.ndata[PREDDEF_FIELD] = torch.zeros(num_nodes)
                    graph.ndata[PREDDEF_FIELD] = (
                        (graph.ndata.pop("train_mask") * PreDefSplit.TRAIN)
                        + (graph.ndata.pop("val_mask") * PreDefSplit.VALIDATION)
                        + (graph.ndata.pop("test_mask") * PreDefSplit.TESTCALIB)
                    )

                dataset = ClassificationDataset(self.name, graph=graph, args=args)
                self._init_with_dataset(dataset)
            else:
                raise NotImplementedError

    def resplit_calib_test(self, args: ConfExptConfig):
        # calin + test should be re split for a different conformal seed
        if args.conformal_seed is not None:
            dataset = self._base_dataset.resplit_calib_test(args.conformal_seed)
            self._init_with_dataset(dataset)

    def split_calib_tune_qscore(self, tune_frac: float):
        # resplit calib into tune and qscore sets
        dataset = self._base_dataset.split_graph_calib_tune_qscore(tune_frac)
        self._init_with_dataset(dataset)

    def setup_sampler(self, num_layers: int, sampler_type: str = "full"):
        # TODO: implement alternative sampling strategies
        self.num_layers = num_layers
        self.sampler = MultiLayerFullNeighborSampler(num_layers)

    def train_dataloader(self):
        return DataLoader(
            self.graph,
            self.split_dict[Stage.TRAIN],
            self.sampler,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.graph,
            self.split_dict[Stage.VALIDATION],
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.graph,
            self.split_dict[Stage.TEST],
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def all_dataloader(self):
        return DataLoader(
            self.graph,
            self.labeled_nodes,
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def custom_nodes_dataloader(
        self,
        nodes,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler=None,
    ):
        if sampler is None:
            sampler = self.sampler
        return DataLoader(
            self.graph,
            nodes,
            sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )

    def _get_nodes_feature(self, nodes, feature_idx):
        return self.graph.ndata[FEATURE_FIELD][nodes, feature_idx]

    def get_test_nodes_features(self, feature_idx):
        return self._get_nodes_feature(self.split_dict[Stage.TEST], feature_idx)

    def update_features(self, new_feats):
        self._base_dataset.update_features(new_feats)
        self._init_with_dataset(self._base_dataset)


class TensorNormalizer:
    def __init__(self):
        self.comp_avgs = None
        self.comp_stds = None

    def fit(self, xtr):
        self.comp_avgs = xtr.mean(axis=0)
        self.comp_stds = xtr.std(axis=0)

        return self

    def transform(self, x):
        assert (self.comp_avgs is not None) and (self.comp_stds is not None)
        return (x - self.comp_avgs) / self.comp_stds

    def fit_transform(self, xtr):
        self.fit(xtr)
        return self.transform(xtr)

    def invert_transform(self, x_norm):
        return (x_norm * self.comp_stds) + self.comp_avgs
