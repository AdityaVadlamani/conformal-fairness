from typing import List, Optional

import dgl
import torch
from conformal_fairness.config import SharedBaseConfig
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
from sklearn.preprocessing import StandardScaler

from ..constants import *
from ..data import BaseDataModule, BaseDataset
from ..utils.data_utils import get_custom_dataset


class GraphDataset(BaseDataset, DGLDataset):
    def __init__(self, name, graph: dgl.DGLGraph, args: SharedBaseConfig):
        super(GraphDataset, self).__init__(name)

        self.split_config = args.dataset_split_fractions
        self.graph = graph
        self.seed = args.seed
        self.scaled = not (name in NEEDS_FEAT_SCALING)

        self.X = self.graph.ndata[FEATURE_FIELD]
        self.y = self.graph.ndata[LABEL_FIELD]
        self.sens = self.graph.ndata[SENS_FIELD]
        if name in PREDEF_SPLIT_DATASETS:
            self.predef_splits = self.graph.ndata[PREDDEF_FIELD]

    def process(self):
        self.graph = dgl.add_self_loop(self.graph)
        n_nodes = self.graph.ndata[FEATURE_FIELD].shape[0]

        train_mask, val_mask, calib_mask, test_mask = self._setup_masks(n_nodes)

        self.graph.ndata[Stage.TRAIN.mask_dstr] = train_mask
        self.graph.ndata[Stage.VALIDATION.mask_dstr] = val_mask
        self.graph.ndata[Stage.CALIBRATION.mask_dstr] = calib_mask
        self.graph.ndata[Stage.TEST.mask_dstr] = test_mask

        # Scale Dataset if needed
        if not self.scaled:
            standard_scaler = StandardScaler()
            standard_scaler.fit(
                (self.graph.ndata[FEATURE_FIELD])[train_mask, :].numpy()
            )
            self.update_features(
                standard_scaler.transform(self.graph.ndata[FEATURE_FIELD])
            )
            self.scaled = True

        return self

    def resplit_calib_test(self, seed: int):
        n_nodes = self.graph.ndata[FEATURE_FIELD].shape[0]
        self._setup_masks(n_nodes, seed)
        return self

    def split_calib_tune_qscore(self, tune_frac: float):
        tune_calib_nodes, qscore_calib_nodes = self._setup_calib_tune_qscore(
            self.graph.num_nodes(), self.graph.ndata, tune_frac
        )
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


class GraphDataModule(BaseDataModule):
    def __init__(self, config: SharedBaseConfig) -> None:
        super(GraphDataModule, self).__init__(config)

    @property
    def X(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.as_tensor(self.graph.ndata[FEATURE_FIELD])

    @property
    def y(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.as_tensor(self.graph.ndata[LABEL_FIELD])

    @property
    def sens(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        if self.name in FAIRNESS_DATASETS:
            return torch.as_tensor(self.graph.ndata[SENS_FIELD])
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
        return self.graph.num_nodes()

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

    @property
    def edge_index(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.stack(self.graph.edges(), dim=0)

    @property
    def adj_matrix(self) -> torch.Tensor:
        assert self.has_setup, f"Need to call setup before accessing properties"
        return torch.sparse_coo_tensor(
            self.edge_index,
            torch.ones(self.edge_index.shape[1]),
            (self.num_points, self.num_points),
            dtype=torch.float,
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

        def make_ogb_lambda(dataset: str):
            return lambda dir: DglNodePropPredDataset(name=dataset, root=dir)

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
                for dataset in CUSTOM_GRAPH_DATASETS
            }
        )
        if name not in dataset_init_funcs:
            raise NotImplementedError(f"{name} not supported")
        return dataset_init_funcs[name](dataset_dir)

    def prepare_data(self) -> None:
        assert self.name is not None and self.name in GRAPH_DATASETS
        super(GraphDataModule, self).prepare_data()

    def _init_with_dataset(self, dataset: GraphDataset):
        self._base_dataset = dataset
        self._base_dataset.process()

        self.graph = dataset[0]
        # init all available splits
        self.split_dict = {
            stage: dataset.get_mask_inds(stage.mask_dstr)
            for stage in Stage
            if stage.mask_dstr in self.graph.ndata
        }

        self.has_setup = True

    def setup(self, args: SharedBaseConfig) -> None:
        assert self.name is not None and self.name in GRAPH_DATASETS
        if not self.has_setup:
            # Use DGL Dataset when possible
            dataset = self._create_dataset(
                self.name,
                self.dataset_dir,
                pred_attrs=self.config.dataset.pred_attrs,
                discard_attrs=self.config.dataset.discard_attrs,
                sens_attrs=self.config.dataset.sens_attrs,
                dataset_args=self.config.dataset,
            )

            graph = dataset[0]

            if self.name not in CUSTOM_GRAPH_DATASETS:
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
                graph.ndata[PREDDEF_FIELD][split_idx["valid"]] = PreDefSplit.VALIDATION
                graph.ndata[PREDDEF_FIELD][split_idx["test"]] = PreDefSplit.TESTCALIB

            elif self.name in PREDEF_SPLIT_DATASETS:
                # Ensure Consistent Naming (since constants can change)
                num_nodes = graph.ndata["train_mask"].shape[0]
                graph.ndata[PREDDEF_FIELD] = torch.zeros(num_nodes)
                graph.ndata[PREDDEF_FIELD] = (
                    (graph.ndata.pop("train_mask") * PreDefSplit.TRAIN)
                    + (graph.ndata.pop("val_mask") * PreDefSplit.VALIDATION)
                    + (graph.ndata.pop("test_mask") * PreDefSplit.TESTCALIB)
                )

            dataset = GraphDataset(self.name, graph=graph, args=args)
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
            self.labeled_points,
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def custom_dataloader(
        self,
        nodes,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        sampler = kwargs.get("sampler", None)
        if kwargs.get("sampler", None) is None:
            sampler = self.sampler

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            self.graph,
            nodes,
            sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )
