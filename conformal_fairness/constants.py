import os
from enum import Enum

import torch

WORKING_DIRECTORY = os.path.dirname(__file__)

DATASET_DIRECTORY = os.path.join(WORKING_DIRECTORY, "datasets")

RAW_DIR = "raw"
PROCESSED_DIR = "processed"
FEATURE_FILE = "features.pt"
LABEL_FILE = "labels.pt"
EDGE_FILE = "edge_list.pt"
SENS_FILE = "sense_attr.pt"


# OUTPUT_DIRECTORY = os.path.join(WORKING_DIRECTORY, "outputs")

CPU_AFF = "enable_cpu_affinity"
PYTORCH_PRECISION = "medium"
ALL_OUTPUTS_FILE = "all_prob_labels.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Stage(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    CALIBRATION_TUNE = "calibration_tune"
    CALIBRATION_QSCORE = "calibration_qscore"
    TEST = "test"

    @property
    def mask_dstr(self):
        return {
            Stage.TRAIN: "train_mask",
            Stage.VALIDATION: "val_mask",
            Stage.CALIBRATION: "calib_mask",
            Stage.CALIBRATION_TUNE: "calib_tune_mask",
            Stage.CALIBRATION_QSCORE: "calib_qscore_mask",
            Stage.TEST: "test_mask",
        }[self]


class ConformalMethod(str, Enum):
    TPS = "tps"
    APS = "aps"
    RAPS = "raps"
    DAPS = "daps"
    DTPS = "dtps"
    NAPS = "naps"
    CFGNN = "cfgnn"


CORA = "Cora"
CITESEER = "CiteSeer"
PUBMED = "PubMed"
COAUTHOR_CS = "Coauthor_CS"
COAUTHOR_PHYSICS = "Coauthor_Physics"
AMAZON_PHOTOS = "Amazon_Photos"
AMAZON_COMPUTERS = "Amazon_Computers"
FLICKR = "Flickr"

OGBN_PRODUCTS = "ogbn-products"
OGBN_ARXIV = "ogbn-arxiv"
OGBN_DATASETS = [OGBN_PRODUCTS, OGBN_ARXIV]

# Fairness datasets
POKEC_N = "Pokec_n"
POKEC_Z = "Pokec_z"
NBA = "NBA"
GERMAN = "German"
BAIL = "Bail"
CREDIT = "Credit"
ACS_INCOME = "ACSIncome"
ACS_TRAVEL = "ACSTravelTime"
ENEM = 'ENEM'
ACS_EDUC = 'ACSEducation'

CUSTOM_DATASETS = [POKEC_N, POKEC_Z, NBA, GERMAN, BAIL, CREDIT, ACS_INCOME, ACS_TRAVEL, ENEM, ACS_EDUC]

CLASSIFICATION_DATASETS = (
    [
        CORA,
        CITESEER,
        PUBMED,
        COAUTHOR_CS,
        COAUTHOR_PHYSICS,
        AMAZON_PHOTOS,
        AMAZON_COMPUTERS,
        FLICKR,
    ]
    + CUSTOM_DATASETS
    + OGBN_DATASETS
)

NON_GRAPH_DATASETS = [ACS_INCOME, ACS_TRAVEL, ENEM, ACS_EDUC]
NEEDS_FEAT_SCALING = [ENEM]
FAIRNESS_DATASETS = CLASSIFICATION_DATASETS


class PreDefSplit(int, Enum):
    TRAIN = 0
    VALIDATION = 1
    TESTCALIB = 2


PREDEF_SPLIT_DATASETS = [FLICKR] + OGBN_DATASETS
PREDDEF_FIELD = "Pre_Def_Split"

PARTIALLY_LABELED = [POKEC_N, POKEC_Z, NBA]

FEATURE_FIELD = "feat"
LABEL_FIELD = "label"
SENS_FIELD = "sensitive_attr"

NODE_IDS_KEY = "node_ids"
PROBS_KEY = "probs"
SCORES_KEY = "scores"
LABELS_KEY = "labels"

BASEGNN_CKPT_CONFIG_FILE = "basegnn_config.yaml"
BASEGNN_CKPT_PREFIX = "basegnn"

conf_metric_names = Enum(
    "conf_metrics",
    [
        "set_sizes",
        "coverage",
        "efficiency",
        "feature_stratified_coverage",
        "size_stratified_coverage",
        "label_stratified_coverage",
        "singleton_hit_ratio",
        "size_stratified_coverage_violation",
    ],
)
sample_type = Enum("sample_type", "split n_samples_per_class")

layer_types = Enum("layer_types", ["GCN", "GAT", "GraphSAGE", "SGC"])

fairness_metric = Enum(
    "fairness_metrics",
    [
        "Equal_Opportunity",
        "Equalized_Odds",
        "Predictive_Parity",
        "Predictive_Equality",
        "Demographic_Parity",
        "Disparate_Impact",
        "Conditional_Use_Acc_Equality",
        "Overall_Acc_Equality",
    ],
)
