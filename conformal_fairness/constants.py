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


class DatasetType(str, Enum):
    GRAPH = "graph"
    TABULAR = "tabular"


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
CREDIT = "Credit"
ACS_INCOME = "ACSIncome"
ACS_TRAVEL = "ACSTravelTime"
ENEM = "ENEM"
ACS_EDUC = "ACSEducation"

CUSTOM_GRAPH_DATASETS = [
    POKEC_N,
    POKEC_Z,
    CREDIT,
]


GRAPH_DATASETS = (
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
    + OGBN_DATASETS
    + CUSTOM_GRAPH_DATASETS
)

TABULAR_DATASETS = [ACS_INCOME, ACS_TRAVEL, ENEM, ACS_EDUC]
NEEDS_FEAT_SCALING = [ENEM]

FAIRNESS_DATASETS = CUSTOM_GRAPH_DATASETS + TABULAR_DATASETS

CLASSIFICATION_DATASETS = GRAPH_DATASETS + TABULAR_DATASETS


class PreDefSplit(int, Enum):
    TRAIN = 0
    VALIDATION = 1
    TESTCALIB = 2


PREDEF_SPLIT_DATASETS = [FLICKR] + OGBN_DATASETS
PREDDEF_FIELD = "Pre_Def_Split"

PARTIALLY_LABELED = [POKEC_N, POKEC_Z]

FEATURE_FIELD = "feat"
LABEL_FIELD = "label"
SENS_FIELD = "sensitive_attr"

NODE_IDS_KEY = "node_ids"
PROBS_KEY = "probs"
SCORES_KEY = "scores"
LABELS_KEY = "labels"

BASE_MODEL_CKPT_CONFIG_FILE = "base_model_config.yaml"
BASE_MODEL_CKPT_PREFIX = "base_model"


class ConformalMetric(str, Enum):
    SET_SIZES = "set_sizes"
    COVERAGE = "coverage"
    EFFICIENCY = "efficiency"
    FEATURE_STRATIFIED_COVERAGE = "feature_stratified_coverage"
    SIZE_STRATIFIED_COVERAGE = "size_stratified_coverage"
    LABEL_STRATIFIED_COVERAGE = "label_stratified_coverage"
    SINGLETON_HIT_RATIO = "singleton_hit_ratio"
    SIZE_STRATIFIED_COVERAGE_VIOLATION = "size_stratified_coverage_violation"


class LayerType(str, Enum):
    GCN = "GCN"
    GAT = "GAT"
    GRAPHSAGE = "GraphSAGE"


class FairnessMetric(str, Enum):
    EQUAL_OPPORTUNITY = "Equal_Opportunity"
    EQUALIZED_ODDS = "Equalized_Odds"
    PREDICTIVE_PARITY = "Predictive_Parity"
    PREDICTIVE_EQUALITY = "Predictive_Equality"
    DEMOGRAPHIC_PARITY = "Demographic_Parity"
    DISPARATE_IMPACT = "Disparate_Impact"
    CONDITIONAL_USE_ACC_EQUALITY = "Conditional_Use_Acc_Equality"
    OVERALL_ACC_EQUALITY = "Overall_Acc_Equality"
