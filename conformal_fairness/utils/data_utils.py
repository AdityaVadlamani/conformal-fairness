import os
import shutil
from typing import List

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from folktables import (
    ACSDataSource,
    BasicProblem,
    adult_filter,
    state_list,
    travel_time_filter,
)
from sklearn.preprocessing import MinMaxScaler

from ..constants import *


def get_label_scores(labels, scores, mask, dataset):
    mod_labels = labels.unsqueeze(1)

    if dataset in PARTIALLY_LABELED:
        assert not torch.any(mod_labels[mask] < 0)
        mod_labels[mod_labels < 0] = (
            0  # Should be safe since -1 labels won't be in calibration
        )

    label_scores = scores.gather(1, mod_labels).squeeze()
    return label_scores[mask]


def save_processed_dataset(
    dataset_path, features, labels, sens_attr=None, edge_list=None, extra_folders=""
):
    os.makedirs(os.path.join(dataset_path, PROCESSED_DIR, extra_folders), exist_ok=True)
    torch.save(
        features, os.path.join(dataset_path, PROCESSED_DIR, extra_folders, FEATURE_FILE)
    )
    torch.save(
        labels, os.path.join(dataset_path, PROCESSED_DIR, extra_folders, LABEL_FILE)
    )
    if edge_list is not None:
        torch.save(
            edge_list,
            os.path.join(dataset_path, PROCESSED_DIR, extra_folders, EDGE_FILE),
        )
    if sens_attr is not None:
        torch.save(
            sens_attr,
            os.path.join(dataset_path, PROCESSED_DIR, extra_folders, SENS_FILE),
        )


def load_processed_dataset(dataset_path, extra_folders=""):
    edge_list = None
    sens_attr = None

    features = torch.load(
        os.path.join(dataset_path, PROCESSED_DIR, extra_folders, FEATURE_FILE)
    )
    labels = torch.load(
        os.path.join(dataset_path, PROCESSED_DIR, extra_folders, LABEL_FILE)
    )
    if os.path.exists(
        os.path.join(dataset_path, PROCESSED_DIR, extra_folders, EDGE_FILE)
    ):
        edge_list = torch.load(
            os.path.join(dataset_path, PROCESSED_DIR, extra_folders, EDGE_FILE)
        )

    if os.path.exists(
        os.path.join(dataset_path, PROCESSED_DIR, extra_folders, SENS_FILE)
    ):
        sens_attr = torch.load(
            os.path.join(dataset_path, PROCESSED_DIR, extra_folders, SENS_FILE)
        )

    return features, labels, sens_attr, edge_list


def scale_attr(df, col_list):
    """
    Adapted from FAIRMile: https://github.com/heyuntian/FairMILE
    Scale the attributes in col_list to integers starting from 0, then create an array of attributes in col_list.
    :param df:
    :param col_list:
    :return: attribute array of (n, len(col_list))
    """
    num_attrs = len(col_list)
    for attr_id in range(num_attrs):
        attr = col_list[attr_id]
        uniq_values = list(df[attr].unique())
        flag_not_all_int = False
        flag_has_negative = False
        for i in range(len(uniq_values)):
            is_int = isinstance(uniq_values[i], int) or isinstance(
                uniq_values[i], np.int64
            )
            flag_not_all_int = flag_not_all_int or not is_int
            if is_int:
                flag_has_negative = flag_has_negative or (uniq_values[i] < 0)

        if flag_not_all_int or flag_has_negative:
            if flag_not_all_int:
                map_attr = {j: i for i, j in enumerate(uniq_values)}
            else:
                uniq_values = sorted(uniq_values)
                map_attr = {j: i for i, j in enumerate(uniq_values)}
            data = list(map(map_attr.get, df[attr]))
            df[attr] = data
    arr = df[col_list].values
    return arr


def read_nifty(
    pred_attrs, discard_attrs, sens_attrs, dataset_path, edge_file, csv_file
):
    """
    Adapted from FAIR Mile (https://github.com/heyuntian/FairMILE)
    Which adapted from:
    Read Datasets used by Nifty
    Adapted from GitHub @chirag126/nifty/utils.py, line 174, load_german
    https://github.com/chirag126/nifty/blob/main/utils.py
    """
    path = os.path.join(dataset_path, RAW_DIR)

    # Basics
    idx_features_labels = pd.read_csv(os.path.join(path, csv_file))
    header = list(idx_features_labels.columns)
    for attr in pred_attrs:
        header.remove(attr)
    for attr in discard_attrs:
        header.remove(attr)

    # Sensitive attribute: Numpy array
    sens = scale_attr(idx_features_labels, sens_attrs)
    node_num = sens.shape[0]

    # Predict attribute: NumPy array
    labels = scale_attr(idx_features_labels, pred_attrs)

    # Features: NumPy array
    features = np.array(
        (sp.csr_matrix(idx_features_labels[header], dtype=np.float32)).todense()
    )

    # Graph Structure: CSR matrix
    edges_unordered = np.genfromtxt(os.path.join(path, edge_file)).astype(
        "int"
    )  # E x 2
    idx = np.arange(node_num)
    old2new = {j: i for i, j in enumerate(idx)}
    edge_list = np.array(
        list(map(old2new.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    # Make the edge list symmetric
    reverse_edge = np.flip(edge_list, 1)
    bidirectional_edge = np.vstack((edge_list, reverse_edge))
    edge_list = np.unique(bidirectional_edge, axis=0).T  # Get rid of duplicates

    save_processed_dataset(
        dataset_path,
        torch.from_numpy(features),
        torch.from_numpy(labels).reshape((-1,)),
        torch.from_numpy(edge_list),
        torch.from_numpy(sens).reshape((-1,)),
        extra_folders="_".join(sens_attrs),
    )


def prep_credit(
    dataset_path,
    /,
    *,
    pred_attrs: List[str] = [],
    discard_attrs: List[str] = [],
    sens_attrs: List[str] = [],
):
    pred_attrs = pred_attrs or ["NoDefaultNextMonth"]
    discard_attrs = discard_attrs or ["Single"]
    sens_attrs = sens_attrs or ["Age"]

    edge_file = "credit_edges.txt"
    csv_file = "credit.csv"
    read_nifty(
        pred_attrs=pred_attrs,
        discard_attrs=discard_attrs,
        sens_attrs=sens_attrs,
        dataset_path=dataset_path,
        edge_file=edge_file,
        csv_file=csv_file,
    )
    return pred_attrs, discard_attrs, sens_attrs


def prep_pokec_n(
    dataset_path,
    /,
    *,
    pred_attrs: List[str] = [],
    discard_attrs: List[str] = [],
    sens_attrs: List[str] = [],
):
    pred_attrs = pred_attrs or ["I_am_working_in_field"]
    discard_attrs = discard_attrs or []
    sens_attrs = sens_attrs or ["region", "gender"]

    prep_fairgcn_datasets(
        dataset_path,
        pred_attrs=pred_attrs,
        discard_attrs=discard_attrs,
        sens_attrs=sens_attrs,
    )
    return pred_attrs, discard_attrs, sens_attrs


def prep_pokec_z(
    dataset_path,
    /,
    *,
    pred_attrs: List[str] = [],
    discard_attrs: List[str] = [],
    sens_attrs: List[str] = [],
):
    pred_attrs = pred_attrs or ["I_am_working_in_field"]
    discard_attrs = discard_attrs or []
    sens_attrs = sens_attrs or ["region", "gender"]

    prep_fairgcn_datasets(
        dataset_path,
        pred_attrs=pred_attrs,
        discard_attrs=discard_attrs,
        sens_attrs=sens_attrs,
    )
    return pred_attrs, discard_attrs, sens_attrs


def schl_transform(x):
    temp = x
    e = 0.000000001
    x = (
        (temp >= 1 - e) * (temp <= 15 + e)
    ) * 1  # No schooling (+ primar _ high school only)
    x += ((temp >= 16 - e) * (temp <= 16 + e)) * 2  # High School - no college
    x += ((temp >= 17 - e) * (temp <= 17 + e)) * 3  # GED - no college
    x += ((temp >= 18 - e) * (temp <= 20 + e)) * 4  # Started college/associates
    x += ((temp >= 21 - e) * (temp <= 21 + e)) * 5  # Bachelor's Degree
    x += ((temp >= 22 - e) * (temp <= 24 + e)) * 6  # Grad School/Professional Degree

    return x - 1


def schl_transform_small(x):
    temp = x
    e = 0.000000001
    x = (
        (temp >= 1 - e) * (temp <= 15 + e)
    ) * 1  # No schooling (+ primar _ high school only)
    x += (
        (temp >= 16 - e) * (temp <= 20 + e)
    ) * 2  # High School - no college  + # GED - no college + # Started college/associates
    x += ((temp >= 21 - e) * (temp <= 21 + e)) * 3  # Bachelor's Degree
    x += ((temp >= 22 - e) * (temp <= 24 + e)) * 4  # Grad School/Professional Degree

    return x - 1


def schl_filter(data):
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    --Adapted from Folktable library/code
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    df = df[df["PWGTP"] >= 1]
    df = df[df["SCHL"].notna()]  # Added this line to allow it to work
    return df


def prep_education_level(dataset_path, dataset_args):
    """
    https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py#L84C1-L84C10
    # Taking Standard ACS TravelTime Data Set However changing targets
    """
    sens_binary = dataset_args.binary_sens
    tgt_transform = (
        schl_transform if not dataset_args.small_class else schl_transform_small
    )
    ACSTravelTime = BasicProblem(
        features=[
            "AGEP",
            "JWMNP",
            "MAR",
            "SEX",
            "DIS",
            "ESP",
            "MIG",
            "RELP",
            "RAC1P",
            "PUMA",
            "ST",
            "CIT",
            "OCCP",
            "JWTR",
            "POWPUMA",
            "POVPIP",
        ],
        target="SCHL",
        target_transform=schl_transform,
        group="RAC1P",
        preprocess=schl_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    year = "2018"
    horizon = "1-Year"  # or 5-Year
    method = "person"  # Or 'household'
    data = ACSDataSource(year, horizon, method, os.path.join(dataset_path, RAW_DIR))

    # features, labels, sens = ACSIncome.df_to_numpy(data.get_data(states = ['CA','NY','TX','AK','AL'], download=True))
    # features = torch.tensor(features, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.int64)
    # sens = torch.tensor(sens, dtype=torch.float32) - 1 # subtract 1 since groups are 1,2, ..., 9
    feat_list = []
    label_list = []
    sens_list = []
    for state in state_list:
        features, labels, sens = ACSTravelTime.df_to_numpy(
            data.get_data(states=[state], download=True)
        )
        feat_list.append(torch.tensor(features, dtype=torch.float32))
        label_list.append(torch.tensor(labels, dtype=torch.int64))
        sens_list.append(
            torch.tensor(sens, dtype=torch.float32) - 1
        )  # subtract 1 since groups are 1,2, ..., 9

    features = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list)
    sens = torch.cat(sens_list)

    if sens_binary:
        sens = (sens > 0) * 1  # White vs. Non-White
    save_processed_dataset(dataset_path, features, labels, sens)


def prep_acs_travel_time(dataset_path, dataset_args):
    """
    https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py#L84C1-L84C10
    # Taking Standard ACS TravelTime Data Set However changing targets
    """
    acs_travel_multi = True
    sens_binary = dataset_args.binary_sens
    target_transform = (
        time_breakdowns if not dataset_args.small_class else (lambda x: x > 20)
    )
    ACSTravelTime = BasicProblem(
        features=[
            "AGEP",
            "SCHL",
            "MAR",
            "SEX",
            "DIS",
            "ESP",
            "MIG",
            "RELP",
            "RAC1P",
            "PUMA",
            "ST",
            "CIT",
            "OCCP",
            "JWTR",
            "POWPUMA",
            "POVPIP",
        ],
        target="JWMNP",
        target_transform=target_transform,
        group="RAC1P",
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    year = "2018"
    horizon = "1-Year"  # or 5-Year
    method = "person"  # Or 'household'
    data = ACSDataSource(year, horizon, method, os.path.join(dataset_path, RAW_DIR))

    # features, labels, sens = ACSIncome.df_to_numpy(data.get_data(states = ['CA','NY','TX','AK','AL'], download=True))
    # features = torch.tensor(features, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.int64)
    # sens = torch.tensor(sens, dtype=torch.float32) - 1 # subtract 1 since groups are 1,2, ..., 9
    feat_list = []
    label_list = []
    sens_list = []
    for state in state_list:
        features, labels, sens = ACSTravelTime.df_to_numpy(
            data.get_data(states=[state], download=True)
        )
        feat_list.append(torch.tensor(features, dtype=torch.float32))
        label_list.append(torch.tensor(labels, dtype=torch.int64))
        sens_list.append(
            torch.tensor(sens, dtype=torch.float32) - 1
        )  # subtract 1 since groups are 1,2, ..., 9

    features = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list)
    sens = torch.cat(sens_list)
    if sens_binary:
        sens = (sens > 0) * 1  # White vs. Non-White
    save_processed_dataset(dataset_path, features, labels, sens)


def time_breakdowns(x):
    return (x >= 15) * 1 + (x >= 30) * 1 + (x >= 45) * 1


def prep_acs_income(
    dataset_path,
    dataset_args,
):
    """
    https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py#L84C1-L84C10
    # Taking Standard ACS Income Data Set However changing targets
    """

    # acs_income_multi = True
    sens_binary = dataset_args.binary_sens
    target_transform = (
        tax_breakdowns if not dataset_args.small_class else (lambda x: x > 50000)
    )
    ACSIncome = BasicProblem(
        features=[
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "OCCP",
            "POBP",
            "RELP",
            "WKHP",
            "SEX",
            "RAC1P",
        ],
        target="PINCP",
        target_transform=target_transform,
        group="RAC1P",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    year = "2018"
    horizon = "1-Year"  # or 5-Year
    method = "person"  # Or 'household'
    data = ACSDataSource(year, horizon, method, os.path.join(dataset_path, RAW_DIR))

    # features, labels, sens = ACSIncome.df_to_numpy(data.get_data(states = ['CA','NY','TX','AK','AL'], download=True))
    # features = torch.tensor(features, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.int64)
    # sens = torch.tensor(sens, dtype=torch.float32) - 1 # subtract 1 since groups are 1,2, ..., 9
    feat_list = []
    label_list = []
    sens_list = []
    for state in state_list:
        features, labels, sens = ACSIncome.df_to_numpy(
            data.get_data(states=[state], download=True)
        )
        feat_list.append(torch.tensor(features, dtype=torch.float32))
        label_list.append(torch.tensor(labels, dtype=torch.int64))
        sens_list.append(
            torch.tensor(sens, dtype=torch.float32) - 1
        )  # subtract 1 since groups are 1,2, ..., 9

    features = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list)
    sens = torch.cat(sens_list)
    if sens_binary:
        sens = (sens > 0) * 1  # White vs. Non-White
    save_processed_dataset(dataset_path, features, labels, sens)


def tax_breakdowns(x):
    """2018 TAX BREAK DOWNS  (ACS COLLECTED DURING 2018)"""
    """https://taxfoundation.org/data/all/federal/2018-tax-brackets/"""
    """Using Individual's Breakdowns since the incomes are individual"""
    quantiles = np.nanquantile(x, np.linspace(0.0, 1.0, 4 + 1))[1:-1]
    val = x < -10
    for q in quantiles:
        val += (x >= q) * 1

    return val
    # return (x>=25000)*1 + (x>=50000)*1 + (x>=75000)*1 + (x>=100000)*1
    # return (x>=9525)*1 + (x>=38700)*1 + (x>=82500)*1 + (x>=157500)*1 + (x>=200000)*1  + (x>=500000)*1


def prep_enem(dataset_path, dataset_args):
    """
    TODO:
    Figure out if we should remove all test scores/ labels, then choose one as the label or leave the remaining as predictors
    Determine if we want a min-max scaler (or standard scaler and if this should be learning on the training data alone)
    Determine if all group attributes should be remove (gender, race) or just the one we are using as the sens attr
    """

    # Acc Math > Languages > Human Science > natural science -- Try a different problem where we add up the number of courses a student is in the top 50% for
    label = [
        "NU_NOTA_CN"
    ]  ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
    other_tests = ["NU_NOTA_MT", "NU_NOTA_CH", "NU_NOTA_LC"]
    all_labels = False
    group_attribute = ["TP_COR_RACA", "TP_SEXO"]
    question_vars = [
        "Q00" + str(x) if x < 10 else "Q0" + str(x) for x in range(1, 25)
    ]  # changed for 2020
    domestic_vars = ["SG_UF_PROVA", "TP_FAIXA_ETARIA"]  # changed for 2020
    all_vars = label + group_attribute + question_vars + domestic_vars + other_tests
    group = "race"  # 'gender' or 'both' are options
    multigroup = not dataset_args.binary_sens  # Could be False
    n_sample = None  # 50000, 1200000 are options used by original authors
    n_classes = (
        4 if not dataset_args.small_class else 2
    )  # also 2 is used by the authors

    load_enem(
        dataset_path=dataset_path,
        features=all_vars,
        grade_attribute=label,
        n_sample=n_sample,
        n_classes=n_classes,
        group=group,
        multigroup=multigroup,
        all_label=all_labels,
    )


def prep_fairgcn_datasets(
    dataset_path,
    pred_attrs: List[str] = [],
    discard_attrs: List[str] = [],
    sens_attrs: List[str] = [],
):
    """
    Adapted from FAIR Mile (https://github.com/heyuntian/FairMILE)
    Which adapted from:
    Read Pokec data used by FairGCN (Dai and Wang, WSDM '21)
    Adapted from https://github.com/EnyanDai/FairGNN/blob/main/src/utils.py
    :param args:
    :param pred_attrs:
    :param discard_attrs:
    :param sens_attrs:
    :return:
    """

    path = os.path.join(dataset_path, RAW_DIR)

    idx_features_labels = pd.read_csv(os.path.join(path, f"features_labels.csv"))

    # All FairGCN datasets have user_id as an attribute, so this is fine
    discard_attrs = ["user_id"] + sens_attrs + pred_attrs

    header = [attr for attr in idx_features_labels.columns if attr not in discard_attrs]

    labels = idx_features_labels[pred_attrs[0]].values

    # Sensitive attribute: Numpy array
    sens_attr = scale_attr(idx_features_labels, sens_attrs)

    if sens_attr.shape[-1] == 2:
        sens_attr = (2 * sens_attr[:, 0] + sens_attr[:, 1]).reshape(-1, 1)

    # Predict attribute: NumPy array
    label_idx = np.where(labels >= 0)[0]
    labels = labels[label_idx]
    labels = np.append(label_idx.reshape(-1, 1), labels.reshape(-1, 1), 1).astype(
        np.int32
    )

    # Features: NumPy array
    features = np.array(
        (sp.csr_matrix(idx_features_labels[header], dtype=np.float32)).todense()
    )

    # Get Edge List
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, f"relationships.txt")).astype(
        "int"
    )
    edge_list = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)

    # Make the edge list symmetric
    reverse_edge = np.flip(edge_list, 1)
    bidirectional_edge = np.vstack((edge_list, reverse_edge))
    edge_list = np.unique(bidirectional_edge, axis=0).T  # Get rid of duplicates
    fill_labels = -1 * np.ones_like(sens_attr)  # Fill Unlabeled with -1
    fill_labels[labels[:, 0]] = np.vstack(labels[:, 1])

    assert sens_attr.shape[-1] == 1, f"{sens_attr.shape}"

    save_processed_dataset(
        dataset_path,
        torch.from_numpy(features),
        torch.from_numpy(fill_labels.reshape((-1,))),
        torch.from_numpy(sens_attr.reshape((-1,))),
        torch.from_numpy(edge_list),
        extra_folders="_".join(sens_attrs),
    )


def update_attrs(ds_name: str, /, *, pred_attrs, discard_attrs, sens_attrs):
    if ds_name in [POKEC_N, POKEC_Z]:
        pred_attrs = pred_attrs or ["I_am_working_in_field"]
        discard_attrs = discard_attrs or []
        sens_attrs = sens_attrs or ["region", "gender"]
        if len(sens_attrs) == 1 and sens_attrs[0] == "region_gender":
            sens_attrs = ["region", "gender"]
    elif ds_name in [CREDIT]:
        pred_attrs = pred_attrs or ["NoDefaultNextMonth"]
        discard_attrs = discard_attrs or ["Single"]
        sens_attrs = sens_attrs or ["Age"]

    return pred_attrs, discard_attrs, sens_attrs


def data_prep(
    ds_name: str,
    ds_dir: str,
    /,
    *,
    pred_attrs: List[str] = [],
    discard_attrs: List[str] = [],
    sens_attrs: List[str] = [],
    dataset_args=None,
):
    """Prepare the raw data for use with DGL"""
    dataset_path = os.path.join(ds_dir, ds_name)
    if not os.path.exists(os.path.join(dataset_path, PROCESSED_DIR)):
        # Create the processed directory
        os.makedirs(os.path.join(dataset_path, PROCESSED_DIR), exist_ok=True)
    try:
        ds_func = {
            POKEC_N: prep_pokec_n,
            POKEC_Z: prep_pokec_z,
            CREDIT: prep_credit,
            ACS_INCOME: prep_acs_income,
            ACS_TRAVEL: prep_acs_travel_time,
            ENEM: prep_enem,
            ACS_EDUC: prep_education_level,
        }

        if ds_name in [ACS_INCOME, ACS_TRAVEL, ACS_EDUC, ENEM]:
            os.makedirs(os.path.join(dataset_path, RAW_DIR), exist_ok=True)
            return ds_func[ds_name](dataset_path, dataset_args)

        return ds_func[ds_name](
            dataset_path,
            pred_attrs=pred_attrs,
            discard_attrs=discard_attrs,
            sens_attrs=sens_attrs,
        )

    except Exception as e:
        # dataset gen failed, delete the processed dir
        if os.path.exists(os.path.join(dataset_path, PROCESSED_DIR)):
            shutil.rmtree(os.path.join(dataset_path, PROCESSED_DIR))
        raise e


def create_graph(ds_name: str, ds_dir: str, extra_folders: str = ""):
    dataset_path = os.path.join(ds_dir, ds_name)
    features, labels, sens_attr, edge_list = load_processed_dataset(
        dataset_path, extra_folders
    )
    """
    Assumptions: 
    features: (num nodes x num features)
    labels: (num nodes, ) 
    sens attr: (num nodes, )
    edge list (2 x num edge)
    """

    num_nodes = labels.shape[0]
    g = None  #
    if ds_name in TABULAR_DATASETS:
        g = dgl.graph(([], []), num_nodes=num_nodes)
    else:
        g = dgl.graph((edge_list[0, :], edge_list[1, :]), num_nodes=num_nodes)
    g.ndata[FEATURE_FIELD] = features
    g.ndata[LABEL_FIELD] = labels

    if sens_attr is not None:
        g.ndata[SENS_FIELD] = sens_attr
    return [g]


def get_custom_dataset(
    ds_name: str,
    ds_dir: str,
    *,
    pred_attrs: List[str] = [],
    discard_attrs: List[str] = [],
    sens_attrs: List[str] = [],
    force_reprep: bool = False,
    dataset_args=None,
):
    dataset_path = os.path.join(ds_dir, ds_name)
    pred_attrs, discard_attrs, sens_attrs = update_attrs(
        ds_name,
        pred_attrs=pred_attrs,
        discard_attrs=discard_attrs,
        sens_attrs=sens_attrs,
    )
    if force_reprep or not os.path.exists(os.path.join(dataset_path, PROCESSED_DIR)):
        data_prep(
            ds_name,
            ds_dir,
            pred_attrs=pred_attrs,
            discard_attrs=discard_attrs,
            sens_attrs=sens_attrs,
            dataset_args=dataset_args,
        )

    if ds_name in GRAPH_DATASETS:
        return create_graph(ds_name, ds_dir, extra_folders="_".join(sens_attrs))
    else:
        dataset_path = os.path.join(ds_dir, ds_name)
        features, labels, sens_attr, _ = load_processed_dataset(
            dataset_path, "_".join(sens_attrs)
        )
        return features, labels, sens_attr


def get_edge_list(ds_name: str, ds_dir: str, graph: dgl.DGLGraph):
    if ds_name in FAIRNESS_DATASETS:
        return torch.load(os.path.join(ds_dir, ds_name, PROCESSED_DIR, EDGE_FILE))
    else:
        return torch.stack(graph.edges(), dim=0)


## ---------- ENEM Processing Files Below:
## Source:
## https://github.com/HsiangHsu/Fair-Projection/blob/main/fair-projection/enem/multi-group-multi-class/utils.py
def construct_grade(df, grade_attribute, n, all_label=False):
    if all_label:
        n = 2
        label = None
        for attr in grade_attribute:
            v = df[attr].values
            quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n + 1))
            if label is None:
                label = (pd.cut(v, quantiles, labels=np.arange(n)) == 1) * 1
            else:

                label += (pd.cut(v, quantiles, labels=np.arange(n)) == 1) * 1
        return label
    v = df[grade_attribute[0]].values
    quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n + 1))
    return pd.cut(v, quantiles, labels=np.arange(n))


def construct_race(df, protected_attribute):
    race_dict = {
        "Branca": 0,
        "Preta": 1,
        "Parda": 2,
        "Amarela": 3,
        "Indigena": 4,
    }  # changed to match ENEM 2020 numbering
    return df[protected_attribute].map(race_dict)


def load_enem(
    dataset_path,
    features,
    grade_attribute,
    n_sample,
    n_classes,
    group="race",
    multigroup=False,
    all_label=False,
):
    """
    For a given (X,y) features = X, grade_attribute = y, g is the group such that (X,y) \in g
    """
    if all_label:
        features = list(
            set(features)
            | set(["NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_CN"])
        )
        grade_attribute = ["NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_CN"]
    ## load csv
    df = pd.read_csv(
        os.path.join(dataset_path, RAW_DIR, "MICRODADOS_ENEM_2020.csv"),
        encoding="cp860",
        sep=";",
    )
    # print('Original Dataset Shape:', df.shape)

    ## Remove all entries that were absent or were eliminated in at least one exam
    ix = (
        ~df[["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]]
        .applymap(lambda x: False if x == 1.0 else True)
        .any(axis=1)
    )
    df = df.loc[ix, :]

    ## Remove "treineiros" -- these are individuals that marked that they are taking the exam "only to test their knowledge". It is not uncommon for students to take the ENEM in the middle of high school as a dry run
    df = df.loc[df["IN_TREINEIRO"] == 0, :]

    ## drop eliminated features
    df.drop(
        [
            "TP_PRESENCA_CN",
            "TP_PRESENCA_CH",
            "TP_PRESENCA_LC",
            "TP_PRESENCA_MT",
            "IN_TREINEIRO",
        ],
        axis=1,
        inplace=True,
    )

    ## subsitute race by names
    # race_names = ['N/A', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    race_names = [np.nan, "Branca", "Preta", "Parda", "Amarela", "Indigena"]
    df["TP_COR_RACA"] = (
        df.loc[:, ["TP_COR_RACA"]].applymap(lambda x: race_names[x]).copy()
    )

    ## remove repeated exam takers
    ## This pre-processing step significantly reduces the dataset.
    # df = df.loc[df.TP_ST_CONCLUSAO.isin([1,2])]
    df = df.loc[df.TP_ST_CONCLUSAO.isin([1])]

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variable
    df["gradebin"] = construct_grade(df, grade_attribute, n_classes, all_label)
    if multigroup:
        df["racebin"] = construct_race(df, "TP_COR_RACA")
    else:
        df["racebin"] = np.logical_or(
            (df["TP_COR_RACA"] == "Branca").values,
            (df["TP_COR_RACA"] == "Amarela").values,
        ).astype(int)
    df["sexbin"] = (df["TP_SEXO"] == "M").astype(int)

    df.drop(["TP_COR_RACA", "TP_SEXO"] + grade_attribute, axis=1, inplace=True)

    ## encode answers to questionaires
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ["Q00" + str(x) if x < 10 else "Q0" + str(x) for x in range(1, 25)]
    for q in question_vars:
        if q != "Q005":
            df_q = pd.get_dummies(df[q], prefix=q)
            df.drop([q], axis=1, inplace=True)
            df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## check if age range ('TP_FAIXA_ETARIA') is within attributes
    if "TP_FAIXA_ETARIA" in features:
        q = "TP_FAIXA_ETARIA"
        df_q = pd.get_dummies(df[q], prefix=q)
        df.drop([q], axis=1, inplace=True)
        df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## encode SG_UF_PROVA (state where exam was taken)
    df_res = pd.get_dummies(df["SG_UF_PROVA"], prefix="SG_UF_PROVA")
    df.drop(["SG_UF_PROVA"], axis=1, inplace=True)
    df = pd.concat([df, df_res], axis=1)

    df = df.dropna()
    ## Scaling ##
    scaler = MinMaxScaler()

    remove_set = set(["gradebin", "racebin"])
    if group == "gender":
        remove_set = set(["gradebin", "sexbin"])
    elif group == "both":
        remove_set = set(["gradebin", "racebin", "sex_bin"])

    scale_columns = list(set(df.columns.values) - remove_set)
    df[scale_columns] = pd.DataFrame(
        scaler.fit_transform(df[scale_columns]), columns=scale_columns, index=df.index
    )
    # print('Preprocessed Dataset Shape:', df.shape)

    if n_sample is not None:
        df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)
    # df['gradebin'] = df['gradebin'].astype(int)

    labels = df.pop("gradebin").to_numpy()
    sens = None
    if group == "race":
        sens = df.pop("racebin").to_numpy()
    elif group == "gender":
        sens = df.pop("sexbin").to_numpy()
    elif group == "both":
        temp = df.pop("racebin").to_numpy()
        num_vals = len(np.unique(temp))
        sens = temp + num_vals * df.pop("sexbin").to_numpy()

    # features are the remaining values in the df
    labels = torch.tensor(labels, dtype=torch.int64)
    sens = torch.tensor(sens, dtype=torch.float32)
    feat = torch.tensor(df.to_numpy(), dtype=torch.float32)
    save_processed_dataset(dataset_path, feat, labels, sens)
