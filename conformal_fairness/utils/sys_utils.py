import os
import sys
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import List, Union

import lightning.pytorch as L
import psutil
from dgl.dataloading import DataLoader

from ..constants import CPU_AFF


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def dl_affinity_setup(dl: DataLoader, avl_affinities: Union[List[int], None] = None):
    # setup cpu affinity for dgl dataloader
    # TODO: multi node issues
    if avl_affinities is None:
        avl_affinities = psutil.Process().cpu_affinity()
    assert avl_affinities is not None, "No available cpu affinities"

    cx = getattr(dl, CPU_AFF)
    cx_fn = partial(
        cx,
        loader_cores=avl_affinities[: dl.num_workers],
        compute_cores=avl_affinities[dl.num_workers :],
        verbose=False,
    )
    # cx_fn = partial(cx, verbose=False)
    return cx_fn


def enter_cpu_cxs(
    datamodule: L.LightningDataModule, dl_strs: List[str], stack: ExitStack
):
    """Enter cpu contexts on stack and return dataloaders"""
    dls = []
    avl_affinities = psutil.Process().cpu_affinity()
    with suppress_stdout():
        for dl_str in dl_strs:
            dl = getattr(datamodule, dl_str)()
            stack.enter_context(dl_affinity_setup(dl, avl_affinities)())
            dls.append(dl)
    return dls
