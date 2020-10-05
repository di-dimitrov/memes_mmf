# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import contextlib
import itertools
import json
import platform
import random
import socket
import tempfile
import unittest
from typing import Callable, List, Optional

import torch


def compare_tensors(a, b):
    return torch.equal(a, b)


def dummy_args(model="cnn_lstm", dataset="clevr"):
    args = argparse.Namespace()
    args.opts = [f"model={model}", f"dataset={dataset}"]
    args.config_override = None
    return args


def is_network_reachable():
    try:
        # check if host name can be resolved
        host = socket.gethostbyname("one.one.one.one")
        # check if host is actually reachable
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
    except OSError as e:
        if e.errno == 101:
            pass
    return False


NETWORK_AVAILABLE = is_network_reachable()
CUDA_AVAILBLE = torch.cuda.is_available()


def skip_if_no_network(testfn, reason="Network is not available"):
    return unittest.skipUnless(NETWORK_AVAILABLE, reason)(testfn)


def skip_if_no_cuda(testfn, reason="Cuda is not available"):
    return unittest.skipUnless(CUDA_AVAILBLE, reason)(testfn)


def skip_if_windows(testfn, reason="Doesn't run on Windows"):
    return unittest.skipIf("Windows" in platform.system(), reason)(testfn)


def skip_if_macos(testfn, reason="Doesn't run on MacOS"):
    return unittest.skipIf("Darwin" in platform.system(), reason)(testfn)


def compare_state_dicts(a, b):
    same = True
    same = same and (list(a.keys()) == list(b.keys()))
    if not same:
        return same

    for val1, val2 in zip(a.values(), b.values()):
        if isinstance(val1, torch.Tensor):
            same = same and compare_tensors(val1, val2)
        elif not isinstance(val2, torch.Tensor):
            same = same and val1 == val2
        else:
            same = False
        if not same:
            return same

    return same


@contextlib.contextmanager
def make_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield temp_dir.name
    finally:
        # Don't clean up on Windows, as it always results in an error
        if "Windows" not in platform.system():
            temp_dir.cleanup()


def build_random_sample_list():
    from mmf.common.sample import Sample, SampleList

    first = Sample()
    first.x = random.randint(0, 100)
    first.y = torch.rand((5, 4))
    first.z = Sample()
    first.z.x = random.randint(0, 100)
    first.z.y = torch.rand((6, 4))

    second = Sample()
    second.x = random.randint(0, 100)
    second.y = torch.rand((5, 4))
    second.z = Sample()
    second.z.x = random.randint(0, 100)
    second.z.y = torch.rand((6, 4))

    return SampleList([first, second])


DATA_ITEM_KEY = "test"


class NumbersDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples, data_item_key=DATA_ITEM_KEY):
        self.num_examples = num_examples
        self.data_item_key = data_item_key

    def __getitem__(self, idx):
        return {self.data_item_key: torch.tensor(idx, dtype=torch.float32)}

    def __len__(self):
        return self.num_examples


class SimpleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, 4)

    def forward(self, prepared_batch):
        batch = prepared_batch[DATA_ITEM_KEY]
        model_output = {"losses": {"loss": torch.sum(self.linear(batch))}}
        return model_output


def assertModulesEqual(mod1, mod2):
    for p1, p2 in itertools.zip_longest(mod1.parameters(), mod2.parameters()):
        return p1.equal(p2)


def search_log(log_file: str, search_condition: Optional[List[Callable]] = None):
    """Searches a log file for a particular search conditions which can be list
    of functions and returns it back

    Args:
        log_file (str): Log file in which search needs to be performed
        search_condition (List[Callable], optional): Search conditions in form of list.
            Each corresponding to a function to test a condition. Defaults to None.

    Returns:
        JSONObject: Json representation of the search line

    Throws:
        AssertionError: If no log line is found meeting the conditions
    """
    if search_condition is None:
        search_condition = {}

    lines = []

    with open(log_file) as f:
        lines = f.readlines()

    filtered_line = None
    for line in lines:
        line = line.strip()
        if "progress" not in line:
            continue
        info_index = line.find(" : ")
        line = line[info_index + 3 :]
        res = json.loads(line)

        meets_condition = True
        for condition_fn in search_condition:
            meets_condition = meets_condition and condition_fn(res)

        if meets_condition:
            filtered_line = res
            break

    assert filtered_line is not None, "No match for search condition in log file"
    return filtered_line
