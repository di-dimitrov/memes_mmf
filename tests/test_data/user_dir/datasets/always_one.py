# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder


class AlwaysOneDataset(torch.utils.data.Dataset):
    DATASET_LEN = 20

    def __init__(self, dataset_type="train", *args, **kwargs):
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset_name = "always_one"

    def __len__(self):
        return self.DATASET_LEN

    def __getitem__(self, idx):
        sample = Sample()
        sample.input = torch.tensor([idx, idx, idx, idx], dtype=torch.float)
        sample.targets = torch.tensor(1, dtype=torch.long)
        return sample


@registry.register_builder("always_one")
class AlwaysOneBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("always_one")

    def build(self, config, dataset_type="train"):
        pass

    @classmethod
    def config_path(cls):
        return "configs/always_one.yaml"

    def load(self, config, dataset_type="train", *args, **kwargs):
        return AlwaysOneDataset(dataset_type)
