import json
import logging
import math
import os
import zipfile

from collections import Counter

from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.datasets.builders.propaganda.dataset import PropagandaTask3Dataset
from mmf.utils.general import get_mmf_root

logger = logging.getLogger(__name__)

@registry.register_builder("propaganda")
class PropagandaBuilder(MMFDatasetBuilder):
        # Init should call super().__init__ with the key for the dataset
    def __init__(
        self,
        dataset_name="propaganda",
        dataset_class=PropagandaTask3Dataset,
        *args,
        **kwargs
        ):
            super().__init__(dataset_name, dataset_class, *args, **kwargs)
            self.dataset_class = PropagandaTask3Dataset

    def build(self, config,*args, **kwargs):
        self.data_folder = os.path.join(
            get_mmf_root(), config.data_dir
        )
        super().build(config, *args, **kwargs)

    def load(self, config, dataset_type, *args, **kwargs):
        #self.dataset = PropagandaTask3Dataset(
        #    config, dataset, data_folder=self.data_folder
        #)
        
        self.dataset = super().load(config,dataset_type,*args,**kwargs)
        return self.dataset

    @classmethod
    def config_path(self):
        return "mmf/configs/datasets/propaganda/defaults.yaml"