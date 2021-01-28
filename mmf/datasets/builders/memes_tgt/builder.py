import json
import logging
import math
import os
import zipfile

from collections import Counter

from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.datasets.builders.memes.dataset import (
MemesTgtFeatureDataset,
MemesTgtDataset
)

from mmf.utils.general import get_mmf_root

logger = logging.getLogger(__name__)

@registry.register_builder("memes_tgt")
class MemesTgtBuilder(MMFDatasetBuilder):
        # Init should call super().__init__ with the key for the dataset
    def __init__(
        self,
        dataset_name="memes_tgt",
        dataset_class=MemesTgtDataset,
        *args,
        **kwargs
        ):
            super().__init__(dataset_name, dataset_class, *args, **kwargs)
            self.dataset_class = MemesTgtDataset

    def build(self, config,*args, **kwargs):
        self.data_folder = os.path.join(
            get_mmf_root(), config.data_dir
        )
        super().build(config, *args, **kwargs)

    def load(self, config, dataset_type, *args, **kwargs):

        config = config

        if self._use_features:
            self.dataset_class = MemesTgtFeatureDataset

        
        self.dataset = super().load(config,dataset_type,*args,**kwargs)
        return self.dataset

    @classmethod
    def config_path(self):
        return "mmf/configs/datasets/memes_tgt/defaults.yaml"
        
    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(self.dataset_name + "_num_final_outputs", 2)