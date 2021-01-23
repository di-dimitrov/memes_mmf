import os
import json

import numpy as np
import torch

from PIL import Image

from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize



class MemesBinaryDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        print(config)
        assert (
            self._use_binary
        )
        
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        self._data_folder = self._data_dir
        
    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor
        
    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        
        if "input_ids" in processed_text:
            current_sample.update(processed_text)
            
        if "covid_memes" in sample_info['id']:
            id = int(sample_info['id'].split("covid_memes_")[1]) + 10000
        else:
            id = int(sample_info['id'].split("covid_memes_")[1])
        current_sample.id = torch.tensor(id, dtype=torch.int)

        
        if sample_info == 'not harmful':
            label = torch.tensor(0, dtype=torch.long)
        else:
            label = torch.tensor(1, dtype=torch.long)
        current_sample.targets = label
 
        current_sample.image = self.image_db[idx]["images"][0]

        return current_sample
        
class MemesNonBinaryDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        print(config)
        assert (
            self._use_non_binary
        )
        
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        self._data_folder = self._data_dir
        
    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor
        
    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        
        if "input_ids" in processed_text:
            current_sample.update(processed_text)
            
        if "covid_memes" in sample_info['id']:
            id = int(sample_info['id'].split("covid_memes_")[1]) + 10000
        else:
            id = int(sample_info['id'].split("covid_memes_")[1])
        current_sample.id = torch.tensor(id, dtype=torch.int)

        if sample_info == 'not harmful':
            label = torch.tensor(0, dtype=torch.long)
        elif  sample_info == 'somewhat harmful':
            label = torch.tensor(1, dtype=torch.long)
        else: 
            label = torch.tensor(2, dtype=torch.long)
        current_sample.targets = label
 
        
        current_sample.image = self.image_db[idx]["images"][0]

        return current_sample