# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from torch import nn


@registry.register_model("simple")
class Simple(BaseModel):
    @dataclass
    class Config:
        in_dim: int = 4
        hidden_dim: int = 10
        out_dim: int = 10

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/simple.yaml"

    def build(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.config.in_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.out_dim),
        )

    def forward(self, sample_list):
        return {"scores": self.classifier(sample_list.input)}
