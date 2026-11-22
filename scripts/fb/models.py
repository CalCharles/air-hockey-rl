"""Air-hockey-specific backward observation selection, using upstream FB losses.

The selector lives inside B, so online/target B, mixed-goal sampling, and all
inference helpers share the same input restriction. Full observations still go
to F and the actor. BatchNorm is featurewise, so selecting after normalization
cannot leak historical features into B.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal

from torch import nn
from metamotivo import config_from_dict
from metamotivo.fb import FBAgent, FBAgentConfig, FBModel, FBModelConfig
from metamotivo.nn_models import build_backward

CURRENT_INDICES = (12, 13, 14, 27, 28, 29)


class CurrentObservationBackward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = build_backward(6, cfg.archi.z_dim, cfg.archi.b)

    def forward(self, obs):
        return self.network(obs[..., list(CURRENT_INDICES)])


@dataclass
class ModelConfig(FBModelConfig):
    backward_input: Literal["full", "current"] = "full"


@dataclass
class AgentConfig(FBAgentConfig):
    model: ModelConfig = field(default_factory=ModelConfig)


class AirHockeyFBModel(FBModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config_from_dict(kwargs, ModelConfig)
        if self.cfg.backward_input not in ("full", "current"):
            raise ValueError("backward_input must be full or current")
        if self.cfg.backward_input == "current":
            if self.cfg.obs_dim not in (30, 32):
                raise ValueError("Current backward inputs require the 30-feature history observation")
            self._backward_map = CurrentObservationBackward(self.cfg).to(self.cfg.device)
            self._backward_map.train(False).requires_grad_(False)


class AirHockeyFBAgent(FBAgent):
    def __init__(self, **kwargs):
        # Upstream constructor fixes the model class to FBModel. Only substitute
        # construction; training, optimization, target updates and save/load stay upstream.
        self.cfg = config_from_dict(kwargs, AgentConfig)
        self.cfg.train.fb_target_tau = float(min(max(self.cfg.train.fb_target_tau, 0), 1))
        self._model = AirHockeyFBModel(**asdict(self.cfg.model))
        self.setup_training()
        self.setup_compile()
        self._model.to(self.cfg.model.device)


def load_policy(path, device):
    cfg = json.loads((Path(path) / "config.json").read_text())
    cls = AirHockeyFBModel if "backward_input" in cfg else FBModel
    return cls.load(str(path), device=device)
