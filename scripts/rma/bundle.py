"""Save / load helpers for RMA checkpoints.

Phase 1 writes, next to the usual ``model.pth`` / ``args.yaml`` /
``config.yaml`` of a TD3 checkpoint dir, an ``rma_meta.json`` describing the
actor layout (obs / env-param / latent dims, encoder widths, the DR
normalisation ranges).  Phase 2 writes ``adaptation_module.pth`` containing
the state dict plus its own layout meta.  Everything needed to rebuild a
deployable policy is therefore in the run directory.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
import yaml

from scripts.rma.env_wrapper import EnvParamNormalizer
from scripts.rma.networks import AdaptationModule, HistoryActor, RMAActor

RMA_META_FILENAME = "rma_meta.json"
ADAPTATION_FILENAME = "adaptation_module.pth"


def build_rma_meta(actor, normalizer: EnvParamNormalizer, encoder_hidden, hidden_layer_size: int, num_hidden_layers: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "mode": "history" if isinstance(actor, HistoryActor) else "privileged",
        "obs_dim": int(actor.obs_dim),
        "act_dim": int(actor.act_dim),
        "latent_dim": int(actor.latent_dim),
        "use_last_action": bool(actor.use_last_action),
        "hidden_layer_size": int(hidden_layer_size),
        "num_hidden_layers": int(num_hidden_layers),
        "normalizer": normalizer.to_dict(),
    }
    if isinstance(actor, HistoryActor):
        meta.update(
            {
                "history_len": int(actor.history_len),
                "feature_dim": int(actor.feature_dim),
                "step_features": str(getattr(actor, "step_features", "latest_frame")),
                "embed_dim": int(actor.encoder.embed[0].out_features),
                "conv_channels": int(actor.encoder.convs[0].out_channels),
            }
        )
    else:
        meta.update({"env_param_dim": int(actor.env_param_dim), "encoder_hidden": [int(h) for h in encoder_hidden]})
    return meta


def write_rma_meta(out_dir: str, meta: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RMA_META_FILENAME)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def actor_from_meta(meta: Dict[str, Any]):
    if meta.get("mode", "privileged") == "history":
        actor = HistoryActor(
            obs_dim=int(meta["obs_dim"]),
            act_dim=int(meta["act_dim"]),
            history_len=int(meta["history_len"]),
            feature_dim=int(meta["feature_dim"]),
            latent_dim=int(meta["latent_dim"]),
            use_last_action=bool(meta["use_last_action"]),
            hidden_layer_size=int(meta["hidden_layer_size"]),
            num_hidden_layers=int(meta["num_hidden_layers"]),
            embed_dim=int(meta.get("embed_dim", 32)),
            conv_channels=int(meta.get("conv_channels", 32)),
        )
        actor.step_features = str(meta.get("step_features", "latest_frame"))
        return actor
    return RMAActor(
        obs_dim=int(meta["obs_dim"]),
        act_dim=int(meta["act_dim"]),
        env_param_dim=int(meta["env_param_dim"]),
        latent_dim=int(meta["latent_dim"]),
        use_last_action=bool(meta["use_last_action"]),
        hidden_layer_size=int(meta["hidden_layer_size"]),
        num_hidden_layers=int(meta["num_hidden_layers"]),
        encoder_hidden=tuple(int(h) for h in meta["encoder_hidden"]),
        action_scale=1.0,
        action_bias=0.0,
    )


def _resolve_phase1_dir(path: str) -> Tuple[str, str]:
    """Accept a run dir, a checkpoint dir or a model.pth; return (dir, model_path)."""
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path)), os.path.abspath(path)
    model_path = os.path.join(path, "model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"no model.pth under {path}")
    return os.path.abspath(path), model_path


def load_phase1(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a phase-1 RMA policy bundle.

    Returns dict(actor, meta, normalizer, args, config, dir, model_path).
    ``args.yaml`` / ``config.yaml`` are looked up in the checkpoint dir, then
    its parent (the run dir writes them at both levels).
    """
    ckpt_dir, model_path = _resolve_phase1_dir(path)
    meta_path = os.path.join(ckpt_dir, RMA_META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{meta_path} missing — is this an RMA phase-1 checkpoint?")
    with open(meta_path) as f:
        meta = json.load(f)
    actor = actor_from_meta(meta)
    state = torch.load(model_path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "actor" in state and isinstance(state["actor"], dict):
        state = state["actor"]
    actor.load_state_dict(state)
    actor.eval()

    def _find(name: str):
        for d in (ckpt_dir, os.path.dirname(ckpt_dir)):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                with open(p) as f:
                    return yaml.load(f, Loader=yaml.FullLoader)
        return None

    return {
        "actor": actor,
        "mode": meta.get("mode", "privileged"),
        "meta": meta,
        "normalizer": EnvParamNormalizer.from_dict(meta["normalizer"]),
        "args": _find("args.yaml"),
        "config": _find("config.yaml"),
        "dir": ckpt_dir,
        "model_path": model_path,
    }


def adaptation_meta(module: AdaptationModule, step_features: str, embed_dim: int, conv_channels: int, conv_kernels, conv_strides) -> Dict[str, Any]:
    return {
        "feature_dim": int(module.feature_dim),
        "history_len": int(module.history_len),
        "latent_dim": int(module.latent_dim),
        "step_features": str(step_features),
        "embed_dim": int(embed_dim),
        "conv_channels": int(conv_channels),
        "conv_kernels": [int(k) for k in conv_kernels],
        "conv_strides": [int(s) for s in conv_strides],
    }


def save_adaptation_module(out_dir: str, module: AdaptationModule, meta: Dict[str, Any], filename: str = ADAPTATION_FILENAME) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save({"state_dict": module.state_dict(), "meta": meta}, path)
    return path


def load_adaptation_module(path: str, map_location: str = "cpu") -> Tuple[AdaptationModule, Dict[str, Any]]:
    if os.path.isdir(path):
        path = os.path.join(path, ADAPTATION_FILENAME)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    meta = payload["meta"]
    module = AdaptationModule(
        feature_dim=int(meta["feature_dim"]),
        history_len=int(meta["history_len"]),
        latent_dim=int(meta["latent_dim"]),
        embed_dim=int(meta["embed_dim"]),
        conv_channels=int(meta["conv_channels"]),
        conv_kernels=tuple(meta["conv_kernels"]),
        conv_strides=tuple(meta["conv_strides"]),
    )
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module, meta
