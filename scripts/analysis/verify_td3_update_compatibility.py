"""Reproduce CPU eager default-update equivalence against commit c4c422b.

Run from the repository root; requires Git history. Uses fixed real networks and
inputs, compares outputs/gradients with and without CQL, then writes the JSON
verification artifact under docs/. It does not run an environment or train.
"""
import copy, hashlib, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone
import torch
from scripts.td3.tests.test_critic_action_context import updater
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.helper.q_network import TD3QNetwork
from scripts.td3.tests.test_absolute_target_agent import view


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    reference = subprocess.check_output(["git", "show", "c4c422b:scripts/td3/helper/td3_graphed_update.py"], text=True)
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "legacy_update.py"
        source.write_text(reference)
        spec = importlib.util.spec_from_file_location("legacy_update_reference", source)
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)
        rows = []
        for cql in (False, True):
            current = updater(False)
            current.gamma, current.h_eps, current.cql_alpha, current.cql_n_random = .9, .001, .1, 2
            current.actor = DeterministicAgent(view())
            current.actor_target = copy.deepcopy(current.actor)
            current.qfs = [TD3QNetwork(obs_dim=30, act_dim=2, hidden_layer_size=64, num_hidden_layers=2) for _ in range(2)]
            current.qfs_target = copy.deepcopy(current.qfs)
            current.actor_action_delta_penalty = .005
            old = object.__new__(legacy.GraphedTD3Update)
            old.__dict__ = copy.deepcopy(current.__dict__)
            batch = 16
            obs, next_obs = torch.randn(batch, 30), torch.randn(batch, 30)
            action, previous = torch.randn(batch, 2).tanh(), torch.randn(batch, 2).tanh()
            rewards = torch.randn(batch)
            dones = torch.arange(batch).remainder(3).eq(0).float()
            weights, noise = torch.rand(batch), torch.randn(batch, 2).clamp(-.5, .5) * .2
            random_actions = torch.rand(2 * batch, 2) * 2 - 1 if cql else None
            arguments = (obs, next_obs, action, previous, rewards, dones, weights, noise, random_actions)
            a, b = old._critic_core(*arguments), current._critic_core(*arguments)
            output_difference = max((x - y).abs().max().item() for x, y in zip(a, b))
            a[0].backward(); b[0].backward()
            gradient_difference = max((x.grad-y.grad).abs().max().item() for oq,nq in zip(old.qfs,current.qfs) for x,y in zip(oq.parameters(),nq.parameters()))
            a_actor, b_actor = old._actor_core(obs, previous), current._actor_core(obs, previous)
            actor_output_difference = max((x-y).abs().max().item() for x,y in zip(a_actor,b_actor))
            a_actor[0].backward(); b_actor[0].backward()
            actor_gradient_difference = max((x.grad-y.grad).abs().max().item() for x,y in zip(old.actor.parameters(),current.actor.parameters()))
            row = dict(cql=cql, critic_output_max_abs_difference=output_difference, critic_gradient_max_abs_difference=gradient_difference, actor_output_max_abs_difference=actor_output_difference, actor_gradient_max_abs_difference=actor_gradient_difference)
            assert all(value == 0. for key,value in row.items() if key != "cql"), row
            rows.append(row)
        result = dict(created=datetime.now(timezone.utc).isoformat(), reference_commit=subprocess.check_output(["git","rev-parse","c4c422b"],text=True).strip(), current_update_source_sha256=hashlib.sha256(Path("scripts/td3/helper/td3_graphed_update.py").read_bytes()).hexdigest(), mode="CPU eager; default critic_last_action=False; real 64-wide residual actor and twin critics; fixed torch seed 20261029", batch_size=16, checks=rows, fixture_parameters=dict(gamma=.9, h_transform_eps=.001, cql_alpha=.1, cql_n_random=2, actor_action_delta_penalty=.005), limitation="No CUDA graph equivalence claim")
        Path("docs/juggling-update-compatibility.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
