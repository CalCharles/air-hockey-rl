"""Capture the complete upstream FB update, including GPU replay sampling.

Losses and optimizers are upstream. Warmup/capture mutations are restored in
place, including BatchNorm buffers, targets, Adam moments and CUDA RNG. The
unused z-history buffer is omitted (use_mix_rollout must remain false).
"""
import gc

import torch
from metamotivo.nn_models import eval_mode, _soft_update_params
from scripts.td3.helper.td3_graphed_update import _snapshot_optimizer_state, _restore_optimizer_state


class FastFBUpdate:
    def __init__(self, agent, replay, use_graph=True, fused_adam=True):
        self.agent, self.replay = agent, replay
        self.device = torch.device(agent.device)
        self.use_graph = use_graph
        if use_graph and self.device.type != "cuda":
            raise ValueError("CUDA graphs require a CUDA agent")
        if agent.cfg.train.use_mix_rollout or agent.cfg.compile or agent.cfg.cudagraphs:
            raise ValueError("FastFBUpdate owns graph capture and requires use_mix_rollout=False")
        if fused_adam and self.device.type == 'cuda':
            for name in ('forward_optimizer', 'backward_optimizer', 'actor_optimizer'):
                old = getattr(agent, name)
                groups = [{**group, 'fused': True, 'foreach': False, 'capturable': True}
                          for group in old.param_groups]
                new = torch.optim.Adam(groups, fused=True, capturable=True)
                new.load_state_dict(old.state_dict())
                for group in new.param_groups:
                    group.update(fused=True, foreach=False, capturable=True)
                setattr(agent, name, new)
        self.optimizers = [agent.forward_optimizer, agent.backward_optimizer, agent.actor_optimizer]
        for opt in self.optimizers:
            for group in opt.param_groups:
                group['capturable'] = self.device.type == 'cuda'
                # Loading an eager CPU-step Adam checkpoint requires GPU step tensors.
                for param in group['params']:
                    if 'step' in opt.state[param]:
                        opt.state[param]['step'] = opt.state[param]['step'].to(self.device)
        self.size = torch.tensor(float(len(replay)), device=self.device)
        self.graph = None
        self.out = None

    def sample(self):
        indices = (torch.rand(self.agent.cfg.train.batch_size, device=self.device) * self.size).long()
        self.last_indices = indices  # Inspectable in tests; fresh draws on each graph replay.
        storage = self.replay.storage
        return {"observation": storage['observation'][indices], "action": storage['action'][indices],
                "next": {"observation": storage['next']['observation'][indices],
                         "terminated": storage['next']['terminated'][indices]}}

    def step_batch(self, batch):
        a = self.agent
        obs, action = batch['observation'], batch['action']
        next_obs = batch['next']['observation']
        discount = a.cfg.train.discount * ~batch['next']['terminated']
        a._model._obs_normalizer(obs)
        a._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(a._model._obs_normalizer):
            obs = a._model._obs_normalizer(obs)
            next_obs = a._model._obs_normalizer(next_obs)
        z = a.sample_mixed_z(train_goal=next_obs).clone()
        clip = a.cfg.train.clip_grad_norm or None
        result = a.update_fb(obs=obs, action=action, discount=discount, next_obs=next_obs,
                             goal=next_obs, z=z, q_loss_coef=a.cfg.train.q_loss_coef or None,
                             clip_grad_norm=clip)
        result.update(a.update_actor(obs=obs, action=action, z=z, clip_grad_norm=clip))
        with torch.no_grad():
            _soft_update_params(a._forward_map_paramlist, a._target_forward_map_paramlist, a.cfg.train.fb_target_tau)
            _soft_update_params(a._backward_map_paramlist, a._target_backward_map_paramlist, a.cfg.train.fb_target_tau)
        return result

    def _step(self):
        return self.step_batch(self.sample())

    def capture(self):
        gc.collect()
        refs = list(self.agent._model.state_dict().values())
        saved = [p.detach().clone() for p in refs]
        optim_states = [_snapshot_optimizer_state(opt) for opt in self.optimizers]
        rng = torch.cuda.get_rng_state(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = self._step()
            torch.cuda.synchronize(self.device)
        finally:
            with torch.no_grad():
                for ref, original in zip(refs, saved):
                    ref.copy_(original)
                for opt, original in zip(self.optimizers, optim_states):
                    _restore_optimizer_state(opt, original)
            torch.cuda.set_rng_state(rng, self.device)
            torch.cuda.synchronize(self.device)
        self.graph, self.out = graph, out

    def update(self, count=1):
        if len(self.replay) < self.agent.cfg.train.batch_size:
            raise ValueError("Insufficient replay for update")
        self.size.fill_(float(len(self.replay)))
        if self.use_graph:
            if self.graph is None:
                self.capture()
            for _ in range(count):
                self.graph.replay()
            return self.out
        for _ in range(count):
            self.out = self._step()
        return self.out
