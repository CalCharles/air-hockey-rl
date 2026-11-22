"""Synthetic update-only benchmark, with warmup excluded; no training jobs launched."""
import argparse
import json
import time
import numpy as np
import torch
from metamotivo.buffers.buffers import DictBuffer
from scripts.fb.fb_training import Args, create_agent
from scripts.fb.fast_update import FastFBUpdate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    for mode in ('full', 'current'):
        cfg = Args(backward_input=mode)
        replay = DictBuffer(4096, device='cuda')
        rng = np.random.default_rng(0)
        replay.extend({'observation': rng.normal(size=(4096,32)).astype('float32'),
                       'action': rng.uniform(-1,1,(4096,2)).astype('float32'),
                       'next': {'observation':rng.normal(size=(4096,32)).astype('float32'),
                                'terminated':np.zeros((4096,1), dtype=bool)}})
        for graph in (False, True):
            agent = create_agent(cfg,32,2)
            update = FastFBUpdate(agent, replay, use_graph=graph)
            update.update(10)
            torch.cuda.synchronize()
            start=time.perf_counter()
            values=update.update(args.updates)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            print(json.dumps({'backward':mode,'graph':graph,'updates_per_second':args.updates/elapsed,
                              'finite':all(torch.isfinite(v).all().item() for v in values.values())}),flush=True)
            del update,agent

if __name__=='__main__': main()
