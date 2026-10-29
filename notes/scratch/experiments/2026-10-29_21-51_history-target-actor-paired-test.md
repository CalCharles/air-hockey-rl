# Frozen target-actor comparison

Recorded 2026-10-29 16:51:47 -0500. A paired evaluation of the saved TD3 target actor is **running**, with no aggregate result yet. This diagnostic tests the existing Polyak-averaged weights, not a new training initialization.

Source `runs/td3/noisy_history_20261029_seed0/train/checkpoint_3600000/actor_target.pth`, SHA256`c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b`. The original training code creates actor/target networks, copies the random actor state into the target (`train_base_policy.py`), then updates the target by Polyak averaging during that same scratch run. It is not a pretrained or residual source. No new training starts from these weights.

The target differs from the online actor by parameter L2norm0.776621288631065. An inference-only bundle `target_actor_3m6_bundle/` links model.pth to actor_target.pth and shares unchanged saved args/config/RMA metadata. Frozen bundle loading succeeds and the canonical environment assertion passes. No actor outputs, PID/smoothing, sensing, geometry, contacts, limits or DR were altered. Deterministic target inference uses the same HistoryAgent entry point as online-actor evaluation.

[Locked provenance and hashes](../../../docs/juggling-history-target-actor-test-provenance.json). Paired test:32starts700000–700031,horizon20000,output `target_actor_3m6_paired_20k_eval/` under the run root. These are deliberately **reused** starts from E034, not independent acceptance. Online actor results on the same starts: mean13,099.8125,median14,790,14/32caps,18failures. No target result is claimed yet. Hypothesis: parameter averaging might reduce sensitivity to checkpoint variability; an improvement is unproven.

Reproduction: construct a new bundle with the four links described above, then:

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir NEW_TARGET_BUNDLE --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 700000 --gif-steps 0
```

Also reviewed replay configuration: this RMA/history path constructs its flat success replay with success_buffer_size1,000,000; the generic buffer_size100,000 field does not control its capacity. The existing code and launch log agree. This confirms the documented capacity, not a new finding about the cause of policy variability; no buffer change was made.

All three learners and fixed4M watcher remain live. No new GIF for an unfinished diagnostic. Checks: distinct saved weights, source lineage review, frozen hash, bundle reload/canonical guard, live test handle, links/images, anchors/TOCs and whitespace. No training or simulator implementation change and no new implementation tests.
