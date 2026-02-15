"""End-to-end real-data system identification: paddle then puck.

Runs the paddle pipeline first to discover paddle physics parameters, then
feeds the optimal paddle params into the puck pipeline which optimizes puck
dynamics using free-space trajectory segments.

Each pipeline can also be configured independently via prefixed arguments
(--paddle-* and --puck-*). Shared arguments (data, dt, cfg) are specified once.

Usage:
    python scripts/domain_adaptation/system_identification/run_system_id.py \
        --data-dir /path/to/data/ \
        --trajectory-file trajectory_data100.hdf5 \
        --paddle-num-iterations 100 \
        --puck-num-iterations 80 \
        --puck-traj-length 12
"""

import argparse
import os

from scripts.domain_adaptation.system_identification.paddle.real_data_paddle_pipeline import (
    RealDataPaddlePipeline,
)
from scripts.domain_adaptation.system_identification.puck.real_data_puck_pipeline import (
    RealDataPuckPipeline,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end real-data system identification (paddle + puck)"
    )

    # ── Shared arguments ──────────────────────────────────────────
    shared = parser.add_argument_group("Shared")
    shared.add_argument("--cfg", type=str, default=None,
                        help="Optional YAML config (air_hockey key expected)")
    shared.add_argument("--data-dir", type=str,
                        default="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/",
                        help="Directory containing trajectory HDF5 files")
    shared.add_argument("--trajectory-file", type=str, default="trajectory_data.hdf5",
                        help="Trajectory HDF5 filename")
    shared.add_argument("--dt", type=float, default=0.05,
                        help="Timestep for velocity finite differences")
    shared.add_argument("--wdb-entity", type=str, default="",
                        help="W&B entity for logging (empty = disabled)")

    # ── Pipeline selection ────────────────────────────────────────
    selection = parser.add_argument_group("Pipeline selection")
    selection.add_argument("--skip-paddle", action="store_true",
                           help="Skip paddle pipeline (requires --paddle-params-path)")
    selection.add_argument("--skip-puck", action="store_true",
                           help="Skip puck pipeline (run paddle only)")
    selection.add_argument("--paddle-params-path", type=str, default=None,
                           help="Pre-computed paddle params YAML (skips paddle pipeline)")

    # ── Paddle pipeline arguments ─────────────────────────────────
    paddle = parser.add_argument_group("Paddle pipeline")
    paddle.add_argument("--paddle-sys-id-path", type=str,
                        default="scripts/domain_adaptation/sys_id_configs/real2sim/paddle_id_params.yaml",
                        help="Paddle parameter bounds YAML")
    paddle.add_argument("--paddle-object-name", type=str, default="paddle",
                        help="Object to compare: paddle, puck, or all")
    paddle.add_argument("--paddle-comp-type", type=str, default="posl2",
                        help="Comparison metric for paddle")
    paddle.add_argument("--paddle-num-resamples", type=int, default=200)
    paddle.add_argument("--paddle-traj-length", type=int, default=8)
    paddle.add_argument("--paddle-num-population", type=int, default=14)
    paddle.add_argument("--paddle-num-iterations", type=int, default=100)
    paddle.add_argument("--paddle-variance", type=float, default=0.25)
    paddle.add_argument("--paddle-simulation-iterations", type=int, default=1)
    paddle.add_argument("--paddle-run-dir", type=str,
                        default="scripts/domain_adaptation/system_identification/paddle/real2sim_runs")
    paddle.add_argument("--paddle-run-id", type=str, default="8")

    # ── Puck pipeline arguments ───────────────────────────────────
    puck = parser.add_argument_group("Puck pipeline")
    puck.add_argument("--puck-sys-id-path", type=str,
                      default="scripts/domain_adaptation/sys_id_configs/real2sim/puck_id_params.yaml",
                      help="Puck parameter bounds YAML")
    puck.add_argument("--puck-comp-type", type=str, default="posl2",
                      help="Comparison metric for puck")
    puck.add_argument("--puck-num-resamples", type=int, default=200)
    puck.add_argument("--puck-traj-length", type=int, default=8)
    puck.add_argument("--puck-num-population", type=int, default=14)
    puck.add_argument("--puck-num-iterations", type=int, default=100)
    puck.add_argument("--puck-variance", type=float, default=0.25)
    puck.add_argument("--puck-simulation-iterations", type=int, default=1)
    puck.add_argument("--puck-run-dir", type=str,
                      default="scripts/domain_adaptation/system_identification/puck/real2sim_runs")
    puck.add_argument("--puck-run-id", type=str, default="1")

    return parser


def run_paddle(args):
    """Run paddle pipeline and return path to optimal params."""
    print(f"\n{'#'*60}")
    print("  STAGE 1: Paddle System Identification")
    print(f"{'#'*60}\n")

    paddle_kwargs = {
        "cfg": args.cfg,
        "data_dir": args.data_dir,
        "trajectory_file": args.trajectory_file,
        "sys_id_path_paddle": args.paddle_sys_id_path,
        "object_name": args.paddle_object_name,
        "comp_type": args.paddle_comp_type,
        "dt": args.dt,
        "num_resamples": args.paddle_num_resamples,
        "traj_length": args.paddle_traj_length,
        "num_population": args.paddle_num_population,
        "num_iterations": args.paddle_num_iterations,
        "variance": args.paddle_variance,
        "simulation_iterations": args.paddle_simulation_iterations,
        "run_dir": args.paddle_run_dir,
        "run_id": args.paddle_run_id,
        "wdb_entity": args.wdb_entity,
    }

    pipeline = RealDataPaddlePipeline(**paddle_kwargs)
    pipeline.load_real_data()
    pipeline.run_simulation()

    return os.path.join(args.paddle_run_dir, str(args.paddle_run_id), "optimal_params.yaml")


def run_puck(args, paddle_params_path):
    """Run puck pipeline using optimal paddle params."""
    print(f"\n{'#'*60}")
    print("  STAGE 2: Puck System Identification")
    print(f"  Paddle params: {paddle_params_path}")
    print(f"{'#'*60}\n")

    puck_kwargs = {
        "cfg": args.cfg,
        "data_dir": args.data_dir,
        "trajectory_file": args.trajectory_file,
        "sys_id_path_puck": args.puck_sys_id_path,
        "paddle_params_path": paddle_params_path,
        "comp_type": args.puck_comp_type,
        "dt": args.dt,
        "num_resamples": args.puck_num_resamples,
        "traj_length": args.puck_traj_length,
        "num_population": args.puck_num_population,
        "num_iterations": args.puck_num_iterations,
        "variance": args.puck_variance,
        "simulation_iterations": args.puck_simulation_iterations,
        "run_dir": args.puck_run_dir,
        "run_id": args.puck_run_id,
        "wdb_entity": args.wdb_entity,
    }

    pipeline = RealDataPuckPipeline(**puck_kwargs)
    pipeline.load_real_data()
    pipeline.run_simulation()

    return os.path.join(args.puck_run_dir, str(args.puck_run_id), "optimal_params.yaml")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    paddle_params_path = args.paddle_params_path

    # Stage 1: Paddle
    if not args.skip_paddle:
        paddle_params_path = run_paddle(args)
    else:
        if paddle_params_path is None:
            parser.error("--paddle-params-path is required when using --skip-paddle")
        print(f"Skipping paddle pipeline, using params from: {paddle_params_path}")

    # Stage 2: Puck
    if not args.skip_puck:
        puck_params_path = run_puck(args, paddle_params_path)
    else:
        print("Skipping puck pipeline.")

    # Summary
    print(f"\n{'='*60}")
    print("  System Identification Complete")
    print(f"{'='*60}")
    if not args.skip_paddle:
        print(f"  Paddle params: {paddle_params_path}")
    if not args.skip_puck:
        print(f"  Puck params:   {puck_params_path}")
    print(f"{'='*60}")
