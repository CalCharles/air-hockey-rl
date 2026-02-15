import h5py
import numpy as np
from pathlib import Path

# ---- input file ----
HDF5_PATH = Path(
    "/data2/air_hockey/air_hockey_state_data/"
    "datastor1/calebc/public/data/mouse/state_data_all_new/"
    "trajectory_data2.hdf5"
)


def inspect_file(hdf5_path: Path):
    print(f"Opening: {hdf5_path}\n")

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())

        print("Top-level keys:")
        print(keys)
        print()

        # --- basic dataset info ---
        T = f[keys[0]].shape[0]
        print(f"Timesteps (T): {T}\n")

        print("Dataset shapes:")
        for k in keys:
            ds = f[k]
            print(f"{k:14s} shape={ds.shape}, dtype={ds.dtype}")

        # --- paddle inspection ---
        if "paddle" not in f:
            print("\n❌ No paddle dataset found")
            return

        paddle = f["paddle"][:]  # load once
        print("\n=== Paddle inspection ===")
        print(f"paddle shape: {paddle.shape}")

        labels = ["x", "y", "z", "rx", "ry", "rz"]

        for i, name in enumerate(labels):
            col = paddle[:, i]
            print(
                f"{name:>2s}: "
                f"min={col.min(): .4f}, "
                f"max={col.max(): .4f}, "
                f"mean={col.mean(): .4f}, "
                f"std={col.std(): .6f}"
            )


if __name__ == "__main__":
    inspect_file(HDF5_PATH)
