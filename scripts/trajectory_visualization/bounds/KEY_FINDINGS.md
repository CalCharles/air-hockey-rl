# Key Findings - Trajectory Bounds Analysis

## 🎯 Quick Reference: Maximum Values

These are the **absolute maximum values** observed across all 417 trajectories (207,023 frames):

```
┌─────────────────────────────┬──────────────┬────────┐
│ Metric                      │ Maximum      │ Unit   │
├─────────────────────────────┼──────────────┼────────┤
│ Linear Velocity             │ 1.074393     │ m/s    │
│ Angular Velocity            │ 0.010541     │ rad/s  │
│ Acceleration                │ 30.985776    │ m/s²   │
│ Force                       │ 88.728680    │ N      │
│ Torque                      │ 11.541066    │ N⋅m    │
│ Paddle-to-Target Distance   │ 0.284864     │ m      │
└─────────────────────────────┴──────────────┴────────┘
```

## 📊 Key Observations

### 1. Velocity Characteristics
- **Maximum speed**: ~1.07 m/s (about 2.4 mph) - reasonable for air hockey gameplay
- **Typical speed**: ~0.24 m/s mean (about 0.5 mph) - most motion is moderate
- **Angular motion**: Very limited rotation (max 0.01 rad/s) - wrist stays stable

### 2. Force Profile
- **Impact forces**: Up to 88.7 N - likely puck collisions
- **Normal operation**: ~6.2 N mean - modest forces during play
- **Distribution**: Right-skewed - most forces are low, occasional spikes

### 3. Acceleration Patterns
- **Consistent baseline**: ~9.8 m/s² (1g) - gravity compensation active
- **Peak acceleration**: ~31 m/s² (3.2g) - rapid direction changes
- **Tight distribution**: Low variance suggests controlled motion

### 4. Control Performance
- **Typical error**: ~6.8 cm mean distance to target
- **Maximum error**: ~28.5 cm - some significant tracking deviations
- **Control quality**: Median ~6 cm suggests generally good tracking

## 🔧 Practical Applications

### For Simulation Bounds
Use these values to set realistic limits in simulated environments:

```python
# Recommended bounds (with 10% safety margin)
MAX_LINEAR_VELOCITY = 1.2      # m/s
MAX_ANGULAR_VELOCITY = 0.012   # rad/s
MAX_ACCELERATION = 35.0        # m/s²
MAX_FORCE = 100.0              # N
MAX_TORQUE = 13.0              # N⋅m
MAX_TRACKING_ERROR = 0.32      # m
```

### For Data Normalization
Use percentile values to avoid outlier sensitivity:

```python
# 99th percentile normalization (recommended)
NORM_LINEAR_VELOCITY = 0.764   # m/s
NORM_ANGULAR_VELOCITY = 0.0027 # rad/s
NORM_ACCELERATION = 11.45      # m/s²
NORM_FORCE = 17.93             # N
NORM_TORQUE = 1.804            # N⋅m
NORM_TRACKING_ERROR = 0.206    # m
```

### For Anomaly Detection
Values exceeding these thresholds may indicate anomalies:

```python
# 95th percentile thresholds
ANOMALY_VELOCITY = 0.554       # m/s
ANOMALY_FORCE = 10.686         # N
ANOMALY_ACCELERATION = 10.404  # m/s²
ANOMALY_DISTANCE = 0.155       # m
```

## 📈 Distribution Insights

### High-Frequency Ranges
Most data points fall within these ranges:

- **Velocity**: 0.05 - 0.40 m/s (68% of data within 1σ of mean)
- **Force**: 3.3 - 9.1 N (68% of data)
- **Acceleration**: 9.0 - 10.2 m/s² (tight clustering)
- **Distance**: 2.3 - 11.4 cm (68% of data)

### Outlier Characteristics
- **Velocity outliers**: > 0.76 m/s (1% of frames)
- **Force outliers**: > 17.9 N (1% of frames) - likely puck impacts
- **Acceleration outliers**: > 11.4 m/s² (1% of frames)
- **Distance outliers**: > 20.6 cm (1% of frames)

## 🎮 Gameplay Implications

### Robot Capabilities
- Can achieve speeds up to 1 m/s when needed
- Maintains stable wrist orientation (minimal rotation)
- Handles impact forces up to 89 N
- Can execute rapid direction changes (3g acceleration)

### Control Strategy
- Operates at ~25% of maximum velocity most of the time
- Maintains tracking accuracy within ~7 cm on average
- Occasionally deviates significantly (up to ~28 cm)
- Force control shows good modulation (6 N typical, 89 N max)

### Physics Validation
- Acceleration baseline matches expected gravity (9.8 m/s²)
- Force range consistent with puck mass and velocities
- Torque values reasonable for wrist-mounted paddle
- Distance errors align with control loop performance

## 📁 Files Generated

All analysis outputs are in `scripts/trajectory_visualization/bounds/`:

1. **`statistics.json`** - Complete statistics in JSON format
2. **`statistics_summary.txt`** - Human-readable summary
3. **`magnitude_frequency_distributions.png`** - Distribution plots (linear)
4. **`magnitude_frequency_distributions_log_scale.png`** - Distribution plots (log)
5. **`analyze_all_trajectories.py`** - Analysis script
6. **`ANALYSIS_README.md`** - Detailed documentation
7. **`KEY_FINDINGS.md`** - This file

## 🔄 Updating the Analysis

To re-run the analysis with new trajectory data:

```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
python scripts/trajectory_visualization/bounds/analyze_all_trajectories.py
```

The script automatically finds all trajectory files in `/nfs/data/airhockey/` and processes them.

## 📝 Notes

- Analysis date: November 29, 2025
- Trajectories analyzed: 417 (1 failed due to empty file)
- Total data points: 207,023 frames
- Data source: UR5e robot real-world trajectories
- Sampling rate: ~20 Hz (varies by trajectory)

---

*For detailed methodology and complete statistics, see `ANALYSIS_README.md`*


