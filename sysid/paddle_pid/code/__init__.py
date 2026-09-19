"""Paddle (UR5 + PID plant) system identification from scripted paddle-motion recordings.

Everything paddle-sysid lives here: loading the recorded trials, the train / validation
split, the Box2D replay + per-step position-error metric, the CMA-ES fit of the PID gains,
and the reports. See ``sysid/paddle_pid/code/README.md``.
"""
