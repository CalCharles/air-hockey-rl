"""Shared ``--agent`` kind normalization for rollout/eval entrypoints."""


def normalize_agent_kind(kind: str) -> str:
    """Accept underscore or hyphen spellings (e.g. ``sac_gcrl`` -> ``sac-gcrl``)."""
    return str(kind).strip().replace("_", "-")
