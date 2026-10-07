#!/usr/bin/env python3
"""Dump the exact `{side_task_goals}` text the paper's live monitor prompt uses,
per environment, from the pinned control-tower 1.0.1 registry.

Must run inside the linuxarena-paper venv (needs control_tower + the pulled
`.environments/` registry):

    cd ../benchmark/linuxarena-paper && UV_FROZEN=1 uv run python \
        "<this repo>/experiments/logit-curve-replication/data_export/dump_side_task_goals.py"

Writes experiments/<slug>/gpu/side_task_goals.json:
    {"control_tower_version": ..., "registry_lock": {...},
     "goals_text": {env_id: "<rendered block>"},
     "goals": {env_id: [{"id":..., "goal":...}, ...]}}

The rendered block is produced by control-tower's own
`get_side_task_goals(trajectory)` so it is byte-for-byte what
`build_action_prompt` substitutes. `gpu/format_monitor_prompt.py` (pure Python)
reads `goals_text` instead of importing control-tower.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parents[1].name
REPO = HERE.parents[3]
# Checkout of the LinuxArena paper repo; override with LINUXARENA_PAPER.
PAPER = Path(os.environ.get("LINUXARENA_PAPER", REPO.parent / "benchmark" / "linuxarena-paper"))
sys.path.insert(0, str(PAPER / "src"))

from control_tower.environments.all_environments import (  # noqa: E402
    ENVIRONMENTS,
    get_side_tasks_for_env,
)
from control_tower.protocols.monitoring.prompt_builder_helpers import (  # noqa: E402
    get_side_task_goals,
)
from control_tower.trajectories.trajectory import Trajectory  # noqa: E402

PUBLIC_ENVS = [
    "auto_workflow", "clinical_trial", "file_indexing", "media_processing", "model_registry",
    "shared_dotfiles", "slack_clone", "tiktok", "user_tracking", "web_scraping",
]


def main() -> None:
    goals_text: dict[str, str] = {}
    goals: dict[str, list[dict]] = {}
    for env_id in PUBLIC_ENVS:
        assert env_id in ENVIRONMENTS, f"{env_id} not in registry; run `uv run ct registry pull linuxarena`"
        traj = Trajectory.model_construct(
            trajectory_id="_probe", environment=ENVIRONMENTS[env_id], actions=[],
        )
        goals_text[env_id] = get_side_task_goals(traj)
        goals[env_id] = [
            {"id": tid, "goal": (t.goal.strip() if t.goal else None), "failure_type": getattr(t, "failure_type", None)}
            for tid, t in sorted(get_side_tasks_for_env(env_id).items())
        ]
        # cross-check the rendering against the helper's documented recipe
        expect = "\n\n".join(f"## {g['id']}\n{g['goal'] or 'No goal defined.'}" for g in goals[env_id])
        assert expect == goals_text[env_id], env_id

    lock = (PAPER / ".environments.lock.yml").read_text()
    out = {
        "control_tower_version": importlib.metadata.version("control-tower"),
        "control_tower_pin": "git 699d5ca5c26ab3126fa29c6fa73dbf898aa0ca53 (linuxarena-paper pyproject.toml)",
        "registry_lock_yml": lock,
        "recipe": "sorted(side_tasks_for_env.items()); '## {id}\\n{goal.strip()}' joined by '\\n\\n'; "
                  "'No side tasks are defined for this environment.' if empty",
        "goals_text": goals_text,
        "goals": goals,
    }
    p = REPO / "experiments" / SLUG / "gpu" / "side_task_goals.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for env_id in PUBLIC_ENVS:
        print(f"{env_id:18s} {len(goals[env_id]):2d} side tasks, {len(goals_text[env_id]):6d} chars")
    print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
