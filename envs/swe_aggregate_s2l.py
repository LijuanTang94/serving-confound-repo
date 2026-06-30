"""SWE aggregation task for LOCA-bench — mechanism-ablation experiment.

The agent must compute a FIXED set of aggregate queries over ALL region modules
(which cannot be answered without reading every region — no targeted shortcut),
and write the results to ./answers.json. Context is grown via the number of
region modules (n_regions); the queries (semantics) stay fixed.

This env is used in two conditions, set purely by which MCP servers the config
enables:
  - "tools"  : python_execute ON  -> agent can write a loop over the files
               (externalize the aggregation to code) -> expected context-robust.
  - "reason" : python_execute OFF (filesystem read + write only) -> agent must
               integrate all region contents in-context -> expected to degrade
               as n_regions grows.

Scoring reads ./answers.json (no execution of agent code).
"""
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gem.core import Env

TERMINAL_STATE = "TASK_TERMINATED"
TOL = 1e-6
TIERS = ["gold", "silver", "bronze"]


class SweAggregateS2LEnv(Env):
    def __init__(self, task_dir, n_regions: int = 8, seed: int = 42, **_):
        self.task_dir = task_dir
        self.agent_workspace = Path(task_dir) / "agent_workspace"
        self.proj = self.agent_workspace / "project"
        self.n_regions = int(n_regions)
        self.seed = int(seed)
        self.answers: Dict[str, float] = {}

    def _get_instructions(self) -> str:
        return (
            "You are working in a Python project under ./project .\n\n"
            "project/regions/ contains many region modules. EACH region module "
            "defines three module-level variables: TAX_RATE (float), ACTIVE "
            "(bool), and TIER (one of \"gold\", \"silver\", \"bronze\").\n\n"
            "Compute the following aggregate queries over ALL region modules and "
            "write the answers to a file ./answers.json (a JSON object):\n"
            "  q1 = number of regions with ACTIVE == True\n"
            "  q2 = sum of TAX_RATE over regions with ACTIVE == True, rounded to 2 decimals\n"
            "  q3 = number of regions with TIER == \"gold\"\n"
            "  q4 = max TAX_RATE among regions with ACTIVE == False, rounded to 2 decimals "
            "(use 0 if there are none)\n\n"
            "Write exactly: {\"q1\": <int>, \"q2\": <float>, \"q3\": <int>, \"q4\": <float>} "
            "to ./answers.json (relative to the project root you are working in).\n\n"
            "You must consider EVERY region module — there is no shortcut. "
            "When finished, call the claim_done tool."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        rng = random.Random(self.seed if seed is None else seed)
        if self.agent_workspace.exists():
            shutil.rmtree(self.agent_workspace, ignore_errors=True)
        (self.proj / "regions").mkdir(parents=True, exist_ok=True)
        (self.proj / "__init__.py").write_text("")
        (self.proj / "regions" / "__init__.py").write_text("")

        regions = []
        for i in range(1, self.n_regions + 1):
            tax = round(rng.uniform(0.02, 0.25), 2)
            active = rng.random() < 0.6
            tier = rng.choice(TIERS)
            regions.append((tax, active, tier))
            (self.proj / "regions" / f"region_{i:03d}.py").write_text(
                f"TAX_RATE = {tax}\nACTIVE = {active}\nTIER = \"{tier}\"\n"
            )

        active_taxes = [t for (t, a, _) in regions if a]
        inactive_taxes = [t for (t, a, _) in regions if not a]
        self.answers = {
            "q1": sum(1 for (_, a, _) in regions if a),
            "q2": round(sum(active_taxes), 2),
            "q3": sum(1 for (_, _, tier) in regions if tier == "gold"),
            "q4": round(max(inactive_taxes), 2) if inactive_taxes else 0,
        }
        return self._get_instructions(), {}

    def _check(self):
        # search recursively: the agent may write answers.json under project/ etc.
        matches = sorted(self.agent_workspace.rglob("answers.json"),
                         key=lambda p: len(p.parts))
        got = {}
        if matches:
            try:
                got = json.loads(matches[0].read_text())
            except Exception:
                got = {}
        correct = 0
        details = []
        for q, exp in self.answers.items():
            g = got.get(q)
            ok = isinstance(g, (int, float)) and abs(float(g) - float(exp)) < 1e-2 + TOL
            correct += int(ok)
            details.append({"q": q, "expected": exp, "got": g, "ok": ok})
        return correct, len(self.answers), details

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        correct, total, details = self._check()
        reward = (correct / total) if total else 0.0
        obs = f"Evaluation: {correct}/{total} queries correct (reward={reward:.3f})."
        info = {"n_correct": correct, "n_queries": total,
                "n_regions": self.n_regions, "details": details}
        return obs, reward, True, True, info
