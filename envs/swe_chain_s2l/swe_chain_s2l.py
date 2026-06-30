"""SWE dependency-chain integration task (Design C) for LOCA-bench.

Each target order's compute_total() must be implemented by TRACING a multi-hop
chain across modules and integrating the gathered values:

    order.GATEWAY  -> gateways/gateway_<g>.py : REGION
                   -> regions/region_<r>.py   : TAX_RATE, DEFAULT_POLICY
                   -> policies/policy_<p>.py   : DISCOUNT
    compute_total() == SUBTOTAL * (1 + TAX_RATE) - DISCOUNT

This resists the grep shortcut: there is no single pattern that yields the answer;
the agent must follow pointers and integrate several distributed values. Context
is grown by adding decoy (unreferenced) gateway/region/policy modules. The number
of TARGET orders is FIXED, so intrinsic work is constant while context scales.

Scoring imports each order module and calls compute_total() directly (no shell).
"""
import importlib.util
import random
import shutil
import string
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gem.core import Env

TERMINAL_STATE = "TASK_TERMINATED"
TOL = 1e-6

_FILLER = ("Internal note: this module carries historical migration and rationale "
           "commentary that does not affect runtime behavior. ")


def _filler_doc(n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    reps = max(1, (n_tokens * 4) // len(_FILLER))
    return _FILLER * reps


class SweChainS2LEnv(Env):
    def __init__(self, task_dir, n_decoys: int = 2, decoy_doc_tokens: int = 120,
                 seed: int = 42, n_targets: int = 6, **_):
        self.task_dir = task_dir
        self.agent_workspace = Path(task_dir) / "agent_workspace"
        self.proj = self.agent_workspace / "project"
        self.n_decoys = int(n_decoys)
        self.decoy_doc_tokens = int(decoy_doc_tokens)
        self.seed = int(seed)
        self.n_targets = int(n_targets)
        self.expected: Dict[int, float] = {}

    # ---- generation helpers -------------------------------------------------
    def _code(self, rng) -> str:
        return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(5))

    def _get_instructions(self) -> str:
        return (
            "You are working in a Python project under ./project .\n\n"
            "Some modules in project/orders/ define an order with attributes "
            "GATEWAY (str) and SUBTOTAL (number) and a function compute_total() "
            "that currently `raise NotImplementedError`.\n\n"
            "For each such order, implement compute_total() to return the order "
            "total by TRACING this chain of modules:\n"
            "  1. project/gateways/gateway_<GATEWAY>.py defines REGION (a string).\n"
            "  2. project/regions/region_<REGION>.py defines TAX_RATE (float) and "
            "DEFAULT_POLICY (a string).\n"
            "  3. project/policies/policy_<DEFAULT_POLICY>.py defines DISCOUNT (float).\n\n"
            "Then compute_total() must return exactly:\n"
            "    SUBTOTAL * (1 + TAX_RATE) - DISCOUNT\n"
            "using the TAX_RATE and DISCOUNT reached by following THIS order's "
            "GATEWAY through the chain above.\n\n"
            "Rules:\n"
            "- Do not change file names, the GATEWAY value, or SUBTOTAL.\n"
            "- Only implement orders whose compute_total() currently raises "
            "NotImplementedError.\n"
            "- Many gateway/region/policy modules are unrelated decoys; follow only "
            "the chain reachable from each order's own GATEWAY.\n\n"
            "When finished, call the claim_done tool."
        )

    def _write(self, relpath: str, content: str) -> None:
        p = self.proj / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        rng = random.Random(self.seed if seed is None else seed)
        if self.agent_workspace.exists():
            shutil.rmtree(self.agent_workspace, ignore_errors=True)
        for d in ("", "gateways", "regions", "policies", "orders"):
            (self.proj / d).mkdir(parents=True, exist_ok=True)
            (self.proj / d / "__init__.py").write_text("")

        used_codes = set()

        def new_code():
            while True:
                c = self._code(rng)
                if c not in used_codes:
                    used_codes.add(c)
                    return c

        # ---- M real chains: order -> gateway -> region -> policy ----------
        self.expected = {}
        for i in range(1, self.n_targets + 1):
            g, r, p = new_code(), new_code(), new_code()
            tax = round(rng.choice([0.05, 0.07, 0.1, 0.15, 0.2, 0.08]), 2)
            disc = float(rng.choice([5, 10, 15, 20, 25, 30]))
            subtotal = rng.randint(50, 500)
            self._write(f"policies/policy_{p}.py", f"DISCOUNT = {disc}\n")
            self._write(f"regions/region_{r}.py",
                        f"TAX_RATE = {tax}\nDEFAULT_POLICY = \"{p}\"\n")
            self._write(f"gateways/gateway_{g}.py", f"REGION = \"{r}\"\n")
            self._write(
                f"orders/order_{i:02d}.py",
                f"GATEWAY = \"{g}\"\nSUBTOTAL = {subtotal}\n\n\n"
                "def compute_total():\n"
                "    raise NotImplementedError(\"Trace the chain and integrate.\")\n",
            )
            self.expected[i] = subtotal * (1 + tax) - disc

        # ---- decoys: unreferenced gateway/region/policy triples -----------
        for _ in range(self.n_decoys):
            g, r, p = new_code(), new_code(), new_code()
            tax = round(rng.uniform(0.01, 0.25), 2)
            disc = float(rng.randint(1, 40))
            doc = _filler_doc(self.decoy_doc_tokens)
            self._write(f"policies/policy_{p}.py", f'"""{doc}"""\nDISCOUNT = {disc}\n')
            self._write(f"regions/region_{r}.py",
                        f'"""{doc}"""\nTAX_RATE = {tax}\nDEFAULT_POLICY = "{p}"\n')
            self._write(f"gateways/gateway_{g}.py", f'"""{doc}"""\nREGION = "{r}"\n')

        return self._get_instructions(), {}

    # ---- scoring ------------------------------------------------------------
    def _check(self):
        # Run in a subprocess with cwd=agent_workspace so that the agent's
        # compute_total() can resolve `import project.gateways...` etc. This also
        # isolates arbitrary agent code and avoids import caching across orders.
        import json
        import subprocess
        import sys

        ids = sorted(self.expected)
        runner = (
            "import json, importlib, sys, os\n"
            "sys.path.insert(0, os.getcwd())\n"
            f"ids = {ids!r}\n"
            "res = {}\n"
            "for i in ids:\n"
            "    try:\n"
            "        m = importlib.import_module('project.orders.order_%02d' % i)\n"
            "        res[i] = m.compute_total()\n"
            "    except Exception as e:\n"
            "        res[i] = {'__error__': repr(e)}\n"
            "print(json.dumps(res))\n"
        )
        runner_path = Path(self.task_dir) / f"_chk_{uuid.uuid4().hex[:8]}.py"
        runner_path.write_text(runner)
        got_map = {}
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path)],
                cwd=str(self.agent_workspace),
                capture_output=True, text=True, timeout=60,
            )
            got_map = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except Exception:
            got_map = {}
        finally:
            runner_path.unlink(missing_ok=True)

        correct = 0
        details = []
        for i in ids:
            got = got_map.get(str(i), got_map.get(i))
            ok = False
            if isinstance(got, (int, float)):
                ok = abs(float(got) - self.expected[i]) < TOL
            correct += int(ok)
            details.append({"order": f"order_{i:02d}",
                            "expected": round(self.expected[i], 4),
                            "got": got, "ok": ok})
        return correct, len(self.expected), details

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        correct, total, details = self._check()
        reward = (correct / total) if total else 0.0
        obs = (f"Evaluation: {correct}/{total} target orders correct "
               f"(reward={reward:.3f}).")
        info = {"n_correct": correct, "n_targets": total,
                "n_decoys": self.n_decoys, "details": details}
        return obs, reward, True, True, info
