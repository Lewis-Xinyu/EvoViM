"""Bridge paper environments to the public episode interface."""

from __future__ import annotations

import base64
import io
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .base import PublicObs, StepResult


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
PROFILES = {
    "dmvhard2": "dmvhard2_shop_image_v1",
    "gameworld": "gameworld_screenshot_v1",
    "mcgallery": "mcgallery_screenshot_v1",
    "postermaze": "postermaze_firstperson_image_v1",
    "rngbench": "rngbench_matching_image_v1",
}

# Give the benchmark only the asset locations it needs. The host owns model credentials.
RESOURCE_VARIABLES = ("EVOVIM_RNGBENCH_ROOT", "EVOVIM_GAMEWORLD_ROOT", "EVOVIM_DMV_ROOT",
                      "DMVHARD_BASE_DIR", "PLAYWRIGHT_BROWSERS_PATH", "PLAYWRIGHT_NODEJS_PATH")


def _observation(value):
    data = base64.b64decode(value["image_png_base64"], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        return PublicObs(value["text"], image.convert("RGB"), value["frame_id"])


class BenchmarkAdapter:
    public_text_fields = ("long_term_context", "short_term_context")
    has_image = True

    def __init__(self, name: str, config: dict):
        self.name = name
        self.task_id = config["task_id"]
        self.max_steps = int(config["max_steps"])
        self.observation_profile = config.get("observation_profile", PROFILES[name])
        python = Path(config.get("python") or sys.executable).expanduser().absolute()
        self.argv = [str(python), "-B", str(BENCHMARKS / "server.py"), name, self.task_id,
                     self.observation_profile, str(self.max_steps), config.get("variant", "")]
        self._process = None
        self._workdir = None

    def _start(self):
        self._workdir = TemporaryDirectory(prefix=f"evovim-{self.name}-")
        work = Path(self._workdir.name)
        env = {key: os.environ[key] for key in RESOURCE_VARIABLES if key in os.environ}
        env.update(HOME=str(work), TMPDIR=str(work), PATH=os.pathsep.join([str(Path(self.argv[0]).parent), os.defpath]),
                   PYTHONHASHSEED="0", LANG="en_US.UTF-8")
        self._errfile = work / "benchmark.stderr"
        with self._errfile.open("wb") as errors:
            self._process = subprocess.Popen(self.argv, cwd=work, env=env, stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=errors, start_new_session=True)

    def _call(self, op, **args):
        proc = self._process
        try:
            proc.stdin.write((json.dumps({"op": op, **args}) + "\n").encode())
            proc.stdin.flush()
            deadline = time.monotonic() + 120
            reply = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while not reply.endswith(b"\n"):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError(f"{self.name} timed out during {op}")
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise EOFError
                    reply.extend(chunk)
        except (BrokenPipeError, EOFError) as error:
            detail = self._errfile.read_text(errors="replace")[-2000:]
            raise RuntimeError(f"{self.name} exited during {op}: {detail}") from error
        packet = json.loads(reply)
        if "error" in packet:
            raise RuntimeError(packet["error"])
        return packet["result"]

    def reset(self, seed):
        if self._process is None:
            self._start()
        result = self._call("reset", seed=seed)
        self._last = _observation(result["observation"])
        self._description = result["description"]
        self._instruction = result["instruction"]
        actions = self.legal_actions()
        self.fallback_action = actions[0] if actions else "look"

    def observe(self):
        return self._last

    def task_instruction(self):
        return self._instruction

    def action_spec(self):
        return self._description["action_spec"]

    def legal_actions(self):
        return self.action_spec().get("actions", [])

    def metric_description(self):
        return self._description["metric"]

    def step(self, action):
        result = self._call("step", action=action)
        self._last = _observation(result["observation"])
        metrics = result["metrics"]
        terminated, truncated = result["terminated"], result["truncated"]
        return StepResult(self._last, terminated or truncated, result["reward"], metrics,
                          terminated, truncated, metrics, result.get("action_mapping"))

    def score(self, private_record):
        return float(private_record["primary"])

    def close(self):
        proc = self._process
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            finally:
                proc.stdout.close()
                self._process = None
        if self._workdir:
            self._workdir.cleanup()
            self._workdir = None
