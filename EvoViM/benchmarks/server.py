"""Run one benchmark in its own interpreter and exchange JSON lines with the host."""

import base64
import contextlib
import io
import importlib
import json
import sys
from pathlib import Path


BACKENDS = {
    "dmvhard2": "DMVHard2Backend",
    "gameworld": "GameWorldBackend",
    "mcgallery": "MCGalleryBackend",
    "postermaze": "PosterMazeBackend",
    "rngbench": "MatchingPairsBackend",
}


def dispatch(backend, request):
    op = request["op"]
    if op == "reset":
        observation = backend.reset(request["seed"])
        return {"observation": observation, "description": backend.describe(),
                "instruction": backend.instruction()}
    if op == "step":
        return backend.step(request["action"])
    raise ValueError(f"Unknown benchmark operation {op!r}")


def main(name, task, profile, horizon, variant):
    # Some upstream games print during import and reset. Reserve stdout for replies.
    wire = sys.stdout
    sys.path.insert(0, str(Path(__file__).resolve().parent / name))
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module(f"{name}_backend")
        backend = getattr(module, BACKENDS[name])(name, task, profile, int(horizon), variant=variant)
        try:
            for line in sys.stdin:
                request = json.loads(line)
                if request["op"] == "close":
                    break
                try:
                    result = dispatch(backend, request)
                    observation = result["observation"]
                    frame = io.BytesIO()
                    observation.pop("image").convert("RGB").save(frame, format="PNG")
                    observation["image_png_base64"] = base64.b64encode(frame.getvalue()).decode("ascii")
                    reply = {"result": result}
                except Exception as error:
                    reply = {"error": f"{type(error).__name__}: {error}"}
                print(json.dumps(reply, allow_nan=False), file=wire, flush=True)
        finally:
            backend.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
