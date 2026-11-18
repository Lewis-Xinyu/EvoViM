# EvoViM

EvoViM runs a visual memory program for an agent. A task adapter supplies observations and actions, and a model decides what to store, recall and do next. The package includes one general-purpose starting program. You can also load a complete program that was selected for your own task.

## Install and try it

In a Python 3.10+ virtual environment on macOS or Linux, run these commands from this checkout.

```sh
python -m pip install .
evovim demo
```

The demo runs offline. A scripted model saves an image from the grid, recalls it on the next step and ends the episode. The output reports the actions, score and visual requests. The trace is temporary by default.

```sh
evovim demo --output ./grid-trace
```

## Run a selected memory program

Pass a trusted program directory containing `candidate.py`, the files it imports and, if provided, its `source_manifest.json`. An existing experiment's retained `lines/M/parent/` directory can be copied on its own. This demo exercises the program on the built-in grid. Evaluating it on its original task requires that task's adapter and model.

```sh
evovim demo --bundle /path/to/program
```

For another domain, call `run_episode` with an adapter, a model implementing `generate(call) -> ModelReply`, and the path to your chosen program. The adapter translates its inputs into a `PublicObs` with `long_term_context`, `short_term_context` and an optional image. It defines legal actions, state transitions and scoring. The grid demo is a working example of this interface. [Demo](visual_rsi/demo.py) · [Adapter contract](visual_rsi/adapters/base.py) · [Episode runner](visual_rsi/host/episode.py)

## Evolve a program

The optional evolution command revises the program using episode traces and evaluates each candidate before selecting it. Each new run keeps the state it needs to resume. The included grid configuration uses macOS Seatbelt for worker isolation, a signed-in Codex CLI for revisions and a multimodal Chat Completions endpoint for the episode model.

Set `OPENAI_API_KEY` in your environment and substitute your model identifier below. You can set `EVOVIM_EPISODE_BASE_URL` for a compatible endpoint. This command calls the configured models.

```sh
export EVOVIM_EPISODE_MODEL="your-vision-model-id"
export EVOVIM_META_WORKSPACE="$(mktemp -d /tmp/evovim-workspace.XXXXXX)"
evovim run --manifest examples/gridmemory.yaml --run-root runs/grid
```

The temporary workspace holds the files passed to the revision model. Keep its path when resuming this run. The selected program is written to `runs/grid/environments/gridmemory/lines/M/parent/`. The episode runner can load it directly. [Example configuration](examples/gridmemory.yaml)

The starting scaffold derives from AutoMem commit `1987b8f6d202212ec3c0c78ab2a01ae02aca4e8b`. [Upstream AutoMem](https://github.com/autoLearnMem/AutoMem/tree/1987b8f6d202212ec3c0c78ab2a01ae02aca4e8b)
