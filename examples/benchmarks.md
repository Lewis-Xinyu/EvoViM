# Original benchmark environments

EvoViM includes five backends for the six paper tasks. Each runs in its own interpreter. Install just the assets for the task you want, keep the environment variables exported and use a fresh directory for each episode. The four-card demo requires none of the downloads below.

| Task | Upstream dependency | Step limit |
| --- | --- | ---: |
| Card Matching | RNGBench card renderer and faces | 360 |
| DMV-Hard | DMV-Bench image registry and base photos | 124 |
| Minecraft Gallery | GameWorld Minecraft Clone and Chromium | 60 |
| Poster Maze | RNGBench 3D maze renderer | 140 |
| Tetris | GameWorld Tetris and Chromium | 100 |
| Pac-Man | GameWorld Pac-Man and Chromium | 100 |

## Card Matching and Poster Maze

Run from the EvoViM checkout with its virtual environment active. The same RNGBench checkout serves both adapters. [Pinned RNGBench revision](https://github.com/InternLM/RNGBench/tree/4b33c24cdb660c3d417bb2ae90f31a1e5462b430)

```sh
mkdir -p upstream
git clone https://github.com/InternLM/RNGBench.git upstream/RNGBench
export EVOVIM_RNGBENCH_ROOT="$PWD/upstream/RNGBench"
git -C "$EVOVIM_RNGBENCH_ROOT" checkout 4b33c24cdb660c3d417bb2ae90f31a1e5462b430
evovim evaluate gpt/card-matching --smoke --output runs/cards-check
evovim evaluate gpt/poster-maze --smoke --output runs/maze-check
```

## DMV-Hard

Download the upstream base photographs as well as the registry. Set `DMVHARD_BASE_DIR` if the photos live outside the upstream checkout. [Pinned DMV-Bench revision](https://github.com/yyyujintang/DMV-Bench/tree/76b6512625041a665625aab86523d0f2b38627f2) · [Image dataset](https://huggingface.co/datasets/yyyujintang/DMV-Bench-Images)

```sh
mkdir -p upstream
git clone https://github.com/yyyujintang/DMV-Bench.git upstream/DMV-Bench
export EVOVIM_DMV_ROOT="$PWD/upstream/DMV-Bench"
git -C "$EVOVIM_DMV_ROOT" checkout 76b6512625041a665625aab86523d0f2b38627f2
python -m pip install huggingface_hub
python "$EVOVIM_DMV_ROOT/scripts/download_images.py" --with-base
evovim evaluate haiku/dmv-hard --smoke --output runs/dmv-check
```

## Minecraft Gallery, Tetris and Pac-Man

GameWorld serves a local browser game and captures each frame with Playwright. A separate Python 3.12 environment keeps its browser and Pillow versions independent of the EvoViM host. Use the same `--benchmark-python` option for full evaluations. [Pinned GameWorld runtime](https://github.com/gameworld-project/gameworld/tree/3c26bdab436800fd61ef40543b64ca40d12c7e4a) · [Pinned game library](https://github.com/gameworld-project/GameWorld-Games/tree/55322928fa8bd51cb1719bd3807a32634aa5d3cb)

```sh
mkdir -p upstream
git clone https://github.com/gameworld-project/gameworld.git upstream/GameWorld
export EVOVIM_GAMEWORLD_ROOT="$PWD/upstream/GameWorld"
git -C "$EVOVIM_GAMEWORLD_ROOT" checkout 3c26bdab436800fd61ef40543b64ca40d12c7e4a
git clone https://github.com/gameworld-project/GameWorld-Games.git "$EVOVIM_GAMEWORLD_ROOT/games/gameworld-games"
git -C "$EVOVIM_GAMEWORLD_ROOT/games/gameworld-games" checkout 55322928fa8bd51cb1719bd3807a32634aa5d3cb

python3.12 -m venv upstream/gameworld-env
GAME_PYTHON="$PWD/upstream/gameworld-env/bin/python"
"$GAME_PYTHON" -m pip install -r examples/gameworld-requirements.txt
export PLAYWRIGHT_BROWSERS_PATH="$PWD/upstream/browsers"
"$GAME_PYTHON" -m playwright install chromium
evovim evaluate gpt/minecraft-gallery --smoke --benchmark-python "$GAME_PYTHON" --output runs/gallery-check
evovim evaluate gpt/tetris --smoke --benchmark-python "$GAME_PYTHON" --output runs/tetris-check
evovim evaluate gpt/pac-man --smoke --benchmark-python "$GAME_PYTHON" --output runs/pacman-check
```

## Interpreting a rerun

The program ID fixes the code selected at the end of a lineage. Full evaluations use the paper task ID and step limit, 20 calls per LOG and PLAN phase and a 4,096-token completion limit. A Qwen run uses the archived sampling settings and expects a deployment that accepts `chat_template_kwargs.enable_thinking`. The repository distributes memory programs. Model access and benchmark assets come from the provider and the upstream projects.

The one-step checks use the first held-out seed and scripted model replies. In a macOS verification of all 18 programs, the initial observation texts and first model prompts matched the archive. The static benchmarks also reproduced the archived first image. Browser startup timing affected some Minecraft Gallery and Tetris screenshots at the same seed. A seeded browser run can therefore produce different frames before the first model action. [Archived program versions and episode scores](../EvoViM/programs/catalog.json)
