import copy
import logging
import os
import re

from omegaconf import OmegaConf

from agents.base import BaseAgent
from client import LLMResponse
from prompt_builder.history import Message

from agents.board_parser import decode_frame, detect_events, format_board_block

logger = logging.getLogger(__name__)

MAX_LOG_ITERATIONS = 5
MAX_VALIDATE_ITERATIONS = 5
MAX_SEARCH_RESULTS = 100          

FALLBACK_ACTIONS = {
    "nle": "search", "minihack": "search", "crafter": "noop",
}
FALLBACK_ACTION_DEFAULT = "wait"

_DIRECTION_COMMANDS = frozenset([
    "open", "close", "kick", "fight", "fire", "zap",
])
_ITEM_COMMANDS = frozenset([
    "eat", "quaff", "wear", "wield", "read", "apply", "puton", "remove",
    "takeoff", "dip", "rub", "invoke", "quiver", "throw", "offer", "drop",
    "force", "loot", "cast", "pickup",
])
_TWO_STEP_COMMANDS = _DIRECTION_COMMANDS | _ITEM_COMMANDS
_ALL_DIRECTIONS = frozenset([
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
])

def repair_combined_action(raw_action):
    stripped = raw_action.strip()
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return stripped, False
    cmd = parts[0].lower()
    if cmd == "far" and parts[1].strip().lower() in _ALL_DIRECTIONS:
        return stripped, False
    if cmd in _TWO_STEP_COMMANDS:
        return cmd, True
    return stripped, False

NLE_GAME_RULES = """\
ACTION FORMAT RULES — Read this before acting!

MOVEMENT: Output a single direction word: north, south, east, west, northeast, northwest, southeast, southwest. These move exactly 1 tile. You can also output 'far north', 'far east', far south', 'far west', 'far northeast', 'far southeast', 'far southwest', 'far northwest' to sprint in that direction until hitting an obstacle (wall, monster, item, door). 

MELEE COMBAT: To attack an adjacent monster, simply MOVE INTO its tile. Example: if a goblin is to your east, output 'east' to attack it. This is the primary way to fight. Do NOT output 'fight east'.

FIGHT COMMAND: 'fight' is ONLY for attacking a monster you believe is there but CANNOT see (invisible/hidden). After outputting 'fight', the game will ask "In what direction?" — respond with JUST the direction.

RANGED COMBAT:
- fire: shoots your quivered ammunition. Two-step: output 'fire', then the direction when prompted. Uses whatever is in your quiver automatically.
- quiver: select ammunition for your quiver. Two-step: output 'quiver', then the item letter when prompted. Do this before using 'fire'.
- throw: throw any item. Three-step: output 'throw', then the item letter, then the direction. Works for daggers, darts, potions, etc.
- zap: use a wand. Multi-step: output 'zap', then the wand letter, then the direction (for directional wands like fire/cold/death). Some wands do not ask for direction.

DOORS: Output 'open'. The game then asks "In what direction?" — respond with the direction word. For locked doors, try 'kick' then direction.

STAIRS: You must be standing ON the stairs symbol. Then output 'up' for '<' stairs, or 'down' for '>' stairs. IMPORTANT: At Dungeon Level 1 (Dlvl:1), the '<' staircase leads OUT of the dungeon and ENDS THE GAME. Your goal is to go DEEPER — find '>' stairs down. Only go 'up' at Dlvl:1 if you want to quit.

EATING: Output 'eat'. The game asks "What do you want to eat? [abc or ?*]". Respond with JUST the single letter of the food item. Do NOT combine them — 'eat d' is INVALID; first 'eat', then 'd'.

ALL ITEM COMMANDS: eat, quaff, wear, wield, read, apply, puton, remove, takeoff, dip, rub, invoke — these are ALL two-step: first output the command word alone, then respond with the item letter when the game prompts you. NEVER combine command + letter in one action.

PICKING UP ITEMS: Items on the floor are collected AUTOMATICALLY when you walk onto them — you will see a message assigning an inventory letter. Gold ($) is always collected instantly. You do NOT need to output 'pickup' in normal play. Use 'pickup' only if multiple items are on one tile and you need to choose, or to retrieve something you previously dropped. Use 'loot' to take items from containers (boxes, chests) on the floor.

SEARCHING: Output 'search' to look for hidden doors/passages. You may need to search the same spot multiple times.

DIRECTION PROMPTS: When the game asks "In what direction?", respond with a direction word (north, south, east, west, northeast, etc.).

ITEM PROMPTS: When the game asks "What do you want to X? [abc or ?*]", respond with JUST the single letter of the item.

PRAYING: In desperate situations (very low HP, starving), 'pray' can save you. But do not pray more than once every ~300 turns or your god may get angry.

ENGRAVING: Writing 'Elbereth' on the ground can scare away many monsters. Use 'engrave' then follow the prompts.
"""

NLE_SYMBOLS = """\
MAP SYMBOLS — Reference for reading the ASCII map.

TERRAIN:
  @ = you (the player)
  . = floor / open ground (walkable)
  # = corridor (walkable)
  (space) = rock / solid stone (CANNOT walk through)
  - = horizontal wall (CANNOT walk through)
  | = vertical wall (CANNOT walk through)
  + = closed door (use 'open' + direction) OR spellbook on floor
  < = staircase up (use 'up' when standing on it)
  > = staircase down (use 'down' when standing on it)
  ^ = trap (DANGEROUS — try to avoid stepping on it)
  } = water / moat / lava (dangerous, may drown or burn)
  { = fountain (can 'quaff' from it)
  _ = altar (can 'offer' sacrifices)
  \\ = throne
  ` = boulder

ITEMS ON FLOOR:
  $ = gold (auto-collected by walking on it)
  ) = weapon
  [ = armor / shield
  ! = potion
  ? = scroll
  / = wand
  = = ring
  " = amulet
  ( = tool
  * = gem or rock
  % = food or corpse (can eat)
  ~ = iron chain or whip

COMMON MONSTER LETTERS (lowercase = small, UPPERCASE = large):
  a = ant / insect
  b = blob
  d = canine (dog, jackal, wolf)
  e = floating eye — do NOT melee attack (paralyzes you!)
  f = feline (cat, panther, tiger)
  g = gremlin / gnome (lowercase)
  h = humanoid (dwarf, mind flayer)
  i = imp / minor demon
  k = kobold
  o = orc
  r = rodent (rat, rock mole)
  s = spider / scorpion
  u = horse / unicorn
  w = worm
  D = dragon (very dangerous)
  G = gnome (capital)
  H = giant
  L = lich (very dangerous undead)
  M = mummy
  N = naga
  O = ogre
  T = troll (regenerates — eat corpse to gain regeneration)
  Z = zombie
  & = demon
  ; = sea creature
  : = lizard-like (newt, gecko)
  ' = golem
  @ = human (could be shopkeeper or guard — do NOT attack shopkeepers!)

PET INDICATORS:
  Your pet appears as the same letter as its monster type but the language observation will say 'tame' (e.g., 'tame little dog').
"""

NLE_MEMORY_SYSTEM_PROMPT = """\
You are an agent playing NetHack. The following are the valid actions you can output, one at a time:

MOVEMENT: north, east, south, west, northeast, southeast, southwest, northwest (move exactly 1 tile). Also: far north, far east, far south, far west, far northeast, far southeast, far southwest, far northwest (sprint in that direction until hitting an obstacle — wall, monster, item, or door).

STAIRS: up (go up, must be standing on '<'), down (go down, must be standing on '>').

UTILITY: wait, search, look, pickup, open (then direction), close (then direction), kick (then direction), fight (attack unseen monster, then direction), inventory.

RANGED: fire (shoots quivered ammo, then direction), throw (then item letter, then direction), zap (then wand letter, then direction for directional wands), quiver (select ammo, then item letter).

ITEM USAGE (all two-step — output the command, then respond with the item letter when prompted; NEVER combine into one action): eat, quaff, wear, wield, read, apply, puton, remove, takeoff, takeoffall, drop, droptype, dip, rub, invoke, force, loot, offer, pay, swap, twoweapon, untrap, wipe.

SPELLS / SPECIAL: cast, engrave, enhance, annotate, call, jump, monster, teleport, overview, pray.

MENU / PROMPT RESPONSES: esc (exit menu), minus, space, apos, 0-9 (for responding to in-game prompts).

CRITICAL ACTION RULES:
- To attack a visible monster in MELEE: simply MOVE into its tile (e.g. output 'north' if monster is north). Do NOT use 'fight' for visible monsters.
- To attack at RANGE: first 'quiver' to select ammo, then 'fire' + direction. Or 'throw' + item letter + direction. Or 'zap' + wand letter + direction.
- Item commands (eat, quaff, wield, etc.) are ALWAYS two-step: first the command word (e.g. 'eat'), then the item letter (e.g. 'd') when the game prompts. 'eat d' as one action is INVALID.
- Items on the floor are auto-collected when you walk onto them. Gold is always instant. You do NOT need 'pickup' in normal play.
- When the game asks "In what direction?", respond with a direction word.
- When the game asks "What do you want to X? [abc or ?*]", respond with JUST the single item letter.
- Consult game_rules.txt in your memory files for detailed action format rules.
- Consult nethack_symbols.txt in your memory files when you see an unfamiliar symbol on the ASCII map.
- Both reference files can be updated — APPEND new knowledge you discover.

READING THE OBSERVATION:
- The language observation describes entities around you with numeric step distances. '1 step away (directly reachable)' means you can reach it in exactly 1 move.
- The MAP ANALYSIS section (if present) shows passable/blocked directions computed programmatically. TRUST IT — it is accurate.
- The ASCII map shows the dungeon layout. Find '@' (yourself) and study adjacent tiles before moving. Refer to nethack_symbols.txt for symbols.
- Read the status line every turn: watch HP, hunger status, and conditions. If Hungry or Weak, eat. If HP is low, consider retreat or healing.
- Check your dungeon level (Dlvl). At Dlvl:1, going UP exits the dungeon and ends the game. Always go DOWN to progress.

Your goal is to get as far as possible in the game.

PLAY!"""

MINIHACK_MEMORY_SYSTEM_PROMPT = """\
You are an agent playing MiniHack. The following are the valid actions you can output, one at a time:

MOVEMENT: north, east, south, west, northeast, southeast, southwest, northwest (move exactly 1 tile).

OTHER ACTIONS: open (open a door, game asks direction — respond with direction word), kick (kick something, game asks direction), search (search for hidden doors/passages).

CRITICAL ACTION RULES:
- To attack a visible monster: simply MOVE into its tile. Do NOT try to combine actions.
- 'open' is two-step: first 'open', then the direction when prompted.
- Items on the floor are auto-collected when you walk onto them. Gold is always instant.
- Consult game_rules.txt in your memory files for detailed action format rules.
- Consult nethack_symbols.txt for map symbol meanings.
- Both files can be updated — APPEND new knowledge you discover.

READING THE OBSERVATION:
- The MAP ANALYSIS section (if present) shows passable/blocked directions. TRUST IT — it is accurate.
- The language observation uses numeric step distances: '1 step away (directly reachable)' means exactly 1 move away. Study these carefully before choosing a direction.
- The ASCII map shows the level. Find '@' (yourself). '<' is stairs up, '>' is stairs down, '-' and '|' are walls, '.' is floor, '#' is corridor. Do NOT move into walls or rock (spaces).
- Read the status line: watch HP and hunger.
- Repeating the same action when the observation does not change is pointless — try something different.

Your goal is to explore the level and reach the stairs down.

PLAY!"""

INITIAL_FILES_BY_ENV = {
    "crafter": {
        "map_notes.txt": "",
        "crafting_progress.txt": "",
        "survival_log.txt": "",
        "actions_log.txt": "",
        "strategy.txt": "",
    },
    "nle": {
        "game_rules.txt": NLE_GAME_RULES,
        "nethack_symbols.txt": NLE_SYMBOLS,
        "dungeon_map.txt": "",
        "monster_encounters.txt": "",
        "inventory.txt": "",
        "actions_log.txt": "",
        "strategy.txt": "",
    },
    "minihack": {
        "game_rules.txt": NLE_GAME_RULES,
        "nethack_symbols.txt": NLE_SYMBOLS,
        "dungeon_map.txt": "",
        "actions_log.txt": "",
        "inventory.txt": "",
        "strategy.txt": "",
    },
}

INITIAL_FILES_DEFAULT = {
    "exploration.txt": "",
    "actions_log.txt": "",
    "inventory.txt": "",
    "strategy.txt": "",
}

SPATIAL_GUIDANCE = {
    "crafter": (
        "SPATIAL AWARENESS: Carefully read the terrain description around you. "
        "Track what surrounds you (water, lava, trees, stone, ore, etc.) and in "
        "which direction. In map_notes.txt, log your position and nearby features "
        "in a structured way, e.g.: 'Pos (3,5): N=tree, E=stone, S=water, W=grass'. "
        "Track what you have crafted and what you still need in crafting_progress.txt. "
        "Note: the 'Do' action interacts with what is directly in front of you — if "
        "nothing interactable is there, 'Do' will have NO EFFECT.\n\n"
        "STATUS CHECK: Before choosing an action, read your status (health, food, "
        "drink, energy). If any resource is getting low, prioritize addressing it. "
        "Dying from starvation or thirst wastes all your progress."
    ),
    "nle": (
        "SPATIAL AWARENESS: The ASCII map shows the dungeon layout. Refer to "
        "nethack_symbols.txt in your memory for what each symbol means. '@' is "
        "you — find it on the map and study what surrounds it in each direction. "
        "BEFORE choosing a direction, verify on the map that the destination tile "
        "is passable (floor '.', corridor '#', or open door). Do NOT move into "
        "walls ('-', '|') or solid rock (space). Record STATIC features (walls, "
        "stairs, doors, corridors) in dungeon_map.txt using structured format: "
        "'(x,y): floor | N:wall | E:door | S:corridor | W:dark'.\n\n"
        "DYNAMIC ENTITIES: Monsters and pets MOVE every turn. Do NOT record their "
        "positions as permanent landmarks. Rely on the CURRENT observation for "
        "monster/NPC locations.\n\n"
        "STATUS CHECK: Read the status line carefully every turn. Watch for "
        "Hunger (Hungry/Weak/Fainting — eat immediately), low HP (consider "
        "retreat, healing, or prayer), and status conditions.\n\n"
        "ACTION FORMAT: Many actions are multi-step (command, then item/direction "
        "when prompted). Consult game_rules.txt in your memory if unsure about "
        "the correct format for any action. To attack a monster in melee, simply "
        "move into its tile."
    ),
    "minihack": (
        "SPATIAL AWARENESS: The ASCII map shows the level layout. Refer to "
        "nethack_symbols.txt in your memory for what each symbol means. '@' is "
        "you — find it on the map and study what surrounds it. '<' is stairs up, "
        "'>' is stairs down. BEFORE choosing a direction, verify on the map that "
        "the target tile is passable — do NOT move into '-' (wall), '|' (wall), "
        "or solid rock (space characters). Record STATIC features in dungeon_map.txt: "
        "'(x,y): floor | N:wall | E:stairs_up | S:corridor | W:dark'.\n\n"
        "DYNAMIC ENTITIES: Monsters and pets MOVE every turn. Do NOT record their "
        "positions as permanent landmarks. Rely on the CURRENT observation for "
        "monster/NPC locations.\n\n"
        "STATUS CHECK: Read the status line every turn. Watch for Hunger and "
        "low HP.\n\n"
        "ACTION FORMAT: Many actions are multi-step. Consult game_rules.txt in "
        "your memory if unsure. To attack a monster, simply move into its tile."
    ),
}

SPATIAL_GUIDANCE_DEFAULT = ("SPATIAL AWARENESS: Pay careful attention to spatial descriptions and any map or grid in the observation. Log positions and landmarks in a structured format so you can navigate effectively.")

LOG_SYSTEM = """\
You are playing a game and maintaining your external memory.

{action_description}

{diff_notice}

{spatial_guidance}

{map_analysis}

ENVIRONMENT RESPONSE TO YOUR PREVIOUS ACTION:
{observation}

YOUR MEMORY FILES:
{file_index}

AVAILABLE COMMANDS:
  <|SEARCH|>filename.txt|keyword<|END|>    Search a file for lines matching a keyword
  <|READ|>filename.txt<|END|>              Read an entire file
  <|TAIL|>filename.txt|N<|END|>            Read the last N lines of a file
  <|COUNT|>filename.txt|keyword<|END|>     Count lines matching a keyword
  <|APPEND|>filename.txt|Your note<|END|>  Add a line to a file
  <|WRITE|>filename.txt|Content<|END|>     Overwrite a file entirely
  <|CREATE|>new_file.txt<|END|>            Create a new empty file
  <|DONE|>                                 Finish logging, move on to action selection

IMPORTANT RULES:
1. BEFORE appending or writing to ANY file, first SEARCH or READ it to check whether the information is already recorded. Do NOT duplicate entries.
2. When logging, attribute the entry to the action that caused it. Format: "Step N action 'X': outcome description"
3. Not everything needs logging — skip routine movements with no new information. But DO log:
   - Failed or invalid actions (e.g. "You can't go there")
   - UNCHANGED observations (the action had NO EFFECT — this is important!)
   - Discoveries (new rooms, items, stairs, doors, monsters)
   - Inventory changes (picked up / lost items)
   - Damage taken or dealt
   - Unexpected outcomes
   - Important spatial information (coordinates, landmarks, obstacles)

You may issue multiple commands per turn. Finish with <|DONE|>."""

LOG_CONTINUE = """\
Command results:
{results}

Updated files:
{file_index}

Continue logging (remember: SEARCH before writing to avoid duplicates), or output <|DONE|> to move on to action selection."""

PLAN_SYSTEM = """\
{instruction_prompt}

You have an external memory system to help you make better decisions. Before committing to an action, you should search your memory for relevant information.

{spatial_guidance}

{stuckness_warning}

{map_analysis}

YOUR MEMORY FILES:
{file_index}

CURRENT OBSERVATION:
{observation}

PROCESS — follow these steps:
1. Carefully read the observation. If there is an ASCII map or grid, study it to understand what is adjacent to you in each direction.
2. Check your status (health, hunger, resources) — address urgent needs first.
3. Think about what action you want to take.
4. BEFORE committing, SEARCH your memory for information related to your planned action, current location, or situation. For example:
   - If you plan to move "east", search dungeon_map/exploration for "east" or the current coordinates to check for walls/obstacles.
   - If you plan to use an item or interact, search game_rules.txt for the correct action format.
   - If you see an unfamiliar symbol on the map, search nethack_symbols.txt.
5. Review what memory tells you. If it reveals your planned action is problematic (failed before, hits a wall, wrong format), REVISE your plan and search again for the new candidate.
6. When you are confident, commit with:
   <|ACTION|>your_chosen_action<|END|>

AVAILABLE COMMANDS (read-only — no writing in this phase):
  <|SEARCH|>filename.txt|keyword<|END|>    Search for lines matching keyword
  <|READ|>filename.txt<|END|>              Read entire file
  <|TAIL|>filename.txt|N<|END|>            Read last N lines of a file
  <|COUNT|>filename.txt|keyword<|END|>     Count matching lines
  <|ACTION|>your_chosen_action<|END|>      Commit to an action

You MUST eventually output exactly one <|ACTION|> to commit.
Think step by step: propose -> search -> validate -> commit or revise."""

PLAN_CONTINUE = """\
Search results:
{results}

Based on these results, decide:
- If your planned action looks safe, commit: <|ACTION|>your_action<|END|>
- If memory warns against it, revise your plan, search for the new action, then commit.

You can issue more SEARCH/READ/TAIL/COUNT commands, or commit with <|ACTION|>."""

_OPEN = {
    "READ":   r"<\|?READ\|?>",
    "SEARCH": r"<\|?SEARCH\|?>",
    "TAIL":   r"<\|?TAIL\|?>",
    "COUNT":  r"<\|?COUNT\|?>",
    "APPEND": r"<\|?APPEND\|?>",
    "WRITE":  r"<\|?WRITE\|?>",
    "CREATE": r"<\|?CREATE\|?>",
    "ACTION": r"<\|?ACTION\|?>",
    "DONE":   r"<\|?DONE\|?>",
}
_END = r"<\|?END\|?[>}]"

_ACTION_PATTERNS = [
    re.compile(r'<\|?ACTION\|?>(.*?)<\|?END\|?[>}]', re.DOTALL | re.IGNORECASE),
    re.compile(r'<ACTION>(.*?)</ACTION>', re.DOTALL | re.IGNORECASE),
    re.compile(r'(?:^|\n)\s*ACTION:\s*([^\n]{1,40})\s*(?:\n|$)', re.IGNORECASE),
]

class MemoryFileSystem:
    def __init__(self, root_dir, initial_files=None, max_search_results=MAX_SEARCH_RESULTS):
        self.root_dir = root_dir
        self.max_search_results = max_search_results  
        os.makedirs(self.root_dir, exist_ok=True)
        if initial_files:
            for name, content in initial_files.items():
                self._write_raw(self._safe_name(name), str(content))

    def _safe_name(self, filename):
        name = os.path.basename(filename.replace("\\", "/"))
        name = name.replace("..", "")
        if not name:
            name = "unnamed.txt"
        return name

    def _path(self, filename):
        return os.path.join(self.root_dir, filename)

    def _exists(self, filename):
        return os.path.isfile(self._path(filename))

    def _read_raw(self, filename):
        path = self._path(filename)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_raw(self, filename, content):
        with open(self._path(filename), "w", encoding="utf-8") as f:
            f.write(content)

    def _list_files(self):
        if not os.path.isdir(self.root_dir):
            return []
        return sorted(
            f for f in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, f))
        )

    def read(self, filename):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY]"
        return content

    def search(self, filename, keyword):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY — no matches]"
        matches = [
            line for line in content.splitlines()
            if keyword.lower() in line.lower()
        ]
        if not matches:
            return "[No lines matching '{}']".format(keyword)
        
        total = len(matches)
        if total > self.max_search_results:
            matches = matches[-self.max_search_results:]
            return "[Showing last {} of {} matches for '{}']\n{}".format(
                self.max_search_results, total, keyword, "\n".join(matches),
            )
        return "\n".join(matches)

    def tail(self, filename, n=10):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY]"
        lines = content.splitlines()
        tail_lines = lines[-n:]
        return "\n".join(tail_lines)

    def count(self, filename, keyword):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[0 matches for '{}' — file is empty]".format(keyword)
        matches = [
            line for line in content.splitlines()
            if keyword.lower() in line.lower()
        ]
        return "[{} line(s) matching '{}']".format(len(matches), keyword)

    def append(self, filename, content):
        filename = self._safe_name(filename)
        path = self._path(filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content.rstrip("\n") + "\n")
        return "[Appended to {}]".format(filename)

    def write(self, filename, content):
        filename = self._safe_name(filename)
        if content.strip():
            self._write_raw(filename, content.rstrip("\n") + "\n")
        else:
            self._write_raw(filename, "")
        return "[Wrote to {}]".format(filename)

    def create(self, filename):
        filename = self._safe_name(filename)
        if self._exists(filename):
            return "[File '{}' already exists]".format(filename)
        self._write_raw(filename, "")
        return "[Created {}]".format(filename)

    def file_index(self):
        files = self._list_files()
        if not files:
            return "  (no files yet)"
        lines = []
        for name in files:
            content = self._read_raw(name) or ""
            n_lines = len(content.splitlines()) if content.strip() else 0
            lines.append("  - {} ({} lines)".format(name, n_lines))
        return "\n".join(lines)

    def snapshot(self):
        result = {}
        for name in self._list_files():
            result[name] = self._read_raw(name) or ""
        return result

    def reset(self, initial_files=None):
        import shutil
        if os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        if initial_files:
            for name, content in initial_files.items():
                self._write_raw(self._safe_name(name), str(content))

_DIRECTION_OFFSETS = {
    "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1),
    "northeast": (-1, 1), "northwest": (-1, -1),
    "southeast": (1, 1), "southwest": (1, -1),
}
_TILE_PASSABLE = set(".#{_")
_TILE_WALL = set("-|")
_TILE_DANGEROUS = set("}^")
_TILE_ITEM = set("$)[!?/=\"(*%~(")

def _classify_tile(char):
    if char == ">":
        return "stairs_down"
    elif char == "<":
        return "stairs_up"
    elif char == "+":
        return "door"
    elif char in _TILE_WALL:
        return "wall"
    elif char == " ":
        return "rock"
    elif char in _TILE_DANGEROUS:
        return "dangerous"
    elif char in _TILE_PASSABLE:
        return "passable"
    elif char in _TILE_ITEM:
        return "item"
    elif char.isalpha() and char != "@":
        return "monster"
    elif char == "$":
        return "item"
    else:
        return "unknown"

def _offset_to_direction(row_offset, col_offset):
    parts = []
    if row_offset < 0:
        parts.append("north")
    elif row_offset > 0:
        parts.append("south")
    if col_offset > 0:
        parts.append("east")
    elif col_offset < 0:
        parts.append("west")
    return "".join(parts) if parts else "here"

def parse_nle_map(obs_text):
    lines = obs_text.split("\n")
    player_row_idx = None
    player_col = None

    for i, line in enumerate(lines):
        at_col = line.find("@")
        if at_col < 0:
            continue
        stripped = line.rstrip()
        if len(stripped) < 3:
            continue
        alpha_count = sum(1 for c in stripped if c.isalpha() and c != "@")
        if len(stripped) > 0 and alpha_count / len(stripped) < 0.5:
            player_row_idx = i
            player_col = at_col
            break

    if player_row_idx is None:
        return ""

    map_rows = []
    player_map_row = 0
    for i in range(max(0, player_row_idx - 15), min(len(lines), player_row_idx + 15)):
        line = lines[i]
        stripped = line.rstrip()
        if not stripped:
            continue
        alpha_count = sum(1 for c in stripped if c.isalpha() and c != "@")
        total = max(len(stripped), 1)
        if alpha_count / total < 0.55 or i == player_row_idx:
            map_rows.append((i, stripped))

    for j, (orig_idx, _text) in enumerate(map_rows):
        if orig_idx == player_row_idx:
            player_map_row = j
            break

    analysis = []
    passable_dirs, blocked_dirs, dangerous_dirs = [], [], []
    stairs_dirs, monster_dirs, door_dirs = [], [], []

    for direction, (dr, dc) in _DIRECTION_OFFSETS.items():
        r = player_map_row + dr
        c = player_col + dc
        if 0 <= r < len(map_rows) and 0 <= c < len(map_rows[r][1]):
            char = map_rows[r][1][c]
            tile_type = _classify_tile(char)
            if tile_type == "stairs_down":
                stairs_dirs.append("{} → '>' STAIRS DOWN — go there!".format(direction))
                passable_dirs.append("{} (stairs down '>')".format(direction))
            elif tile_type == "stairs_up":
                stairs_dirs.append("{} → '<' stairs up".format(direction))
                passable_dirs.append("{} (stairs up '<')".format(direction))
            elif tile_type == "door":
                door_dirs.append("{} → '+' closed door (use 'open')".format(direction))
                passable_dirs.append("{} (door '+')".format(direction))
            elif tile_type in ("passable", "item"):
                passable_dirs.append("{} ('{}')".format(direction, char))
            elif tile_type == "wall":
                blocked_dirs.append("{} (wall '{}')".format(direction, char))
            elif tile_type == "rock":
                blocked_dirs.append("{} (rock/void)".format(direction))
            elif tile_type == "dangerous":
                dangerous_dirs.append("{} → '{}' DANGEROUS".format(direction, char))
            elif tile_type == "monster":
                monster_dirs.append("{} → '{}' monster (move into tile to attack)".format(direction, char))
                passable_dirs.append("{} (monster '{}')".format(direction, char))
            else:
                blocked_dirs.append("{} ('{}')".format(direction, char))
        else:
            blocked_dirs.append("{} (edge/void)".format(direction))

    stairs_locations = []
    for j, (_orig_idx, row_text) in enumerate(map_rows):
        for c_idx, ch in enumerate(row_text):
            if ch == ">":
                row_offset = j - player_map_row
                col_offset = c_idx - player_col
                dist = max(abs(row_offset), abs(col_offset))
                if dist > 1:
                    stairs_locations.append("'>' seen ~{} tiles {} on map".format(
                        dist, _offset_to_direction(row_offset, col_offset)))

    analysis.append("=== MAP ANALYSIS ===")
    if stairs_dirs:
        analysis.append("*** STAIRS ADJACENT: " + "; ".join(stairs_dirs))
    if stairs_locations:
        analysis.append("*** STAIRS ON MAP: " + "; ".join(stairs_locations[:3]))
    if monster_dirs:
        analysis.append("MONSTERS ADJACENT: " + "; ".join(monster_dirs))
    if door_dirs:
        analysis.append("DOORS: " + "; ".join(door_dirs))
    if dangerous_dirs:
        analysis.append("DANGER: " + "; ".join(dangerous_dirs))
    if passable_dirs:
        analysis.append("CAN MOVE: " + ", ".join(passable_dirs))
    if blocked_dirs:
        analysis.append("BLOCKED: " + ", ".join(blocked_dirs))
    return "\n".join(analysis)

def parse_log_calls(text):
    calls = []
    has_done = bool(re.search(_OPEN["DONE"], text))

    for m in re.finditer(_OPEN["READ"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("READ", {"filename": fn}))

    for m in re.finditer(_OPEN["SEARCH"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("SEARCH", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["TAIL"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            try:
                n = int(parts[1].strip())
                calls.append(("TAIL", {"filename": parts[0].strip(), "n": n}))
            except ValueError:
                pass

    for m in re.finditer(_OPEN["COUNT"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("COUNT", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["APPEND"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("APPEND", {
                "filename": parts[0].strip(),
                "content": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["WRITE"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip():
            calls.append(("WRITE", {
                "filename": parts[0].strip(),
                "content": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["CREATE"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("CREATE", {"filename": fn}))

    return calls, has_done

def parse_plan_calls(text):
    calls = []
    committed_action = None

    for m in re.finditer(_OPEN["READ"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("READ", {"filename": fn}))

    for m in re.finditer(_OPEN["SEARCH"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("SEARCH", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["TAIL"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            try:
                n = int(parts[1].strip())
                calls.append(("TAIL", {"filename": parts[0].strip(), "n": n}))
            except ValueError:
                pass

    for m in re.finditer(_OPEN["COUNT"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("COUNT", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for pattern in _ACTION_PATTERNS:
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip()
            if candidate and len(candidate) < 60:
                committed_action = candidate
                break

    return calls, committed_action

# ---------------------------------------------------------------------------
# Tetris-line support.  The board is decoded from the current public
# screenshot by agents.board_parser; derived state (JSON-safe) is shown to
# the model in every request and recorded in stats.txt as a one-line event
# log.  The LLM keeps full control of its memory files through the tools.
# ---------------------------------------------------------------------------

TETRIS_STRATEGY = """\
TETRIS POLICY (general; verify against the live BOARD block each step)
Goal: reach the target score stated in the task while never top-ing out.

SCORING (rates verified from this game's own observations)
- hard_drop: 2 points per row dropped, locks the piece immediately.
- soft_drop: 1 point per row (drops at most ~8 rows).
- move_left/move_right/rotate/wait: 0 points; every action still applies
  one gravity row, so they only cost time.
- A line clear pays a bonus and lowers the stack (none observed in play
  yet; do not chase clears with extra moves).
- A top-out resets the score to 0: survival dominates point-greed.

PLACEMENT
- The SUGGESTED PLAN is a slide of at most 4 moves plus hard_drop that
  maximizes the raw drop distance of the current piece. Commit its first
  action each step; the plan is recomputed from the pixels every step.
- Never wait, soft_drop or rotate: they pay nothing and cost a gravity row.
- If no active piece is detected, the piece is resting on the stack or is
  still entering from the top: commit hard_drop (it locks the resting
  piece or drops the entering one).

SURVIVAL
- Watch "headroom" in the BOARD block. If it is small, keep committing the
  plan: it already prefers the deepest available column.
- After a restart the in-game score starts at 0, but your best score so far
  still counts for the task - play the new game as well.

MEMORY LAYOUT
- stats.txt: auto event log (pieces, landings, restarts) with (est) score
  estimates. Written by the system - never duplicate it in your own notes.
- strategy.txt: this policy, then your tagged strategic notes written with
  APPEND_NOTE (tags: plan, threat, mistake, progress, reset).
- plan.txt / notes.txt: ephemeral one-liners (LOG_NOTE) - short-lived
  observations only.
- The board block, your score estimate and the last lines of these files are
  shown in every prompt automatically. READ/SEARCH/TAIL only for older
  content; BOARD_STATE only to attach the score/next panel image.
- In LOG: write at most ONE note, only for a reset, a line clear, a mistake,
  a plan change or a pattern the auto log cannot show; otherwise reply DONE.
"""

LOG_SYSTEM_TETRIS = """\
You are playing Tetris and maintaining your external memory.

{action_description}

{diff_notice}

OUTCOME OF YOUR PREVIOUS ACTION (computed from the two consecutive frames; these lines are already auto-logged to stats.txt):
{event_lines}

BOARD NOW (computed from the current pixels - the same board you will see in the action phase):
{board_snapshot}

{progress_line}

MEMORY TAIL (auto-shown each step; READ/SEARCH/TAIL are only for older content):
{memory_tail}

YOUR MEMORY FILES:
{file_index}

AVAILABLE COMMANDS (reply with exactly ONE command per turn):
  <|APPEND_NOTE|>{{"tag": "plan|threat|mistake|progress|reset", "note": "one line"}}<|END|>  record ONE strategic note in strategy.txt
  <|LOG_NOTE|>{{"file": "<name>.txt", "note": "one short line"}}<|END|>   one ephemeral line in notes.txt or plan.txt
  <|READ|>filename.txt<|END|>              Read an entire file (older content only)
  <|SEARCH|>filename.txt|keyword<|END|>    Search a file for lines matching a keyword
  <|TAIL|>filename.txt|N<|END|>            Read the last N lines of a file
  <|BOARD_STATE|>{{"panels": "left|right|both"}}<|END|>  only to attach the score/next panel image
  <|DONE|>                                 Finish logging, move on to action selection

{mode_block}"""

# The LOG model call is gated (see _log_gate_reasons): routine outcomes are
# covered by the auto log in stats.txt, and the LLM is consulted here only when
# the outcome is strategically notable (event mode) or on the scheduled
# consolidation step.  The mode block is computed by code from the observed
# events; the LLM still decides whether, and what, to write.
LOG_MODE_EVENT = """\
MODE: EVENT LOGGING - the outcome of your previous action is noteworthy (trigger: {reasons}).
The raw event lines are already auto-logged to stats.txt - your note must add strategy, not duplicate them.
Write exactly ONE note with the matching command, then reply <|DONE|>:
- a "board reset" event -> APPEND_NOTE tag "reset", one line (e.g. game {game_index} restarted, playing the new game).
- line(s) cleared -> APPEND_NOTE tag "progress"; include the exact panel score if you can read it from the current screenshot.
- the environment executed a different action than you output -> APPEND_NOTE tag "mistake".
- you deviated from the suggested plan (step {prev_step}: suggested "{last_hint}", you committed "{last_committed}") -> APPEND_NOTE tag "plan" if the deviation was deliberate (say why), tag "mistake" if not.
- the board block was not detected in the frame -> APPEND_NOTE tag "threat" (e.g. "board block unavailable - following the screenshot directly"); you may also SAVE_FRAME this frame for later recall.
- an unchanged observation -> LOG_NOTE to notes.txt, one line.
If the MEMORY TAIL already holds a matching note for this exact outcome, reply <|DONE|> directly without writing.
At most ONE memory write this phase. Keep every note to a single line. Finish with <|DONE|>."""

LOG_MODE_CONSOLIDATION = """\
MODE: PERIODIC MEMORY CHECK (scheduled every 10 steps; no notable event this step).
1. Score reconciliation: the running estimate is derived from drop distances and misses line-clear bonuses. Read the exact score from the score panel in the current screenshot (no tool call needed). If you can read it confidently and it differs from the estimate by 50 or more, write ONE APPEND_NOTE tag "progress" with both numbers (e.g. "panel 640 vs est 560 - clear bonuses not in estimate").
2. Stale notes: if a strategic note in the MEMORY TAIL no longer holds (threat gone, plan abandoned), supersede it by APPENDING ONE APPEND_NOTE tag "plan" naming the superseded step. Never rewrite strategy.txt with WRITE - it is long and a rewrite risks losing the policy.
At most ONE write, then <|DONE|>. If nothing is off or stale, reply <|DONE|> directly - most checks need no write."""

PLAN_SYSTEM_TETRIS = """\
{instruction_prompt}

You have an external memory system. The observation below also contains a board state computed from the current screenshot pixels.

{stuckness_warning}

YOUR MEMORY FILES:
{file_index}

MEMORY TAIL (auto-shown each step; READ/SEARCH/TAIL are only for older content):
{memory_tail}

{progress_line}

CURRENT OBSERVATION:
{observation}

{pending_note_rule}DECISION PROCEDURE:
1. Read the BOARD block: grid, stack heights, headroom, active piece, ghost, events and the SUGGESTED PLAN computed from the current pixels.
2. Commit the FIRST action of the SUGGESTED PLAN exactly as given. The plan is a slide (move_left/move_right, at most 4) followed by hard_drop, chosen to maximize the drop distance of the current piece.
3. Deviate only when the screenshot clearly contradicts the board block (a different piece, a row about to clear, a dangerous peak). Then commit your own action; BOARD_STATE only re-checks the change and can attach the score/next panel, it does not show a board you do not already have.
4. Scoring economics: hard_drop pays 2 points per row, soft_drop 1 point per row (at most ~8), and move/rotate/wait pay nothing while each still costs one gravity row; a top-out resets the score to 0. So hard_drop is the best action in every state: never wait, never soft_drop, never rotate.
5. Your best score across games counts for the task, so keep playing after a restart. Your running score estimate is shown above; stats.txt holds the full auto event log - TAIL it only for older entries.
6. If no active piece is detected, the piece is resting on the stack or still entering from the top: commit hard_drop (it locks the resting piece or drops the entering one).

When confident, commit exactly one action with <|ACTION|>action<|END|>.
"""


class MemoryAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        self.max_log_iterations = OmegaConf.select(
            config, "agent.max_log_iterations", default=MAX_LOG_ITERATIONS,
        )
        self.max_validate_iterations = OmegaConf.select(
            config, "agent.max_validate_iterations", default=MAX_VALIDATE_ITERATIONS,
        )
        self.max_search_results = OmegaConf.select(
            config, "agent.max_search_results", default=MAX_SEARCH_RESULTS,
        )
        self.fallback_action_override = OmegaConf.select(
            config, "agent.fallback_action", default=None,
        )
        base_dir = OmegaConf.select(config, "agent.memory_dir", default=None)
        self._memory_base = base_dir or os.path.join(os.getcwd(), "balrog_memory")
        self._memory_dir = self._memory_base
        self._env_name = None
        self.memory = None
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        # Computed board state (JSON-safe; checkpointed by the worker).
        self._board_state = None
        self._board_prev = None
        self._board_events = []
        self._board_block = ""
        self._decoded_frame_id = None
        self._score_estimate = None
        self._game_index = 0
        # LOG-gate bookkeeping (JSON-safe; checkpointed with the state).
        self._last_hint = None
        self._last_committed = None
        self._deviated = False
        self._decode_fail_streak = 0
        self._pending_event_note = False
        self._log_gate = None
        logger.info("Memory agent files at: %s", self._memory_dir)

    # ----------------------------- helpers --------------------------------

    def _is_tetris(self):
        try:
            instruction = self.prompt_builder.system_prompt or ""
        except Exception:
            instruction = ""
        return "tetris" in instruction.casefold()

    def _get_initial_files(self):
        if self._is_tetris():
            return {
                "plan.txt": "",
                "stats.txt": "",
                "notes.txt": "",
                "strategy.txt": TETRIS_STRATEGY,
            }
        if self._env_name and self._env_name in INITIAL_FILES_BY_ENV:
            return copy.deepcopy(INITIAL_FILES_BY_ENV[self._env_name])
        return copy.deepcopy(INITIAL_FILES_DEFAULT)

    def _get_spatial_guidance(self):
        if self._env_name and self._env_name in SPATIAL_GUIDANCE:
            return SPATIAL_GUIDANCE[self._env_name]
        return SPATIAL_GUIDANCE_DEFAULT

    def _get_instruction_prompt_override(self):
        if self._env_name == "nle":
            return NLE_MEMORY_SYSTEM_PROMPT
        elif self._env_name == "minihack":
            return MINIHACK_MEMORY_SYSTEM_PROMPT
        return None

    def _get_fallback_action(self):
        if self.fallback_action_override is not None:
            return self.fallback_action_override
        if self._is_tetris():
            # hard_drop is the dominant fallback for this game: it locks a
            # resting piece, drops an entering one, or is a safe no-op;
            # wait/soft_drop only waste a gravity row.
            return "hard_drop"
        if self._env_name and self._env_name in FALLBACK_ACTIONS:
            return FALLBACK_ACTIONS[self._env_name]
        return FALLBACK_ACTION_DEFAULT

    def _reset_board_decode(self):
        self._board_state = None
        self._board_prev = None
        self._board_events = []
        self._board_block = ""
        self._decoded_frame_id = None
        self._score_estimate = None
        self._game_index = 0
        self._last_hint = None
        self._last_committed = None
        self._deviated = False
        self._decode_fail_streak = 0
        self._pending_event_note = False
        self._log_gate = None

    def configure_memory(self, env_name, task, episode_idx=0):
        self._env_name = env_name
        self._memory_dir = os.path.join(
            self._memory_base, env_name, task, "ep_{:02d}".format(episode_idx),
        )
        self.memory = MemoryFileSystem(
            self._memory_dir, initial_files=self._get_initial_files(),
            max_search_results=self.max_search_results,
        )
        self._reset_board_decode()

    def reset(self):
        self.prompt_builder.reset()
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        self._reset_board_decode()
        if self.memory is not None:
            self.memory.reset(initial_files=self._get_initial_files())

    # ------------------------ board decode wiring -------------------------

    def _decode_current_frame(self):
        """Decode the current public screenshot (idempotent per frame id)."""
        image = getattr(self, "current_image", None)
        frame_id = getattr(self, "current_frame_id", None)
        if image is None:
            self._board_state = {"detected": False}
            self._board_events = []
            self._board_block = ""
            self._decoded_frame_id = frame_id
            return
        if frame_id is not None and frame_id == self._decoded_frame_id \
                and self._board_state is not None:
            return
        prev = self._board_state
        try:
            state = decode_frame(image, prev_state=prev)
            events = detect_events(prev, state) if (
                prev and prev.get("detected") and state.get("detected")) else []
        except Exception as exc:
            logger.error("Board decode failed: %s", exc)
            state, events = {"detected": False, "reason": "decode error"}, []
        self._board_prev = prev if prev else {}
        self._board_state = state
        self._board_events = list(events)
        self._decoded_frame_id = frame_id
        self._board_block = format_board_block(state, events)

    def _progress_line(self):
        estimate = self._score_estimate if self._score_estimate is not None else 0
        return ("PROGRESS: running score estimate {} (est, derived from drop "
                "distances; the exact score is on the score panel); game {} "
                "(best score across games counts for the task).").format(
                    int(estimate), self._game_index)

    def _memory_tail(self):
        """Bounded auto-tail of the memory files, shown in every prompt."""
        if self.memory is None:
            return "(memory unavailable)"
        parts = []
        for name, n in (("stats.txt", 3), ("strategy.txt", 2),
                        ("plan.txt", 2), ("notes.txt", 1)):
            try:
                content = self.memory.read(name)
            except Exception:
                continue
            if not content or content == "[FILE IS EMPTY]" \
                    or content.startswith("[ERROR"):
                continue
            lines = [ln for ln in content.splitlines() if ln.strip()]
            if not lines:
                continue
            parts.append("[{}] last {} line(s):\n{}".format(
                name, len(lines[-n:]), "\n".join(lines[-n:])))
        text = "\n".join(parts)
        if len(text) > 900:
            text = text[:900]
        return text or "(no memory content yet)"

    def _next_action_hint(self):
        state = self._board_state
        if not state or not state.get("detected"):
            return ""
        suggestion = state.get("suggestion")
        if suggestion and suggestion.get("plan"):
            return suggestion["plan"][0]
        # No usable suggestion: the piece is resting or entering from the
        # top.  hard_drop is the best action in every such state (it locks
        # a resting piece or drops an entering one); wait/soft_drop pay
        # nothing and still cost a gravity row.
        return "hard_drop"

    def _landing_points(self, prev, cur, effective_action):
        """Rough points for a piece that just locked: drop rows x per-row rate."""
        try:
            cur_locked = set(map(tuple, cur.get("locked") or []))
            prev_locked = set(map(tuple, prev.get("locked") or []))
            prev_active = set(map(tuple, prev.get("active") or []))
            landed = cur_locked - prev_locked - set(map(tuple, cur.get("active") or []))
            if not landed:
                return 0, 0
            land_top = min(r for r, _ in landed)
            if prev_active:
                prev_top = min(r for r, _ in prev_active)
            elif prev.get("strip_cols"):
                prev_top = -1  # piece was still in the top strip
            else:
                prev_top = 0
            distance = max(0, land_top - prev_top)
            per_row = 2 if effective_action == "hard_drop" else 1
            return distance, distance * per_row
        except Exception:
            return 0, 0

    def _log_event_record(self, prev_action, prev_validated_action):
        """Persist the computed event lines (outcome of the PREVIOUS action).

        Only called from the LOG phase, which exists only after a previous
        action has been executed, so the two-phase invariant holds.
        """
        cur = self._board_state or {}
        events = self._board_events
        if not cur.get("detected") or not events:
            return "(no board events this step - nothing auto-logged)"
        prev = self._board_prev or {}
        effective = prev_validated_action or prev_action
        lines = []
        for event in events:
            if event.startswith("board reset"):
                self._game_index += 1
                self._score_estimate = 0
                lines.append("step {} game {}: board reset (game restarted); score estimate reset to 0".format(
                    self.step_count - 1, self._game_index))
                continue
            if event.startswith("piece landed"):
                distance, points = self._landing_points(prev, cur, effective)
                self._score_estimate = (self._score_estimate or 0) + points
                lines.append("step {} game {}: '{}' - {} piece landed ({}): fell ~{} row(s), est +{}; running score est {}".format(
                    self.step_count - 1, self._game_index, effective,
                    prev.get("piece_name") or cur.get("piece_name") or "?",
                    event, distance, points, self._score_estimate))
                continue
            lines.append("step {} game {}: {}".format(
                self.step_count - 1, self._game_index, event))
        if self.memory is not None:
            for line in lines:
                self.memory.append("stats.txt", line)
        return "\n".join(lines)

    def _note_file_lines(self):
        """Non-empty line count of the LLM-written files (stats.txt is auto-logged by code)."""
        total = 0
        if self.memory is None:
            return 0
        for name in ("strategy.txt", "notes.txt", "plan.txt"):
            try:
                content = self.memory.read(name) or ""
            except Exception:
                continue
            if content.startswith("[ERROR") or content == "[FILE IS EMPTY]":
                continue
            total += len([ln for ln in content.splitlines() if ln.strip()])
        return total

    def _log_gate_reasons(self, prev_action, prev_validated_action, diff_notice):
        """Code-computed triggers for a LOG model call (public observations only).

        Routine outcomes are covered by the auto log in stats.txt; the LLM is
        consulted when the outcome is strategically notable, or on the
        scheduled consolidation step. The LLM still decides what to write.
        """
        reasons = []
        cur = self._board_state or {}
        if not cur.get("detected"):
            self._decode_fail_streak += 1
            if self._decode_fail_streak == 1 or self._decode_fail_streak % 10 == 0:
                reasons.append("board_not_detected")
        else:
            self._decode_fail_streak = 0
        for event in self._board_events:
            if event.startswith("board reset"):
                reasons.append("board_reset")
            elif "line(s) cleared" in event:
                reasons.append("line_clear")
        if prev_validated_action is not None and prev_validated_action != prev_action:
            reasons.append("host_fallback")
        if self._deviated:
            reasons.append("deviation")
        if diff_notice:
            reasons.append("observation_unchanged")
        if not reasons and self.step_count % 10 == 0:
            reasons.append("consolidation")
        return reasons

    # ------------------------------ main loop -----------------------------

    def act(self, obs, prev_action=None, prev_validated_action=None):
        self._decode_current_frame()
        obs_text = self._format_obs(obs)
        spatial_guidance = self._get_spatial_guidance()

        override = self._get_instruction_prompt_override()
        if override is not None:
            instruction_prompt = override
        else:
            instruction_prompt = self.prompt_builder.system_prompt or ""

        diff_notice = ""
        if self._prev_obs_text is not None:
            if obs_text.strip() == self._prev_obs_text.strip():
                self._unchanged_count += 1
                if self._unchanged_count >= 3:
                    diff_notice = ("⚠️ CRITICAL WARNING: The observation has been UNCHANGED for {} consecutive actions. You are STUCK. Your recent actions have had NO EFFECT on the environment. You MUST try a COMPLETELY DIFFERENT approach — do not repeat any of your recent actions.").format(self._unchanged_count)
                else:
                    diff_notice = ("⚠️ WARNING: The observation is IDENTICAL to the previous step. Your previous action likely had NO EFFECT. This is important to log — the action may be invalid or inapplicable in the current state.")
            else:
                self._unchanged_count = 0

        map_analysis = ""
        if self._env_name in ("nle", "minihack"):
            map_analysis = parse_nle_map(obs_text)

        total_in = 0
        total_out = 0

        log_debug = None
        if prev_action is not None and self.step_count > 0:
            log_debug = self._log_phase(
                obs_text, prev_action, prev_validated_action,
                diff_notice, spatial_guidance, map_analysis,
            )
            total_in += log_debug["input_tokens"]
            total_out += log_debug["output_tokens"]
            if self._is_tetris():
                self._pending_event_note = bool(
                    log_debug.get("mode") == "event"
                    and not log_debug.get("note_files_grew", True))
            else:
                self._pending_event_note = False
        else:
            self._pending_event_note = False

        plan_debug = self._plan_validate_phase(
            obs_text, instruction_prompt, diff_notice,
            spatial_guidance, map_analysis,
            prev_validated_action=prev_validated_action,
        )
        total_in += plan_debug["input_tokens"]
        total_out += plan_debug["output_tokens"]

        final_action = plan_debug["final_action"]

        repaired_action, was_repaired = repair_combined_action(final_action)
        if was_repaired:
            logger.info(
                "Action repaired: '%s' → '%s' (split combined two-step action)",
                final_action, repaired_action,
            )
            final_action = repaired_action

        self.last_step_debug = {
            "step": self.step_count,
            "log_phase": log_debug,
            "plan_phase": plan_debug,
            "memory_snapshot": self.memory.snapshot(),
            "diff_notice": diff_notice,
            "unchanged_count": self._unchanged_count,
            "map_analysis": map_analysis,
            "board_events": list(self._board_events),
            "score_estimate": self._score_estimate,
        }

        final = LLMResponse(
            model_id=plan_debug.get("model_id", "unknown"),
            completion=final_action,
            stop_reason="stop",
            input_tokens=total_in,
            output_tokens=total_out,
            reasoning=None,
        )

        self._prev_obs_text = obs_text
        self.step_count += 1
        return final

    def _format_obs(self, obs):
        parts = []
        long_ctx = obs.get("text", {}).get("long_term_context", "")
        short_ctx = obs.get("text", {}).get("short_term_context", "")
        if long_ctx:
            parts.append(long_ctx)
        if short_ctx:
            parts.append(short_ctx)
        base = "\n".join(parts) if parts else "(no observation)"
        if not self._is_tetris():
            return base
        if not self._board_block:
            return base + "\n\n(computed board unavailable this step - rely on the screenshot and your memory)"
        hint = self._next_action_hint()
        extra = self._board_block
        if hint:
            extra += ("\nNEXT ACTION: {} - commit it with <|ACTION|>{}<|END|> unless the screenshot "
                      "clearly contradicts the board above").format(hint, hint)
        return base + "\n\n" + extra

    # ---------------------------- tool executors ---------------------------

    def _execute_read_op(self, op, args):
        if op == "READ":
            return self.memory.read(args["filename"])
        elif op == "SEARCH":
            return self.memory.search(args["filename"], args["query"])
        elif op == "TAIL":
            return self.memory.tail(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            return self.memory.count(args["filename"], args["query"])
        return "[Unknown read op: {}]".format(op)

    def _execute_write_op(self, op, args):
        if op == "READ":
            return self.memory.read(args["filename"])
        elif op == "SEARCH":
            return self.memory.search(args["filename"], args["query"])
        elif op == "TAIL":
            return self.memory.tail(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            return self.memory.count(args["filename"], args["query"])
        elif op == "APPEND":
            return self.memory.append(args["filename"], args["content"])
        elif op == "WRITE":
            return self.memory.write(args["filename"], args["content"])
        elif op == "CREATE":
            return self.memory.create(args["filename"])
        return "[Unknown op: {}]".format(op)

    def _format_tool_result(self, op, args, result):
        if op == "READ":
            header = "READ {}".format(args["filename"])
        elif op == "SEARCH":
            header = "SEARCH {} for '{}'".format(
                args["filename"], args["query"],
            )
        elif op == "TAIL":
            header = "TAIL {} (last {} lines)".format(
                args["filename"], args.get("n", 10),
            )
        elif op == "COUNT":
            header = "COUNT {} for '{}'".format(
                args["filename"], args["query"],
            )
        elif op == "APPEND":
            header = "APPEND to {}".format(args["filename"])
            return header, "── {} ──\n{}".format(header, result)
        elif op == "WRITE":
            header = "WRITE to {}".format(args["filename"])
            return header, "── {} ──\n{}".format(header, result)
        elif op == "CREATE":
            header = "CREATE {}".format(args["filename"])
            return header, "── {} ──\n{}".format(header, result)
        else:
            return result, result
        return header, "── {} ──\n{}".format(header, result)

    # ------------------------------- phases --------------------------------

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        prev_step = self.step_count - 1
        file_index = self.memory.file_index()

        if prev_validated_action is not None and prev_validated_action != prev_action:
            action_description = (
                "At the PREVIOUS step (step {prev_step}) your LLM output was: "
                "\"{prev_action}\"\n"
                "This was NOT a valid action. The environment executed the "
                "default action \"{prev_validated}\" instead.\n"
                "THIS IS AN IMPORTANT FAILURE TO LOG — your output did not "
                "match any valid action."
            ).format(
                prev_step=prev_step,
                prev_action=prev_action,
                prev_validated=prev_validated_action,
            )
        else:
            effective = prev_validated_action or prev_action
            action_description = ("At the PREVIOUS step (step {prev_step}) you took action: \"{effective}\"").format(prev_step=prev_step, effective=effective)

        if self._is_tetris():
            event_lines = self._log_event_record(prev_action, prev_validated_action)
            reasons = self._log_gate_reasons(prev_action, prev_validated_action, diff_notice)
            mode = None
            if any(r != "consolidation" for r in reasons):
                mode = "event"
            elif reasons:
                mode = "consolidation"
            self._log_gate = {"reasons": reasons, "mode": mode,
                              "llm_called": mode is not None}
            if mode is None:
                return {
                    "initial_prompt": None,
                    "turns": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "gated": True,
                    "mode": None,
                    "note_files_grew": False,
                }
            if mode == "event":
                mode_block = LOG_MODE_EVENT.format(
                    reasons=", ".join(reasons),
                    game_index=self._game_index,
                    prev_step=prev_step,
                    last_hint=self._last_hint or "(none)",
                    last_committed=self._last_committed or "(none)",
                )
            else:
                mode_block = LOG_MODE_CONSOLIDATION
            pre_marker = self._note_file_lines()
            initial_prompt = LOG_SYSTEM_TETRIS.format(
                action_description=action_description,
                diff_notice=diff_notice,
                event_lines=event_lines,
                file_index=file_index,
                board_snapshot=self._board_block,
                progress_line=self._progress_line(),
                memory_tail=self._memory_tail(),
                mode_block=mode_block,
            )
        else:
            initial_prompt = LOG_SYSTEM.format(
                action_description=action_description,
                diff_notice=diff_notice,
                spatial_guidance=spatial_guidance,
                map_analysis=map_analysis,
                observation=obs_text,
                file_index=file_index,
            )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []

        for iteration in range(self.max_log_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Log phase LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                })
                break

            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, has_done = parse_log_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_write_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if has_done:
                break
            if not calls:
                break

            updated_index = self.memory.file_index()
            continue_msg = LOG_CONTINUE.format(
                results=results_text,
                file_index=updated_index,
            )
            messages.append(Message(role="user", content=continue_msg))

        if self._is_tetris():
            return {
                "initial_prompt": initial_prompt,
                "turns": turns,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "gated": False,
                "mode": mode,
                "note_files_grew": self._note_file_lines() > pre_marker,
            }
        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, spatial_guidance, map_analysis,
                             prev_validated_action=None):
        file_index = self.memory.file_index()

        stuckness_warning = ""
        if diff_notice:
            stuckness_warning = diff_notice
            if prev_validated_action:
                stuckness_warning += (
                    "\nYour last executed action was '{}'. "
                    "Do NOT repeat it — choose something different."
                ).format(prev_validated_action)

        if self._is_tetris():
            pending_rule = ""
            if self._pending_event_note:
                pending_rule = (
                    "NOTE: the noteworthy outcome of your previous action was not "
                    "recorded in memory yet. Issue ONE <|APPEND_NOTE|> with the "
                    "matching tag (reset/progress/mistake/plan) first; then commit "
                    "the action.\n\n"
                )
            self._pending_event_note = False
            initial_prompt = PLAN_SYSTEM_TETRIS.format(
                instruction_prompt=instruction_prompt,
                stuckness_warning=stuckness_warning,
                file_index=file_index,
                observation=obs_text,
                memory_tail=self._memory_tail(),
                progress_line=self._progress_line(),
                pending_note_rule=pending_rule,
            )
        else:
            initial_prompt = PLAN_SYSTEM.format(
                instruction_prompt=instruction_prompt,
                spatial_guidance=spatial_guidance,
                stuckness_warning=stuckness_warning,
                map_analysis=map_analysis,
                file_index=file_index,
                observation=obs_text,
            )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"

        for iteration in range(self.max_validate_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Plan phase LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                    "committed_action": None,
                })
                break

            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, committed_action = parse_plan_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
                "committed_action": committed_action,
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if committed_action is not None:
                final_action = committed_action
                break

            if not calls:
                logger.warning(
                    "Plan phase: no SEARCH/READ/TAIL/COUNT/ACTION found. Raw: %s", assistant_text[:200],
                )
                final_action = self._get_fallback_action()
                break

            continue_msg = PLAN_CONTINUE.format(results=results_text)
            messages.append(Message(role="user", content=continue_msg))

        if final_action is None:
            fallback = self._get_fallback_action()
            final_action = fallback
            logger.warning(
                "Plan phase exhausted %d iterations without <|ACTION|>. Defaulting to '%s'.", self.max_validate_iterations,
                fallback,
            )

        if self._is_tetris():
            hint = self._next_action_hint()
            self._last_hint = hint or None
            self._last_committed = final_action
            self._deviated = bool(hint and final_action != hint)

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "final_action": final_action,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "model_id": model_id,
        }
