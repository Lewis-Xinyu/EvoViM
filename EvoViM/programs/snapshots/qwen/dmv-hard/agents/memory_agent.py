import copy
import json
import logging
import os
import re

from omegaconf import OmegaConf

from agents.base import BaseAgent
from client import LLMResponse
from prompt_builder.history import Message

logger = logging.getLogger(__name__)

DMV_ENV = "dmvhard2"

_PAGE_OBS_RE = re.compile(
    r"Session\s+(\d+)\s*/\s*(\d+)\s*,\s*browsing category\s+'([^']*)'\s*\.\s*"
    r"Product page\s+(\d+)\s*/\s*(\d+)\s*:\s*URL\s+/product/([0-9a-fA-F]{8})")
_PROBE_OBS_RE = re.compile(r"Question\s+(\d+)\s+after session\s+(\d+)")
_INVALID_ANSWER_RE = re.compile(r"'(.+?)'\s+is not a valid answer")
_ANSWER_ACTION_RE = re.compile(r"^answer\s+([0-9a-fA-F]{8})\s*$")


def parse_dmv_obs(text):
    """Parse the public observation text of a product-browsing episode.

    Returns {"kind": "page", session, session_total, category, page, page_total,
    product_id} or {"kind": "probe", question, after_session} or None.  Purely
    generic: it only mirrors the instruction's own wording, no unit constants.
    """
    match = _PAGE_OBS_RE.search(text)
    if match:
        return {"kind": "page", "session": int(match.group(1)),
                "session_total": int(match.group(2)), "category": match.group(3),
                "page": int(match.group(4)), "page_total": int(match.group(5)),
                "product_id": match.group(6).lower()}
    match = _PROBE_OBS_RE.search(text)
    if match:
        return {"kind": "probe", "question": int(match.group(1)),
                "after_session": int(match.group(2))}
    return None


DMV_PLAYBOOK = """
STORE BROWSING & RECALL — WORKFLOW
This episode alternates product-page browsing with recall questions. Every question is answered by the 8-hex id from the URL (/product/<id>) of a product page you saw earlier. Your saved frames and the product index are the only way back to them.

ON A PRODUCT PAGE (observation contains "Product page P/20: URL /product/<id>"):
1. If the STORE STATE line says the page is NOT SAVED, call SAVE_FRAME first, with label "s<session> p<page> <id>". Saving every page is what makes later questions answerable.
2. Optionally APPEND one short visual note to probe_notes.txt (e.g. "s1 p7 3c9f01ab: tall grey fabric sofa, left arm rolled") — useful for detail questions.
3. Commit: <|ACTION|>continue<|END|>

ON A QUESTION (observation contains "Question N after session M"):
1. Never reply continue — it is rejected and scores 0.
2. FIRST call MATCH_PROBE with {} as its argument. It runs three matchings of the question image against your saved pages (full page, small pattern inset, central product photo), prints each top-8 list with a TOP-1 MARGIN, prints a DECISION, and attaches the tile ordered by the best matching (best match = cell #1).
3. Follow the DECISION line:
   - "COMMIT" names an id: the match is decisive - commit exactly that id (glance at the tile to confirm).
   - "CONTESTED": the matchings disagree - do not commit from the scores; find the cell of the attached tile whose page the question image re-photographs and commit that id.
   - "FLAT": no matching is decisive - pick the cell of the attached tile that the question image matches and commit its id; if your answer is not visible there, page with RECALL_TILE (same view, offset=8, then keep the offset moving) for a few tiles at most.
   Re-photographs may be rotated, rescaled, partly visible and differently lit; the match already accounts for that, but confirm on the tile.
4. If two cells look close, verify with RECALL_IMAGE on their image_ids (the product index lists them) before answering.
5. Commit exactly: <|ACTION|>answer <id><|END|> with a saved product id.
6. Each question is about a different product - do not repeat an id you already used; STORE STATE lists your earlier answers and a repeat is rejected once.

"""

DMV_LOG_DISCIPLINE = """
STORE MEMORY DISCIPLINE (product-browsing episode):
- The observation above is the environment's response to your previous action. If it is a NEW product page (URL /product/<id>) and STORE STATE says NOT SAVED, call SAVE_FRAME (label "s<session> p<page> <id>") BEFORE <|DONE|>.
- If it reports your previous answer ("Answer recorded." or "'X' is not a valid answer"), APPEND a line to probe_notes.txt: question number, id you gave, verdict.
- product_index.jsonl is auto-maintained by SAVE_FRAME: READ it to list saved ids; never WRITE to it.
"""

PLAN_REJECTION = """
ACTION REJECTED: '{rejected}' would score 0.
{reason}
{store_state}
Saved products by session (page order):
{candidates}
Compare the question image against a RECALL_TILE grid (view 'inset' for pattern photos, 'full' for product photos), then commit exactly: <|ACTION|>answer <id><|END|>
"""

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
    "dmvhard2": {
        "probe_notes.txt": "",
    },
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
    "dmvhard2": (
        "RECALL TASK: questions are answered by the 8-hex id of a product you saved. "
        "The STORE STATE block is derived from your own saves and the current observation; "
        "track which products are saved and which ids you have already answered."
    ),
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
        # dmvhard2 derived state (all JSON-serialisable; see _dmv_update_state)
        self._dmv = False
        self.product_index = {}
        self.answer_log = []
        self._current_dmv_info = None
        # g2: per-step tile dedup + anti-repetition tracking (JSON-safe)
        self._dmv_tiles_shown = []
        self._dmv_tile_count = 0
        self._dmv_tile_step = None
        self._dmv_rep_rejected = []
        # g3: last MATCH_PROBE outcome (decision + three rankings), step-stamped
        self._dmv_match = None
        logger.info("Memory agent files at: %s", self._memory_dir)

    def _get_initial_files(self):
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
        elif self._env_name == DMV_ENV:
            return DMV_PLAYBOOK + "\n\nUSER INSTRUCTION:\n" + (self.prompt_builder.system_prompt or "")
        return None

    def _get_fallback_action(self):
        if self.fallback_action_override is not None:
            return self.fallback_action_override
        if self._env_name and self._env_name in FALLBACK_ACTIONS:
            return FALLBACK_ACTIONS[self._env_name]
        return FALLBACK_ACTION_DEFAULT

    def configure_memory(self, env_name, task, episode_idx=0):
        self._env_name = env_name
        self._memory_dir = os.path.join(
            self._memory_base, env_name, task, "ep_{:02d}".format(episode_idx),
        )
        self.memory = MemoryFileSystem(
            self._memory_dir, initial_files=self._get_initial_files(),
            max_search_results=self.max_search_results,
        )
        if env_name == DMV_ENV:
            self._dmv = True
            self._load_product_index()

    def reset(self):
        self.prompt_builder.reset()
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        if self.memory is not None:
            self.memory.reset(initial_files=self._get_initial_files())

    def act(self, obs, prev_action=None, prev_validated_action=None):
        obs_text = self._format_obs(obs)
        if self._dmv:
            self._dmv_update_state(obs_text)
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
                derived_block=self._dmv_state_block(),
            )
            total_in += log_debug["input_tokens"]
            total_out += log_debug["output_tokens"]

        plan_debug = self._plan_validate_phase(
            obs_text, instruction_prompt, diff_notice,
            spatial_guidance, map_analysis,
            prev_validated_action=prev_validated_action,
            derived_block=self._dmv_state_block(),
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
        return "\n".join(parts) if parts else "(no observation)"

    # ------------------------------------------------------------------
    # dmvhard2 (product browsing + recall questions): derived state,
    # product index, tile requests and the answer gate.  All state is
    # derived from this episode's own public observations; no unit
    # constants.
    # ------------------------------------------------------------------

    def _dmv_index_path(self):
        return os.path.join(self._memory_dir, "product_index.jsonl")

    def _load_product_index(self):
        path = self._dmv_index_path()
        self.product_index = {}
        self.answer_log = []
        if not os.path.isfile(path):
            self._save_product_index()
            return
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                product_id = str(record.get("id", ""))
                if not re.fullmatch(r"[0-9a-f]{8}", product_id):
                    continue
                self.product_index[product_id] = record

    def _save_product_index(self):
        path = self._dmv_index_path()
        lines = []
        for product_id in sorted(self.product_index, key=lambda pid: (
                self.product_index[pid].get("session", 0),
                self.product_index[pid].get("page", 0), pid)):
            entry = dict(self.product_index[product_id])
            entry["id"] = product_id
            lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=True))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.replace(tmp, path)

    def _dmv_update_state(self, obs_text):
        """Parse the current public observation and update derived state.

        Two-phase invariant preserved: at step t this only reads the
        observation available at step t (the environment's response to
        step t-1's action); it never invents an outcome.
        """
        info = parse_dmv_obs(obs_text)
        self._current_dmv_info = info
        invalid = _INVALID_ANSWER_RE.search(obs_text)
        if invalid:
            for record in reversed(self.answer_log):
                if not record.get("malformed"):
                    record["malformed"] = True
                    break
        if info is None:
            return
        if info["kind"] == "page":
            product_id = info["product_id"]
            if product_id not in self.product_index:
                self.product_index[product_id] = {
                    "session": info["session"], "category": info["category"],
                    "page": info["page"], "image_id": None, "frame_id": None,
                    "inset_box": None,
                }
                self._save_product_index()
        elif info["kind"] == "probe":
            # A new question step: reset the per-step tile dedup so each
            # question gets a fresh budget of tile attachments.
            self._dmv_tiles_shown = []
            self._dmv_tile_count = 0

    def _record_saved_frame(self, image_id):
        """SAVE_FRAME handler hook: link the frame to the current page and
        compute the inset box (code-side computation on a public
        observation; the decision to save is the model's)."""
        info = self._current_dmv_info or {}
        if info.get("kind") != "page":
            return
        product_id = info["product_id"]
        entry = self.product_index.get(product_id)
        if entry is None:
            entry = {"session": info["session"], "category": info["category"],
                     "page": info["page"], "image_id": None, "frame_id": None,
                     "inset_box": None}
            self.product_index[product_id] = entry
        entry["image_id"] = image_id
        entry["frame_id"] = self.current_frame_id
        try:
            if self.current_image is not None:
                width, height = self.current_image.size
                entry["frame_w"] = int(width)
                entry["frame_h"] = int(height)
            else:
                entry["frame_w"] = None
                entry["frame_h"] = None
        except Exception:
            entry["frame_w"] = None
            entry["frame_h"] = None
        try:
            from agents.inset import detect_inset
            detection = detect_inset(self.current_image) if self.current_image is not None else None
        except Exception as exc:  # detection must never break a save
            logger.warning("inset detection failed: %s", exc)
            detection = None
        entry["inset_box"] = list(detection["box"]) if detection else None
        self._save_product_index()

    def _dmv_saved_entries(self):
        return sorted(
            ((pid, entry) for pid, entry in self.product_index.items() if entry.get("frame_id")),
            key=lambda item: (item[1].get("session", 0), item[1].get("page", 0), item[0]),
        )

    def _dmv_candidate_entries(self, sessions, ids):
        entries = self._dmv_saved_entries()
        if ids:
            wanted = {str(value).lower() for value in ids}
            return [(pid, entry) for pid, entry in entries if pid in wanted]
        info = self._current_dmv_info or {}
        if sessions is None:
            if info.get("kind") == "probe":
                return [(pid, entry) for pid, entry in entries
                        if entry.get("session", 0) <= info["after_session"]]
            return entries
        sessions = str(sessions).strip()
        span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", sessions)
        if span:
            low, high = int(span.group(1)), int(span.group(2))
            return [(pid, entry) for pid, entry in entries
                    if low <= entry.get("session", 0) <= high]
        if sessions.isdigit():
            return [(pid, entry) for pid, entry in entries
                    if entry.get("session", 0) == int(sessions)]
        return []

    def _dmv_state_block(self):
        """Bounded derived context shown with every model call (dmv only)."""
        if not self._dmv:
            return ""
        info = self._current_dmv_info or {}
        saved = self._dmv_saved_entries()
        lines = ["[STORE STATE — derived from your own observations]"]
        if info.get("kind") == "page":
            product_id = info["product_id"]
            entry = self.product_index.get(product_id) or {}
            if entry.get("image_id"):
                status = "SAVED as {} (inset crop: {})".format(
                    entry["image_id"], "yes" if entry.get("inset_box") else "no")
            else:
                status = "NOT SAVED — call SAVE_FRAME (label: s{} p{} {}) before continuing".format(
                    info["session"], info["page"], product_id)
            lines.append("Current product page: session {}/{} '{}', page {}/{} — product id {} — {}".format(
                info["session"], info["session_total"], info["category"],
                info["page"], info["page_total"], product_id, status))
            if saved:
                lines.append("Saved frames: {} pages (sessions {} to {}).".format(
                    len(saved), saved[0][1].get("session"), saved[-1][1].get("session")))
        elif info.get("kind") == "probe":
            candidates = [(pid, entry) for pid, entry in saved
                          if entry.get("session", 0) <= info["after_session"]]
            with_inset = sum(1 for _pid, entry in candidates if entry.get("inset_box"))
            lines.append(
                "MODE: QUESTION {} after session {} — a recall question, NOT a product page. "
                "'continue' is rejected and scores 0. Reply exactly 'answer <id>' with a saved product id.".format(
                    info["question"], info["after_session"]))
            lines.append("Candidate products (sessions 1 to {}): {} saved, {} with inset crop.".format(
                info["after_session"], len(candidates), with_inset))
            lines.append(
                "FIRST call MATCH_PROBE with {} as its argument: it runs three matchings of the "
                "question image (full page, pattern inset, product photo), prints a DECISION - "
                "COMMIT names the exact id to commit; CONTESTED means verify on the attached "
                "tile; FLAT means pick the matching tile cell and page RECALL_TILE (same view, "
                "offset) if it is not visible - and attaches the tile ordered by best match.")
            if self.answer_log:
                listing = "; ".join(
                    "Q{}->{}{}".format(record["question"], record["id"],
                                       " (malformed!)" if record.get("malformed") else "")
                    for record in self.answer_log)
                lines.append("Your earlier answers: " + listing)
                ids = [record["id"] for record in self.answer_log]
                repeats = sorted({pid for pid in ids if ids.count(pid) > 1})
                if repeats:
                    lines.append(
                        "You reused ids {} — each question is a different product; "
                        "match the tiles instead of repeating.".format(", ".join(repeats)))
            else:
                lines.append("No answers yet this episode.")
        return "\n".join(lines)

    def _dmv_candidate_listing(self, limit=120):
        groups = {}
        order = []
        for pid, entry in self._dmv_saved_entries():
            key = (entry.get("session", 0), entry.get("category", ""))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(pid)
        lines = []
        shown = 0
        for key in order:
            session, category = key
            ids = groups[key]
            if shown + len(ids) > limit:
                ids = ids[:max(0, limit - shown)] + ["..."]
                shown = limit
            lines.append("  session {} '{}': {}".format(session, category, " ".join(ids)))
            shown += len(groups[key])
        return "\n".join(lines) if lines else "  (none saved yet)"

    def _check_committed_action(self, action):
        """The answer gate. Returns ("accept", "") or ("reject", reason).

        Page mode: accept anything.  Probe mode: the action must be
        'answer <8hex>' with an id the agent actually saved (when the
        index is non-empty).  This converts guaranteed-zero answers
        ('continue' on a question, invented ids) into rejected attempts
        the model can repair inside the phase cap.
        """
        info = self._current_dmv_info or {}
        if not (self._dmv and info.get("kind") == "probe"):
            return "accept", ""
        match = _ANSWER_ACTION_RE.match(action.strip())
        if not match:
            return "reject", (
                "'{}' is not a valid answer for this step. The step shows a QUESTION, not a "
                "product page. The only valid form is: answer <id> where <id> is an 8-hex "
                "product id from /product/<id> that you saved.".format(action))
        product_id = match.group(1).lower()
        if self.product_index and product_id not in self.product_index:
            return "reject", (
                "'{}' is not one of the product ids you saved. Saved ids are listed below; "
                "compare the question image against a RECALL_TILE grid and pick a real id.".format(product_id))
        # Anti-repetition: an id already committed for a DIFFERENT question is
        # almost always a stuck repeat (the environment gives no correctness
        # feedback).  Reject it once so the model re-examines; accept it if the
        # model insists a second time, so a genuine same-product question is
        # never permanently blocked.
        question = info.get("question")
        used_for = [r["question"] for r in self.answer_log if r["id"] == product_id]
        if used_for and question not in used_for:
            rep_key = "{}:{}".format(question, product_id)
            if rep_key in self._dmv_rep_rejected:
                return "accept", ""
            self._dmv_rep_rejected.append(rep_key)
            return "reject", (
                "You already used id {} for question {} - each question is a different "
                "product, so reusing it here is very likely wrong. Re-run MATCH_PROBE or "
                "compare the tiles and commit a DIFFERENT saved id.".format(
                    product_id, ", ".join(str(q) for q in used_for)))
        return "accept", ""

    def _dmv_log_answer(self, action):
        match = _ANSWER_ACTION_RE.match(action.strip())
        info = self._current_dmv_info or {}
        if not match or info.get("kind") != "probe":
            return
        self.answer_log.append({
            "question": info["question"],
            "after_session": info["after_session"],
            "id": match.group(1).lower(),
            "malformed": False,
        })

    def _dmv_fallback_answer(self):
        info = self._current_dmv_info or {}
        if not (self._dmv and info.get("kind") == "probe"):
            return None
        entries = self._dmv_candidate_entries(None, None)
        if not entries:
            return None
        used = {r["id"] for r in self.answer_log}
        for pid, _entry in entries:
            if pid not in used:
                return "answer " + pid
        return "answer " + entries[0][0]

    def _dmv_frame_dims(self, entry):
        """(width, height) of an entry's saved frame: from the size recorded
        at SAVE_FRAME time, else by opening the PNG once (cached back into
        the entry).  (None, None) when unavailable."""
        if entry.get("frame_w") and entry.get("frame_h"):
            return int(entry["frame_w"]), int(entry["frame_h"])
        image_id = entry.get("image_id")
        if not image_id:
            return None, None
        try:
            from PIL import Image
            path = os.path.join(self.visual.root, "{}.png".format(image_id))
            with Image.open(path) as stored:
                width, height = stored.size
            entry["frame_w"] = int(width)
            entry["frame_h"] = int(height)
            return int(width), int(height)
        except Exception:
            return None, None

    def _build_tile_request(self, view=None, sessions=None, ids=None, offset=0,
                            limit=None):
        """RECALL_TILE handler: build a host visual program (a labelled tile
        of saved product frames, cell label = product id).  Views: 'inset' =
        each page's corner pattern square (10 per tile, cells <=192px);
        'product' = each page's central product photo (10 per tile, cells
        <=384px); 'full' = the whole page (15 per tile, cells <=384px).  All
        variants are <=32 ops and <=1536px output by construction.

        When the current step ran MATCH_PROBE and no explicit ids were given,
        the cells follow that matching's ranking for the chosen view (best
        match first), so paging with offset walks the ranked list.
        """
        view = str(view or "full").strip().lower()
        if view not in ("inset", "full", "product"):
            raise ValueError("view must be 'inset', 'product' or 'full'")
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            raise ValueError("offset must be a non-negative integer")
        if limit is not None:
            try:
                limit = max(1, int(limit))
            except (TypeError, ValueError):
                raise ValueError("limit must be a positive integer")
        entries = self._dmv_candidate_entries(sessions, ids)
        match = getattr(self, "_dmv_match", None)
        # The 'full' view is rendered from the full-page (ext) matching.
        rank_key = {"full": "ext", "inset": "inset", "product": "product"}.get(view)
        ordered = bool(
            match and match.get("step") == getattr(self, "step_count", None)
            and not ids and rank_key is not None and rank_key in match)
        if ordered:
            order = match[rank_key]
            rank_of = {pid: index for index, pid in enumerate(order)}
            entries = sorted(
                entries, key=lambda item: rank_of.get(item[0], len(order)))
        total = len(entries)
        # Dedup: within one step, never re-attach an identical tile (each
        # question is exactly one step, so a new observation opens a fresh
        # window; the marker makes the window explicit even if the caller
        # re-enters mid-step).
        step = getattr(self, "step_count", None)
        if self._dmv_tile_step != step:
            self._dmv_tiles_shown = []
            self._dmv_tile_count = 0
            self._dmv_tile_step = step
        per_tile = limit if limit is not None else (
            10 if view in ("inset", "product") else 15)
        tile_key = "{}|{}|{}|{}|{}".format(
            view, sessions, offset, ",".join(ids or []), per_tile)
        if tile_key in self._dmv_tiles_shown:
            return ("You already viewed this exact {} tile ({}-{} of {}) this question. "
                    "Do not re-view it - commit your best answer now: "
                    "<|ACTION|>answer <id><|END|>.").format(
                        view, offset + 1, offset + min(max(total - offset, 0), per_tile), total)
        if total == 0:
            return ("No saved product frames match that scope (products overall: {}, with frames: {}). "
                    "Save pages with SAVE_FRAME first, or widen 'sessions'/'ids'.").format(
                        len(self.product_index), len(self._dmv_saved_entries()))
        window = entries[offset:offset + per_tile]
        if not window:
            return ("offset {} is beyond the last candidate ({} total for this scope). "
                    "Use a smaller offset.").format(offset, total)
        from agents.inset import validate_inset_box
        ops = []
        labels = []
        tile_sources = []
        missing = []
        bad_boxes = set()
        for position, (product_id, entry) in enumerate(window):
            box = None
            dims = None
            if view == "inset":
                box = entry.get("inset_box")
                dims = self._dmv_frame_dims(entry)
                if not box or dims[0] is None:
                    missing.append(product_id)
                    continue
                if not validate_inset_box(box, dims[0], dims[1]):
                    bad_boxes.add(product_id)
            elif view == "product":
                dims = self._dmv_frame_dims(entry)
                if dims[0] is None:
                    missing.append(product_id)
                    continue
            base_alias = "b{}".format(position)
            ops.append({"op": "frame", "id": entry["frame_id"], "as": base_alias})
            source_alias = base_alias
            cell_max = 384
            if view == "inset":
                crop_alias = "c{}".format(position)
                ops.append({"op": "crop", "src": base_alias,
                            "box": [int(v) for v in box], "as": crop_alias})
                source_alias = crop_alias
                cell_max = 192
            elif view == "product":
                crop_alias = "c{}".format(position)
                ops.append({"op": "crop", "src": base_alias,
                            "box": [int(0.2 * dims[0]), int(0.2 * dims[1]),
                                    int(0.8 * dims[0]), int(0.8 * dims[1])],
                            "as": crop_alias})
                source_alias = crop_alias
            tile_alias = "t{}".format(position)
            ops.append({"op": "resize", "src": source_alias,
                        "max_side": cell_max, "as": tile_alias})
            labels.append(product_id + ("*" if product_id in bad_boxes else ""))
            tile_sources.append(tile_alias)
        if not tile_sources:
            return ("No {} view for that window ({}). Use view='full' for those pages.").format(
                view, ", ".join(missing) if missing else "none")
        columns = min(5 if view == "inset" else 4, len(tile_sources))
        ops.append({"op": "tile", "srcs": tile_sources, "cols": columns,
                    "labels": labels, "as": "out"})
        self.visual_request = {"ops": ops, "output": "out"}
        self._dmv_tiles_shown.append(tile_key)
        self._dmv_tile_count += 1
        text = ("{} tile: products {}-{} of {} — cell labels are product ids; the matching "
                "product's id is your answer.").format(
                    view, offset + 1, offset + len(tile_sources), total)
        if bad_boxes:
            text += " (a * marks a pattern box that failed geometry validation)"
        if ordered:
            text += (" The cells follow the MATCH_PROBE {} ranking - the best-matching "
                     "page is the first cell.").format(view)
        if self._dmv_tile_count >= 6:
            text += (" You have now viewed {} tiles this question - that is enough. "
                     "Commit your best answer now: <|ACTION|>answer <id><|END|>.").format(
                         self._dmv_tile_count)
        if offset + len(tile_sources) < total:
            text += " More remain: call RECALL_TILE again with offset={} (same view).".format(
                offset + len(tile_sources))
        else:
            text += " That was the last tile for this scope."
        if missing:
            text += " No {} view for: {} (view them with view='full').".format(
                view, ", ".join(missing))
        return text
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

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis, derived_block=""):
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

        observation_with_state = obs_text if not derived_block else obs_text + "\n\n" + derived_block
        initial_prompt = LOG_SYSTEM.format(
            action_description=action_description,
            diff_notice=diff_notice,
            spatial_guidance=spatial_guidance,
            map_analysis=map_analysis,
            observation=observation_with_state,
            file_index=file_index,
        )
        if self._dmv:
            initial_prompt = initial_prompt.replace(
                "AVAILABLE COMMANDS:", DMV_LOG_DISCIPLINE + "\nAVAILABLE COMMANDS:", 1)

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

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, spatial_guidance, map_analysis,
                             prev_validated_action=None, derived_block=""):
        file_index = self.memory.file_index()

        stuckness_warning = ""
        if diff_notice:
            stuckness_warning = diff_notice
            if prev_validated_action:
                stuckness_warning += (
                    "\nYour last executed action was '{}'. "
                    "Do NOT repeat it — choose something different."
                ).format(prev_validated_action)

        observation_with_state = obs_text if not derived_block else obs_text + "\n\n" + derived_block
        initial_prompt = PLAN_SYSTEM.format(
            instruction_prompt=instruction_prompt,
            spatial_guidance=spatial_guidance,
            stuckness_warning=stuckness_warning,
            map_analysis=map_analysis,
            file_index=file_index,
            observation=observation_with_state,
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
                verdict, rejection = self._check_committed_action(committed_action)
                if verdict == "accept":
                    self._dmv_log_answer(committed_action)
                    final_action = committed_action
                    break
                # The committed action would score 0 (e.g. 'continue' on a
                # question step, or an id that was never saved). Reject it and
                # let the model try again; this counts against the phase cap.
                rejection_msg = PLAN_REJECTION.format(
                    rejected=committed_action,
                    reason=rejection,
                    store_state=self._dmv_state_block() or "(no store state)",
                    candidates=self._dmv_candidate_listing(),
                )
                if results_text:
                    rejection_msg = ("TOOL RESULTS FROM YOUR LAST REPLY:\n\n" + results_text
                                     + "\n\n" + rejection_msg)
                turns[-1]["rejection"] = rejection_msg
                messages.append(Message(role="user", content=rejection_msg))
                continue

            if not calls:
                logger.warning(
                    "Plan phase: no SEARCH/READ/TAIL/COUNT/ACTION found. Raw: %s", assistant_text[:200],
                )
                final_action = self._get_fallback_action()
                break

            continue_msg = PLAN_CONTINUE.format(results=results_text)
            messages.append(Message(role="user", content=continue_msg))

        if final_action is None:
            final_action = self._dmv_fallback_answer() or self._get_fallback_action()
            logger.warning(
                "Plan phase exhausted %d iterations without an accepted <|ACTION|>. Defaulting to '%s'.",
                self.max_validate_iterations, final_action,
            )

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "final_action": final_action,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "model_id": model_id,
        }