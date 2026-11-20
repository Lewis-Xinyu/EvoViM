"""Exercise the retained card matcher on a generated four-card board."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from visual_rsi.adapters.base import PublicObs, StepResult
from visual_rsi.host.episode import run_episode
from visual_rsi.host.model import ModelCall, ModelReply
from visual_rsi.programs import open_program


class FourCardBoard:
    name = "four-card-demo"
    has_image = True
    public_text_fields = ("long_term_context", "short_term_context")
    max_steps = 4
    fallback_action = "aA"
    faces = {"aA": "coral", "aB": "gold", "bA": "coral", "bB": "gold"}

    def reset(self, seed: int):
        self.flips = []
        self.visible = []
        self.message = "Flip the first card."

    def legal_actions(self):
        return list(self.faces)

    def task_instruction(self):
        return "Flip cards by coordinate and find one matching pair in four flips."

    def metric_description(self):
        return "1 if a pair was found, 0 otherwise"

    def observe(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        for i, (position, face) in enumerate(self.faces.items()):
            x, y = 64 * (i % 2), 64 * (i // 2)
            color = face if position in self.visible else "navy"
            draw.rectangle((x + 4, y + 4, x + 59, y + 59), fill=color)
        return PublicObs({"long_term_context": "2x2 board, rows a-b and columns A-B.",
                          "short_term_context": self.message}, image, f"f{len(self.flips):05d}")

    def step(self, action):
        self.flips.append(action)
        matched = False
        if len(self.flips) % 2 == 1:
            self.visible = [action]
            self.message = f"You flipped {action}. Flip the second card."
        else:
            first = self.flips[-2]
            self.visible = [first, action]
            matched = first != action and self.faces[first] == self.faces[action]
            verdict = "Match!" if matched else "No match. Flip the first card."
            self.message = f"You flipped {first} and {action}. {verdict}"
        truncated = len(self.flips) == self.max_steps and not matched
        return StepResult(self.observe(), matched or truncated, private_record=matched,
                          terminated=matched, truncated=truncated, metrics={"success": matched})

    def score(self, private_record):
        return float(private_record)

class CardPlayer:
    """Explore three cells, then play the mate returned by the evolved tool."""

    exploration = ("aA", "aB", "bA")

    def __init__(self):
        self.recalled_pair = []

    def generate(self, call: ModelCall) -> ModelReply:
        if call.phase == "LOG":
            if call.step > 0 and call.call_index == 1:
                return ModelReply("<|STORE_CARD_VISUAL|>{}<|END|>")
            return ModelReply("<|DONE|>")
        if call.step < len(self.exploration):
            action = self.exploration[call.step]
        elif call.call_index == 1:
            return ModelReply("<|QUERY_CARD_VISUAL|>{}<|END|>")
        elif call.call_index == 2:
            # The tool returns one line of JSON in the host's continuation message.
            result = next(line for line in call.messages[-1]["content"].splitlines() if line.startswith("{"))
            query = json.loads(result)
            mate, = query["pending_exact_mates"]
            self.recalled_pair = [query["pending"], mate]
            args = json.dumps({"positions": [mate]})
            return ModelReply(f"<|RECALL_CARD_VISUAL|>{args}<|END|>")
        else:
            action = self.recalled_pair[1]
        return ModelReply(f"<|ACTION|>{action}<|END|>")


def run_card_demo(archive: Path) -> dict:
    player = CardPlayer()
    with open_program("gpt/card-matching") as bundle:
        result = run_episode(FourCardBoard(), bundle, line="M", seed=0, model=player, archive=archive)
    return {**result, "recalled_pair": player.recalled_pair}
