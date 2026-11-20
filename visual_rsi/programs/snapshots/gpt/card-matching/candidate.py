"""Memory Match: explicit, coordinate-indexed visual memory for line M."""
import json

from agents.memory_agent import MemoryAgent
from visual_tools import install_visual
from card_memory import install_cards, parse_observation, empty_ledger


class Agent(MemoryAgent):
    def _get_initial_files(self):
        files = super()._get_initial_files()
        files['card_memory.json'] = json.dumps(empty_ledger())
        files['cards_index.txt'] = ''
        return files

    def _format_obs(self, obs):
        text = super()._format_obs(obs)
        # Only the current response and one pending-round register are derived.
        # No image, card identity or action outcome is stored here.
        self.card_observation = parse_observation(
            text, self.step_count, getattr(self, '_card_pending', None))
        self._card_pending = self.card_observation['pending']
        if not self.card_observation['dimensions']:
            return text
        view = {k: self.card_observation[k] for k in
                ['observation_step', 'action_step', 'outcome', 'revealed',
                 'removed', 'pending', 'next_flip']}
        return text + '\n\nCURRENT RESPONSE PARSED (not persistent memory):\n' + json.dumps(view)

    def reset(self):
        super().reset()
        self.card_observation = {}
        self._card_pending = None

    def _get_spatial_guidance(self):
        return '''CARD MEMORY WORKFLOW (optional explicit tools):
LOG receives the actual result of the preceding action; never log a future result.
Use <|STORE_CARD_VISUAL|>{}<|END|> to remember the currently revealed card crops,
their coordinates and frame provenance, plus confirmed match/removal or mismatch
evidence. It upserts by coordinate, so repeated observations do not duplicate cards.
Matches and invalid-empty responses need this same elected STORE even though no
new picture is visible: prose in actions_log.txt does not update the card ledger.
Match storage updates removal status. A later observation of an elected removed
slot grounds an empty-slot reference from this episode; the just-matched pictures
are never empty templates. QUERY uses the reference to veto stale retrievals.
If QUERY reports stale_active_excluded, you may elect
<|STORE_CARD_VISUAL|>{"sync_empty":true}<|END|> in the next LOG phase to persist
current empty-slot corrections. This is optional; QUERY itself never writes.
You can select positions, supply an optional note, forget mistaken entries, or
correct grid bounds. Ordinary STORE preserves the measured crop coordinate system;
adding grid_bounds to an identity note does not replace an established grid.
Elect STORE with recalibrate=true only when the geometry itself is wrong or the
layout has moved. With no bounds it measures the current grid; with grid_bounds
it uses your explicit rectangle. Check geometry.identity_requires_reobserve:
old incompatible identities are still recallable but must be revealed and saved
again before exact matching. Never interpret missing exact pairs as a reason to
estimate new bounds on a stable board. Inspect its result; this replaces vague color-only identity
notes. The original text and image tools remain available for other information.
PLAN: <|QUERY_CARD_VISUAL|>{}<|END|> retrieves the saved ledger, known exact pairs,
unremembered coordinates and matches for the pending first card, if any. This is
a read: it does not save the current card. If exact matches are absent, reveal
an unremembered eligible card rather than repeatedly guessing similar colors.
Choose an eligible_unremembered coordinate when exploring. Every choice must
belong to eligible_positions. An old pair or text note cannot override a slot
excluded by the current board. Use eligible_known_exact_pairs for first flips.
If choosing a first flip and an exact pair is known, choose either member; on
the next step query the pending card's mate. pending_exact_mates at the beginning
of QUERY's result lists eligible coordinates with equal native saved pixels.
Consult these before considering any nearest_nonconfirmed picture. Every world flip still requires
your own ACTION command. Do not flip the pending card or a removed coordinate.
After a completed mismatch, the two pictures may still be visible; the NEXT
flip is nevertheless the first flip of a new round. Use the response's phase,
not the mere presence of face-up pictures. Never treat a nearest approximate
neighbor as a confirmed pair. For uncertain identities or alignment, invoke
RECALL_CARD_VISUAL with selected positions to inspect isolated, labelled crops
through the host. STORE_CARD_VISUAL can correct a current reveal or explicitly
forget a mistaken entry. QUERY results include conflicts with past mismatches.
Reply with exactly one host command, wait for its result, then choose the next.
'''


def install(agent, registry):
    install_visual(agent, registry)
    install_cards(agent, registry)
