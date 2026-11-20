"""LLM-authored place facts, observed transitions, and an editable route cursor.

All semantic content enters through an explicit tool. Code supplies provenance,
idempotence, field replacement and retrieval, never a world action.
"""
import copy
import json
import re
from pathlib import Path


class PlaceMemory:
    def __init__(self, root):
        self.path = Path(root) / 'place_memory.json'
        if not self.path.exists():
            self.persist(self.empty())

    @staticmethod
    def empty():
        return {'schema': 2, 'epoch': 0, 'places': {}, 'route': {},
                'current': None, 'transitions': {}, 'revisions': [], 'retired': {}}

    def load(self):
        data = json.loads(self.path.read_text(encoding='utf-8'))
        data.setdefault('retired', {})
        return data

    def persist(self, data):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
        tmp.replace(self.path)

    @staticmethod
    def text(value, maximum=600):
        if not isinstance(value, str) or len(value) > maximum:
            raise ValueError('Expected a bounded string')
        return value.strip()

    @staticmethod
    def stamp(step, frame):
        return {'observed_step': int(step), 'frame_id': frame}

    def revise(self, data, kind, key, before, after, stamp):
        if before != after:
            data['revisions'].append({'kind': kind, 'key': key, 'before': before,
                                      'after': after, 'epoch': data['epoch'], **stamp})
            # The active facts and full transition table remain; this is only
            # a bounded audit of superseded field values.
            data['revisions'] = data['revisions'][-64:]

    def update_place(self, step, frame, phase, previous_action, place,
                     exits=None, remaining=None, note=None, confidence=None,
                     outcome=None, from_place=None, progress=None,
                     supersedes=None, inspection=None):
        place = self.text(place, 100)
        if not place:
            raise ValueError('Choose a stable, episode-local landmark name')
        if outcome is not None and (phase != 'LOG' or step < 1 or previous_action is None):
            raise ValueError('An outcome is allowed only in LOG after an observed previous action')
        if progress is not None and outcome is None:
            raise ValueError('Progress belongs to an observed outcome, not a planned action')
        if outcome not in (None, 'moved', 'blocked', 'unchanged', 'unknown', 'reset'):
            raise ValueError('Unknown outcome')
        if confidence not in (None, 'confirmed', 'tentative'):
            raise ValueError('Unknown confidence')
        updates = {}
        for field, value in [('exits', exits), ('remaining', remaining)]:
            if value is not None:
                if not isinstance(value, list) or len(value) > 12:
                    raise ValueError('At most 12 strings in ' + field)
                updates[field] = [self.text(v, 200) for v in value]
        if note is not None:
            updates['note'] = self.text(note)
        if confidence is not None:
            updates['confidence'] = confidence
        if from_place is not None:
            from_place = self.text(from_place, 100)
        if supersedes is not None:
            supersedes = self.text(supersedes, 100)
            if not supersedes or supersedes == place:
                raise ValueError('supersedes must name a different incorrect place')
        if progress not in (None, 'gain', 'none', 'unknown'):
            raise ValueError('Unknown progress')
        data = self.load()
        data['schema'] = 2
        stamp = self.stamp(step, frame)
        prior_transition = data['transitions'].get(str(int(step)-1))
        if outcome is not None and from_place is None:
            # Attribution uses only an explicitly authored predecessor location.
            # Repeated corrections retain the original source, not the new location.
            if prior_transition is not None and prior_transition['epoch'] == data['epoch']:
                from_place = prior_transition.get('from_place')
            elif data['current'] and data['current']['observed_step'] == step-1:
                from_place = data['current']['place']
        if supersedes is not None:
            removed = data['places'].pop(supersedes, None)
            data['retired'][supersedes] = {'replacement': place, **stamp}
            self.revise(data, 'retire_place', supersedes, removed, None, stamp)
        data['retired'].pop(place, None)
        record = copy.deepcopy(data['places'].get(place, {'place': place, 'facts': {}, 'visits': []}))
        for field, value in updates.items():
            fact = {'value': value, **stamp, 'confidence': confidence or 'tentative'}
            before = record['facts'].get(field)
            # Same value updates freshness without duplicate revision records.
            if before is not None and before['value'] != value:
                self.revise(data, 'place', place + '.' + field, before, fact, stamp)
            record['facts'][field] = fact
        record['last_observed'] = stamp
        if step not in record['visits']:
            record['visits'].append(int(step))
        data['places'][place] = record
        data['current'] = {'place': place, **stamp}
        transition = None
        if outcome is not None:
            transition = {'action_step': int(step)-1, 'action': previous_action,
                          'from_place': from_place, 'to_place': place,
                          'outcome': outcome, 'progress': progress or 'unknown',
                          'epoch': data['epoch'], **stamp}
            if inspection is not None:
                transition['inspection'] = copy.deepcopy(inspection)
            elif prior_transition is not None and prior_transition['epoch'] == data['epoch']:
                if 'inspection' in prior_transition:
                    transition['inspection'] = prior_transition['inspection']
            key = str(int(step)-1)
            before = data['transitions'].get(key)
            if before is not None and before != transition:
                self.revise(data, 'transition', key, before, transition, stamp)
            data['transitions'][key] = transition
        self.persist(data)
        return {'place': record, 'transition': transition,
                'superseded_place': supersedes,
                'notice': 'Model-authored claims; current pixels take precedence. Omitted fields retained; [] clears a list.'}

    def update_route(self, step, frame, operation='read', target=None,
                     checkpoint=None, reason=None):
        if operation not in ('read', 'update', 'clear', 'new_scene'):
            raise ValueError('Unknown route operation')
        data = self.load()
        stamp = self.stamp(step, frame)
        if operation in ('clear', 'new_scene'):
            self.revise(data, 'route', 'route', data['route'], {}, stamp)
            data['route'] = {}
            if operation == 'new_scene':
                # Explicit LLM invalidation for a new layout/level/restart.
                # Old transitions keep their epoch for audit, not navigation.
                data['epoch'] += 1
                data['places'] = {}
                data['current'] = None
                data['retired'] = {}
            self.persist(data)
        elif operation == 'update':
            before = copy.deepcopy(data['route'])
            for key, value in [('target', target), ('checkpoint', checkpoint), ('reason', reason)]:
                if value is not None:
                    data['route'][key] = self.text(value)
            data['route'].update(stamp)
            self.revise(data, 'route', 'route', before, data['route'], stamp)
            self.persist(data)
        return {'route': data['route'], 'epoch': data['epoch'],
                'notice': 'Route is an intention, never proof of an action outcome.'}

    def recall(self, query='', limit=6):
        data = self.load()
        query = self.text(query, 200).casefold()
        limit = max(1, min(int(limit), 12))
        current = (data['current'] or {}).get('place')
        records = [r for p, r in data['places'].items()
                   if not query or query in json.dumps(r, ensure_ascii=False).casefold()]
        records.sort(key=lambda r: (r['place'].casefold() == query, r['place'] == current,
                                   r['last_observed']['observed_step']), reverse=True)
        selected = records[:limit]
        names = {r['place'] for r in selected}
        transitions = [v for v in data['transitions'].values()
                       if v['epoch'] == data['epoch'] and
                       (not query or v['from_place'] in names or v['to_place'] in names
                        or query in json.dumps(v, ensure_ascii=False).casefold())]
        transitions.sort(key=lambda v: v['action_step'])
        return {'epoch': data['epoch'], 'current': data['current'], 'route': data['route'],
                'places': selected, 'matched_places': len(records),
                'retired_places': data['retired'],
                'transitions': transitions[-12:],
                'recent_corrections': [v for v in data['revisions'] if v['epoch'] == data['epoch']][-3:],
                'notice': 'Exits and remaining targets are dated model claims. Dynamic threats require current observation.'}

    @staticmethod
    def card(record):
        """Bounded view of authored facts; the full records remain tool-readable."""
        if record is None:
            return None
        facts = {}
        for key in ('exits', 'remaining', 'note'):
            if key in record['facts']:
                fact = record['facts'][key]
                value = fact['value']
                value = value[:6] if isinstance(value, list) else value[:280]
                facts[key] = {'value': value, 'observed_step': fact['observed_step'],
                              'confidence': fact['confidence']}
        return {'place': record['place'], 'last_observed': record['last_observed'], 'facts': facts}

    def handoff(self, step):
        """Index existing explicit writes for PLAN; no new place or target inference."""
        data = self.load()
        active = [v for v in data['transitions'].values() if v['epoch'] == data['epoch']]
        active.sort(key=lambda v: v['action_step'])
        recent = active[-6:]
        current = (data['current'] or {}).get('place')
        current_record = data['places'].get(current)
        route_text = ' '.join(str(v) for v in data['route'].values())
        terms = set(re.findall(r'\w{3,}', (route_text + ' ' + (current or '')).casefold()))
        related = [r for name, r in data['places'].items() if name != current]
        def relevance(record):
            words = set(re.findall(r'\w{3,}', json.dumps(record['facts'], ensure_ascii=False).casefold()))
            return (len(words & terms), record['last_observed']['observed_step'])
        related.sort(key=relevance, reverse=True)
        # Old concrete remaining claims should survive the last-three-outcomes
        # window. These are candidates to verify, not an automatically chosen route.
        frontier = [r for r in data['places'].values()
                    if r['facts'].get('remaining', {}).get('value')]
        frontier.sort(key=lambda r: r['facts']['remaining']['observed_step'])
        attempts = {}
        for transition in active[-24:]:
            if current and transition.get('from_place') == current:
                key = transition['action']
                row = attempts.setdefault(key, {'action': key, 'count': 0,
                                                'outcomes': {}, 'last_action_step': None})
                row['count'] += 1
                outcome = transition['outcome']
                row['outcomes'][outcome] = row['outcomes'].get(outcome, 0) + 1
                row['last_action_step'] = transition['action_step']
        recent_rows = []
        for transition in recent[-3:]:
            row = {k: v for k, v in transition.items() if k != 'inspection'}
            if transition.get('inspection'):
                row['inspected_frames'] = [transition['inspection']['source_frame_id'],
                                           transition['inspection']['current_frame_id']]
            recent_rows.append(row)
        return {'epoch': data['epoch'], 'as_of_step': int(step),
                'route': data['route'], 'last_located': data['current'],
                'current_place_facts': self.card(current_record),
                'related_place_facts': [self.card(r) for r in related[:3]],
                'remaining_claims_to_verify': [self.card(r) for r in frontier[:4]],
                'repeated_attempts_here': [r for r in attempts.values() if r['count'] >= 2],
                'place_count': len(data['places']), 'last_observed_outcomes': recent_rows,
                'recent_identity_corrections': list(data['retired'].items())[-3:]}

    def cue(self, step):
        data = self.load()
        active = sorted((v for v in data['transitions'].values() if v['epoch'] == data['epoch']),
                        key=lambda v: v['action_step'])
        recent = active[-6:]
        # A warning based only on entries the model chose to store. Missing
        # observations break consecutive runs; no inference from frame change.
        consecutive = len(recent) >= 3 and all(
            b['action_step'] == a['action_step']+1 for a, b in zip(recent, recent[1:]))
        nonprogress = consecutive and all(v['progress'] == 'none' for v in recent[-3:])
        text = ('EDITABLE MEMORY HANDOFF (only explicit LLM claims; dates matter; '
                'related/frontier cards are retrieval candidates, not verified navigation):\n' +
                json.dumps(self.handoff(step), ensure_ascii=False))
        if nonprogress and recent[-1]['observed_step'] == step:
            text += ('\nStored observations report three consecutive actions without progress. '
                     'Check the route checkpoint and alternative exits; avoid an unproductive loop. '
                     'Cleared transit or a temporary pause can still be necessary.')
        return text
