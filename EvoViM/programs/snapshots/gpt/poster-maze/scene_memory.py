"""Explicit observation bundles; all persistence is behind model-invoked tools.

Spatial indexing and image registration are computations over selected records,
not automatic decisions about what to store or what action to take.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pattern_registration import content_regions, pixel_box, descriptor, compare


OFFSETS = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}


def opposite(direction):
    return 'NESW'[('NESW'.index(direction) + 2) % 4]


def exits(public):
    pose = public.get('pose')
    if not pose or not public.get('actions_known'):
        return {}
    return {pose['facing']: 'forward' in public['actions'],
            opposite(pose['facing']): 'backward' in public['actions']}


def place_index(records):
    """Latest explicit public edge facts, with reciprocal provenance and conflicts."""
    cells = {}

    def cell(row, col):
        return cells.setdefault((row, col), {'cell': [row, col], 'edges': {},
                                            'steps': [], 'facings': []})

    def edge(row, col, direction, opened, step, source, reciprocal=False):
        target = cell(row, col)
        previous = target['edges'].get(direction)
        if previous and previous['step'] > step:
            return
        changed = list(previous.get('changed_at', [])) if previous else []
        if previous and previous['open'] != opened:
            changed.append(step)
        target['edges'][direction] = {'open': opened, 'step': step, 'source': source,
                                      'reciprocal': reciprocal, 'changed_at': changed}

    # A blocked edge does not establish that a neighboring cell was observed
    # (or even that it lies inside the maze). Do not create phantom places.
    for record in records.values():
        pose = record['public'].get('pose')
        if pose and not record['public'].get('question'):
            cell(pose['row'], pose['col'])

    for image_id, record in sorted(records.items(), key=lambda x: x[1]['step']):
        public = record['public']
        pose = public.get('pose')
        if not pose or public.get('question'):
            continue
        row, col, facing = pose['row'], pose['col'], pose['facing']
        here = cell(row, col)
        if record['step'] not in here['steps']:
            here['steps'].append(record['step'])
        if facing not in here['facings']:
            here['facings'].append(facing)
        transition = record.get('transition')
        if transition and transition['action'] in ('forward', 'backward'):
            before, after = transition['before'], transition['after']
            delta = (after['row'] - before['row'], after['col'] - before['col'])
            direction = next((d for d, offset in OFFSETS.items() if offset == delta), None)
            expected = before['facing'] if transition['action'] == 'forward' else opposite(before['facing'])
            if direction == expected:
                edge(before['row'], before['col'], direction, True, record['step'], image_id)
                edge(after['row'], after['col'], opposite(direction), True, record['step'], image_id, True)
        # Current legal actions are later evidence than arrival through an edge
        # (e.g. a passage could close after the move).
        for direction, opened in exits(public).items():
            edge(row, col, direction, opened, record['step'], image_id)
            dr, dc = OFFSETS[direction]
            nr, nc = row + dr, col + dc
            if nr >= 0 and nc >= 0 and (opened or (nr, nc) in cells):
                edge(nr, nc, opposite(direction), opened, record['step'], image_id, True)
    return cells


def frontal_endpoint(record, cells, central):
    """A conditional inference, never a silently persisted wall claim."""
    if record.get('wall') is not None:
        return {'wall': record['wall'], 'source': 'model claim',
                'claim_step': record.get('wall_step', record['step'])}
    pose = record['public'].get('pose')
    if not pose or not central:
        return {'wall': None, 'source': 'no model wall; crop is not central or pose is unknown'}
    row, col, direction = pose['row'], pose['col'], pose['facing']
    support = []
    grid = record['public'].get('grid')
    # Each iteration consumes a distinct saved/inferred cell; unknown stops it.
    for _ in range(len(cells) + 1):
        fact = cells.get((row, col), {}).get('edges', {}).get(direction)
        if fact is None:
            dr, dc = OFFSETS[direction]
            # Public dimensions are retained only by the selected SAVE_FRAME.
            # Do not guess through interior gaps or override an observed exit.
            if (grid and grid.get('zero_based') and 0 <= row < grid['rows'] and 0 <= col < grid['cols']
                    and not (0 <= row + dr < grid['rows'] and 0 <= col + dc < grid['cols'])):
                support.append({'cell': [row, col], 'direction': direction, 'open': False,
                                'step': record['step'], 'source': 'saved public zero-based grid dimensions',
                                'grid': grid})
                return {'wall': {'row': row, 'col': col, 'side': direction},
                        'source': 'conditional frontal ray ending at declared grid boundary',
                        'condition': 'Requires a frontal end-wall panel and a bounded grid without wrapping or portals; explicit observed exits override this inference.',
                        'support': support}
            return {'wall': None, 'source': 'unobserved gap along frontal ray', 'support': support}
        if any(step > record['step'] for step in fact['changed_at']):
            return {'wall': None, 'source': 'edge changed after image; old scene may be stale', 'support': support}
        support.append({'cell': [row, col], 'direction': direction,
                        'open': fact['open'], 'step': fact['step'], 'image_id': fact['source']})
        if not fact['open']:
            return {'wall': {'row': row, 'col': col, 'side': direction},
                    'source': 'conditional frontal ray through saved public exits',
                    'condition': 'Valid only if this crop depicts the frontal end wall, not a side panel.',
                    'support': support}
        dr, dc = OFFSETS[direction]
        row, col = row + dr, col + dc
    return {'wall': None, 'source': 'ray not resolved', 'support': support}


def group_walls(ranked):
    """Do not present repeated views of one conditional wall as corroboration."""
    groups = {}
    for view in ranked:
        wall = view['location'].get('wall')
        key = ('wall', wall['row'], wall['col'], wall['side']) if wall else ('view', view['image_id'], tuple(view['region']))
        if key not in groups:
            groups[key] = dict(view, other_views_at_wall=[])
        else:
            groups[key]['other_views_at_wall'].append({
                'image_id': view['image_id'], 'step': view['step'], 'region': view['region'],
                'similarity': view['similarity'], 'observer': view['observer'],
                'location': view['location'], 'other_sightings': view.get('other_sightings', [])})
    return list(groups.values())


class SceneMemory:
    def __init__(self, agent):
        self.agent = agent
        self.path = Path(agent._memory_dir) / 'scene_records.json'

    def load(self):
        data = json.loads(self.path.read_text()) if self.path.exists() else {'version': 2, 'records': {}}
        # Original DELETE_IMAGE or checkpoint restoration may remove an image.
        data['records'] = {k: v for k, v in data['records'].items()
                           if k in self.agent.visual.data['images']}
        return data

    def persist(self, data):
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(self.path)

    def metadata(self, image_id, record):
        entry = self.agent.visual.data['images'][image_id]
        return {'image_id': image_id, 'step': record['step'], 'label': entry['label'],
                'note': entry['note'], 'observer': record['public'].get('pose'),
                'exits': exits(record['public']), 'wall_claim': record.get('wall'),
                'regions': record['regions']}

    def save(self, label=None, note=None, region=None, wall=None):
        agent = self.agent
        if agent.current_image is None or not agent.current_frame_id:
            raise ValueError('No current public frame')
        public = agent.public_observation
        if public.get('question'):
            return {'saved': False, 'reason': 'Question image is a query, not a navigation sighting.'}
        regions = [region] if region is not None else content_regions(agent.current_image)
        for proposed in regions:
            pixel_box(agent.current_image, proposed)
        data = self.load()
        existing = next((key for key, record in data['records'].items()
                         if record['frame_id'] == agent.current_frame_id), None)
        pose = public.get('pose')
        if existing:
            old_entry = self.agent.visual.data['images'][existing]
            record = data['records'][existing]
            record['revisions'].append({'step': agent.step_count,
                                         'old_regions': record['regions'], 'old_wall': record.get('wall'),
                                         'old_label': old_entry['label'], 'old_note': old_entry['note']})
            self.agent.visual.annotate(existing, label, note)
            image_id = existing
            if region is not None:
                record['regions'] = [region]
                record['region_source'] = 'model update'
            if wall is not None:
                record['wall'] = wall
                record['wall_step'] = agent.step_count
        else:
            label = label if label is not None else 'view step %s pose %s' % (agent.step_count, pose)
            note = note if note is not None else ''
            saved = agent.visual.save(agent.current_image, agent.step_count, label, note,
                                      frame_id=agent.current_frame_id)
            image_id = saved['id']
            record = {'step': agent.step_count, 'frame_id': agent.current_frame_id,
                      'public': public, 'transition': agent.observed_transition,
                      'regions': regions, 'region_source': 'model' if region is not None else 'adaptive proposal',
                      'wall': wall, 'wall_step': agent.step_count if wall is not None else None,
                      'revisions': []}
            data['records'][image_id] = record
        self.persist(data)
        cells = place_index(data['records'])
        return {'saved': True, **self.metadata(image_id, record),
                'frontal_wall_candidate': frontal_endpoint(record, cells, any(r[0] < .5 < r[2] for r in record['regions'])),
                'warning': 'Candidate is conditional on a FRONTAL panel. Clear approach crops preserve pattern; close crops may be clipped. Use RECALL_IMAGE {} at questions.'}

    def annotate(self, image_id, label=None, note=None, region=None, wall=None, clear_wall=False):
        self.agent.visual._entry(image_id)
        data = self.load()
        record = data['records'].get(image_id)
        if region is not None:
            entry = self.agent.visual.data['images'][image_id]
            with Image.open(Path(self.agent._memory_dir) / entry['path']) as frame:
                pixel_box(frame, region)
        if record is None and (region is not None or wall is not None or clear_wall):
            return {'updated': False, 'reason': 'No public observation bundle for this legacy image.'}
        previous_entry = dict(self.agent.visual.data['images'][image_id])
        self.agent.visual.annotate(image_id, label, note)
        if record is not None:
            record['revisions'].append({'step': self.agent.step_count, 'old_regions': record['regions'],
                                         'old_wall': record.get('wall'),
                                         'old_label': previous_entry['label'], 'old_note': previous_entry['note']})
            if region is not None:
                record['regions'] = [region]
                record['region_source'] = 'model correction'
            if clear_wall or wall is not None:
                record['wall'] = None if clear_wall else wall
                record['wall_step'] = self.agent.step_count
            self.persist(data)
            return {'updated': True, **self.metadata(image_id, record)}
        return {'updated': True, 'image_id': image_id}

    def delete(self, image_id):
        data = self.load()
        result = self.agent.visual.delete(image_id)
        data['records'].pop(image_id, None)
        self.persist(data)
        return result

    def places(self, data):
        cells = place_index(data['records'])
        rows = []
        for key, cell in sorted(cells.items()):
            edges = cell['edges']
            rows.append({'cell': cell['cell'], 'observed_steps': cell['steps'],
                         'checked_facings': cell['facings'],
                         'open': [d for d in 'NESW' if d in edges and edges[d]['open']],
                         'blocked': [d for d in 'NESW' if d in edges and not edges[d]['open']],
                         'unknown': [d for d in 'NESW' if d not in edges],
                         'open_to_unobserved': [d for d in 'NESW' if d in edges and edges[d]['open']
                             and not cells.get((key[0]+OFFSETS[d][0], key[1]+OFFSETS[d][1]), {}).get('steps')],
                         'edge_evidence': edges})
        current = self.agent.public_observation
        pose = current.get('pose')
        conflicts = []
        if pose:
            stored = cells.get((pose['row'], pose['col']), {}).get('edges', {})
            conflicts = [{'direction': direction, 'stored': stored[direction], 'current_open': opened}
                         for direction, opened in exits(current).items()
                         if direction in stored and stored[direction]['open'] != opened]
        return {'places': rows, 'current': current, 'current_conflicts': conflicts,
                'warning': 'Only explicitly saved observations are indexed. Reciprocal edges assume maze passages are undirected. Current public actions override older facts. Unknown is not blocked. You choose exploration and actions.'}

    def _attach(self, views):
        if not views:
            return []
        ops, aliases, legend = [], [], []
        for index, view in enumerate(views[:3]):
            root, crop, thumb = 'f%s' % index, 'c%s' % index, 't%s' % index
            ops.extend([{'op': 'frame', 'id': view['frame_id'], 'as': root},
                        {'op': 'crop', 'src': root, 'box': view['box'], 'as': crop},
                        {'op': 'resize', 'src': crop, 'max_side': 320, 'as': thumb}])
            aliases.append(thumb)
            wall = view['location'].get('wall')
            location = ('%s,%s,%s' % (wall['row'], wall['col'], wall['side'])) if wall else 'wall unknown'
            qualifier = 'wall?' if 'conditional' in view['location']['source'] else 'wall'
            legend.append({'column': index + 1, 'image_id': view['image_id'], 'step': view['step'],
                           'wall_label': '%s %s' % (qualifier, location)})
        # The host's tile labels overlay the source pixels. Keep a text legend
        # instead: small distant crops must not disappear under a label bar.
        ops.append({'op': 'tile', 'srcs': aliases, 'cols': len(aliases), 'as': 'out'})
        self.agent.visual_request = {'ops': ops, 'output': 'out'}
        return legend

    def recall(self, image_id=None, mode='auto', region=None, limit=5):
        data = self.load()
        if image_id is not None:
            entry = self.agent.visual._entry(image_id)
            frame_id = entry.get('frame_id')
            if not frame_id:
                raise ValueError('Saved image has no host frame id')
            self.agent.visual_request = {'ops': [{'op': 'frame', 'id': frame_id, 'as': 'out'}], 'output': 'out'}
            record = data['records'].get(image_id)
            if record is None:
                return self.agent.visual._public(image_id, entry)
            cells = place_index(data['records'])
            return {**self.metadata(image_id, record),
                    'wall_candidates': [frontal_endpoint(record, cells, r[0] < .5 < r[2])
                                        for r in record['regions']]}
        if mode == 'places' or (mode == 'auto' and not self.agent.public_observation.get('question')):
            return self.places(data)
        if self.agent.current_image is None:
            raise ValueError('No current query image')
        if not data['records']:
            return {'matches': [], 'partial_matches': [], 'reason': 'No model-selected navigation frames were saved.'}
        query_region = region if region is not None else content_regions(self.agent.current_image)[0]
        query = descriptor(self.agent.current_image, query_region)
        cells = place_index(data['records'])
        candidates = {}
        # Collapse normalized crops equal to three decimal places before
        # expensive registration. This is an index key, not an identity claim.
        # All sightings retain their own observer and wall provenance.
        for key, record in data['records'].items():
            if not record['public'].get('pose') or record['public'].get('question'):
                continue
            entry = self.agent.visual.data['images'][key]
            with Image.open(Path(self.agent._memory_dir) / entry['path']) as frame:
                for proposed in record['regions']:
                    source = descriptor(frame, proposed)
                    if not np.any(np.abs(source) > 1e-8):
                        continue
                    fingerprint = hashlib.sha256(np.round(source, 3).tobytes()).hexdigest()
                    box = pixel_box(frame, proposed)
                    central = proposed[0] < 0.5 < proposed[2]
                    view = {**self.metadata(key, record), 'region': proposed,
                            'frame_id': record['frame_id'], 'box': box,
                            'frame_clipped': box[0] == 0 or box[1] == 0 or box[2] == frame.width or box[3] == frame.height,
                            'central_crop': central, 'region_source': record['region_source'],
                            'location': frontal_endpoint(record, cells, central)}
                    group = candidates.setdefault(fingerprint, {'source': source, 'views': []})
                    group['views'].append(view)
        ranked = []
        for group in candidates.values():
            score = compare(group['source'], query)
            # Prefer a located, non-clipped view when identical pixels repeat.
            views = sorted(group['views'], key=lambda v: (v['frame_clipped'], v['location']['wall'] is None, v['step']))
            representative = dict(views[0], **score)
            representative['other_sightings'] = [{'image_id': v['image_id'], 'observer': v['observer'],
                                                   'location': v['location']} for v in views[1:]]
            ranked.append(representative)
        ranked.sort(key=lambda r: r['similarity'], reverse=True)
        bounded = group_walls([r for r in ranked if not r['frame_clipped']])[:limit]
        partial = group_walls([r for r in ranked if r['frame_clipped']])[:min(limit, 3)]
        # Lead with complete-frame-bounded crops, and display a clipped candidate
        # only if fewer than three bounded alternatives exist.
        attached_order = self._attach((bounded + partial)[:3])
        return {'query_region': query_region, 'matches': bounded, 'partial_matches': partial,
                'attached_order': attached_order,
                'distinct_crop_comparisons': len(candidates),
                'interpretation': 'The historical tile columns follow attached_order from left to right; its pixels have no overlaid labels. Similarity is a local registration score, NOT a probability or an answer. matches have crops bounded inside the frame, not proof of a complete panel. partial_matches touch a frame edge and can correlate with unrelated repeated blocks. Prefer exact complete-pattern evidence; compare alternatives and attached crops. Equal pixels do not independently corroborate identity. Wall candidates are separate, conditional spatial inferences; unknown stays unknown. For a side panel verify and annotate its wall. Decide the answer yourself.'}
