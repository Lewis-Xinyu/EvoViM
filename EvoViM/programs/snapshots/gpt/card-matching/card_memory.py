"""Explicit visual-card file operations. No model calls or world actions.

Metadata is readable with original file tools; crop pixels stay in scratch
subdirectories and can reach the model only via RECALL_CARD_VISUAL's host program.
Geometry is inferred per episode; nonzero distances are evidence, not identity.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
from registry import Tool


def empty_ledger():
    return {'schema_version': 1, 'grid': None, 'cards': {}, 'nonmatches': {},
            'empty_templates': {}, 'last_log': None}


def parse_observation(text, step, previous_pending=None):
    dim = re.search(r'(\d+)\s*[x×]\s*(\d+)\s+board', text, re.I)
    dimensions = [int(dim[1]), int(dim[2])] if dim else None
    flipped = re.search(r'You flipped ([a-z][A-Z])(?: and ([a-z][A-Z]))?\.', text)
    invalid = re.search(r'Invalid flip ([a-z][A-Z]):', text)
    revealed, removed = [], []
    pending = None
    outcome = 'unparsed'
    if step == 0:
        outcome = 'initial_observation'
    elif flipped:
        revealed = [p for p in flipped.groups() if p]
        if len(revealed) == 2 and re.search(r'\bMatch!', text):
            outcome, removed = 'match', list(revealed)
        elif len(revealed) == 2 and 'No match.' in text:
            outcome = 'mismatch'
        elif len(revealed) == 1 and 'second card' in text.lower():
            outcome, pending = 'first_reveal', revealed[0]
    elif invalid:
        outcome = 'invalid'
        if 'already empty' in text:
            removed = [invalid[1]]
        if 'second card' in text.lower():
            pending = invalid[1] if 'already flipped' in text else previous_pending
            if pending:
                revealed = [pending]
    next_flip = ('second' if 'second card' in text.lower() and
                 outcome not in ('match', 'mismatch') else
                 'first' if 'first card' in text.lower() or outcome in ('match', 'mismatch') else 'unknown')
    return {'observation_step': int(step), 'action_step': int(step)-1 if step else None,
            'dimensions': dimensions, 'outcome': outcome, 'revealed': revealed,
            'removed': removed, 'pending': pending, 'next_flip': next_flip}


def _positions(rows, cols):
    if not (1 <= rows <= 26 and 1 <= cols <= 26):
        raise ValueError('Coordinate grid must fit the observed lowercase/uppercase labels')
    return [chr(97+r)+chr(65+c) for r in range(rows) for c in range(cols)]


def _cluster(values, separation):
    groups = []
    for v in sorted(values):
        if not groups or v - groups[-1][-1] > separation:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [float(np.median(g)) for g in groups]


def detect_grid(image, dimensions, bounds=None):
    if image is None or not dimensions:
        raise ValueError('Current image and public board dimensions are required')
    rows, cols = dimensions
    _positions(rows, cols)
    w, h = image.size
    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError('grid_bounds must contain four integers')
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('grid_bounds must lie inside the current image')
        # Explicit LLM calibration means bounds cover all cells including gaps.
        xs = [x0+(c+.5)*(x1-x0)/cols for c in range(cols)]
        ys = [y0+(r+.5)*(y1-y0)/rows for r in range(rows)]
        cw, ch = (x1-x0)/cols, (y1-y0)/rows
        method = 'model_supplied_bounds'
    else:
        pixels = np.asarray(image.convert('RGB'))
        # Background is measured, not a fixed RGB or mission color.
        perimeter = np.concatenate([pixels[0], pixels[-1], pixels[:,0], pixels[:,-1]])
        colors, counts = np.unique(perimeter, axis=0, return_counts=True)
        background = colors[int(np.argmax(counts))]
        components, _ = label(np.any(pixels != background, axis=2))
        boxes = []
        for region in find_objects(components):
            if region is None:
                continue
            yy, xx = region
            # A card must occupy a substantial fraction of a grid cell.
            # These scale-relative guards reject text labels, not picture colors.
            if (w/(2*cols) <= xx.stop-xx.start <= 2*w/cols and
                    h/(2*rows) <= yy.stop-yy.start <= 2*h/rows):
                boxes.append([xx.start, yy.start, xx.stop, yy.stop])
        xs = _cluster([(b[0]+b[2])/2 for b in boxes], w/(2*cols))
        ys = _cluster([(b[1]+b[3])/2 for b in boxes], h/(2*rows))
        if len(xs) != cols or len(ys) != rows:
            raise ValueError('Cannot infer all grid centers; supply grid_bounds or use original frame tools')
        cw = float(np.median([b[2]-b[0] for b in boxes]))
        ch = float(np.median([b[3]-b[1] for b in boxes]))
        method = 'measured_components'
    boxes = {}
    # Inner 80% avoids card borders/highlights; the rule is scale-relative.
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            box = [int(round(x-.4*cw)), int(round(y-.4*ch)),
                   int(round(x+.4*cw)), int(round(y+.4*ch))]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError('Inferred crop outside image')
            boxes[chr(97+r)+chr(65+c)] = box
    return {'rows': rows, 'cols': cols, 'image_size': [w,h], 'boxes': boxes,
            'method': method}


def fingerprint(crop):
    rgb = crop.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def feature(crop):
    # Fixed general-purpose computation resolution, not an episode-fitted cutoff.
    return np.asarray(crop.convert('RGB').resize((24,24), Image.Resampling.BILINEAR), dtype=np.float32)/255


def distance(a, b):
    return float(np.mean(np.abs(a-b)))


def load_ledger(agent):
    raw = agent.memory.read('card_memory.json')
    if raw.startswith('[FILE IS EMPTY') or raw.startswith('[ERROR:'):
        return empty_ledger()
    data = json.loads(raw)
    if data.get('schema_version') != 1 or not isinstance(data.get('cards'), dict):
        raise ValueError('Invalid card ledger; correct card_memory.json with WRITE')
    return data


def _grid(agent, data, bounds=None, recalibrate=False):
    image = agent.current_image
    if image is None:
        raise ValueError('No current observation image')
    dimensions = agent.card_observation.get('dimensions')
    grid = data.get('grid')
    if (not recalibrate and grid and grid['image_size'] == list(image.size) and
            dimensions == [grid['rows'], grid['cols']]):
        return grid
    if recalibrate:
        # Deliberate elected replacement. A supplied rectangle is an explicit
        # override only here, never a side effect of an ordinary identity write.
        return detect_grid(image, dimensions, bounds)
    try:
        # Prefer native measured cells when starting a crop coordinate system.
        # Approximate model bounds are a fallback, not a new representation.
        return detect_grid(image, dimensions)
    except ValueError:
        if bounds is None:
            raise
        return detect_grid(image, dimensions, bounds)


def geometry_id(grid):
    """Namespace exact identities by their native crop coordinate system."""
    payload = {k:grid[k] for k in ['rows','cols','image_size','boxes']}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def compatible_entry(position, entry, grid):
    if entry.get('bbox') != grid['boxes'].get(position):
        return False
    if entry.get('geometry_id'):
        return entry['geometry_id'] == geometry_id(grid)
    # Read old ledgers without rewriting them. Legacy bboxes provide enough
    # evidence to reject incompatible historical identity crops.
    return True


def _crop_path(agent, position):
    if not re.fullmatch(r'[a-z][A-Z]', position):
        raise ValueError('Invalid coordinate')
    return Path(agent._memory_dir)/'card_visual'/f'{position}.png'


def _pair(a, b):
    return '|'.join(sorted([a,b]))


def _public(position, entry):
    return {k: entry[k] for k in ['status','observed_step','action_step','frame_id','note',
                                 'removed_step','removal_reason']
            if k in entry} | {'position': position, 'signature': entry.get('fingerprint','')[:16]}


def _persist(agent, data):
    # Only called as a consequence of a model-invoked STORE operation.
    agent.memory.write('card_memory.json', json.dumps(data, ensure_ascii=False, indent=2))
    lines = []
    for position, entry in sorted(data['cards'].items()):
        lines.append(json.dumps(_public(position, entry), ensure_ascii=False))
    agent.memory.write('cards_index.txt', '\n'.join(lines))


def _empty_reference(agent, grid, position):
    """Template from a publicly confirmed empty location; no fitted palette.

    Exact uniformity conservatively rejects pictures/animated blank slots.
    Nonuniform empties still retain authoritative text tombstones.
    """
    # A just-completed match can still display its pictures. Wait for a later
    # observation, or an explicit invalid-empty response, before sampling it.
    if (agent.card_observation['outcome'] == 'match' and
            position in agent.card_observation['revealed']):
        return None
    crop = agent.current_image.crop(grid['boxes'][position]).convert('RGB')
    pixels = np.asarray(crop)
    if not np.all(pixels == pixels[0, 0]):
        return None
    return fingerprint(crop), {'position':position, 'frame_id':agent.current_frame_id,
        'observed_step':agent.card_observation['observation_step'],
        'bbox':grid['boxes'][position], 'crop_size':list(crop.size)}


def board_eligibility(agent, data, grid):
    """Elected read-only computation: current pixels veto stale location claims.

    No identity, history, template, correction or action is persisted here.
    """
    obs = agent.card_observation
    templates = dict(data.get('empty_templates', {}))
    visible_cards = set(obs['revealed']) - set(obs['removed'])
    anchors = set(obs['removed']) | {p for p,e in data['cards'].items()
                                     if e.get('status') == 'removed'}
    anchors -= visible_cards
    for p in sorted(anchors):
        if p in grid['boxes']:
            ref = _empty_reference(agent, grid, p)
            if ref:
                templates.setdefault(*ref)
    signatures = {p:fingerprint(agent.current_image.crop(box))
                  for p,box in grid['boxes'].items()}
    # Public reveals override template equality. Disable any reference that
    # aliases a currently revealed uniform picture rather than hiding that card.
    collisions = {signatures[p] for p in visible_cards if p in signatures} & set(templates)
    for sig in collisions:
        templates.pop(sig, None)
    visual_empty = sorted(p for p,sig in signatures.items() if sig in templates)
    removed = sorted(anchors | set(visual_empty))
    stale = sorted(p for p in visual_empty
                   if data['cards'].get(p,{}).get('status') == 'active')
    return {'removed':removed, 'visual_empty':visual_empty,
            'stale_active':stale, 'references':list(templates.values()),
            'template_conflicts':len(collisions),
            'observation_step':obs['observation_step']}


def store_cards(agent, positions=None, forget=None, note='', grid_bounds=None,
                sync_empty=False, clear_empty_templates=False, recalibrate=False):
    obs = agent.card_observation
    if obs.get('action_step') is None:
        return {'stored': [], 'reason': 'No previous action outcome at step zero'}
    data = load_ledger(agent)
    selected = list(obs['revealed'] if positions is None else positions)
    # A caller cannot invent an identity at a face-down coordinate.
    if set(selected) - set(obs['revealed']):
        raise ValueError('positions must be revealed in the current observed response')
    forgotten = list(dict.fromkeys(forget or []))
    selected = [p for p in selected if p not in forgotten]
    grid = _grid(agent, data, grid_bounds, recalibrate)
    for p in selected + forgotten + obs['removed']:
        if p not in grid['boxes']:
            raise ValueError('Coordinate outside the observed board: '+p)
    previous_grid = data.get('grid')
    grid_changed = bool(previous_grid and geometry_id(previous_grid) != geometry_id(grid))
    data['grid'] = grid
    if clear_empty_templates or grid_changed:
        data['empty_templates'] = {}
    templates = data.setdefault('empty_templates', {})
    for sig, ref in list(templates.items()):
        if ref.get('position') in forgotten:
            del templates[sig]
    learned_empty = []
    # Elected STORE can ground a reference in previously elected tombstones,
    # using this observation's pixels. No reference is harvested in _format_obs.
    anchors = set(obs['removed']) | {p for p,e in data['cards'].items()
                                     if e.get('status') == 'removed'}
    if clear_empty_templates:
        anchors = set()
    for p in sorted(anchors - set(forgotten) - set(selected)):
        ref = _empty_reference(agent, grid, p)
        if ref and ref[0] not in templates:
            templates[ref[0]] = ref[1]
            learned_empty.append(p)
    root = Path(agent._memory_dir)/'card_visual'
    root.mkdir(exist_ok=True)
    for p in forgotten:
        data['cards'].pop(p, None)
        _crop_path(agent,p).unlink(missing_ok=True)
        data['nonmatches'] = {key: value for key,value in data['nonmatches'].items()
                              if p not in key.split('|')}
    stored, corrected = [], []
    for p in dict.fromkeys(selected):
        crop = agent.current_image.crop(grid['boxes'][p]).convert('RGB')
        if p in obs['removed'] and np.all(np.asarray(crop) == np.asarray(crop)[0,0]):
            # Some renderers hide the matched identity immediately. Keep an
            # earlier identity if one exists rather than saving a blank as it.
            continue
        sig = fingerprint(crop)
        old = data['cards'].get(p,{})
        entry = {'status': 'active', 'fingerprint': sig, 'bbox': grid['boxes'][p],
                 'geometry_id': geometry_id(grid),
                 'frame_id': agent.current_frame_id, 'observed_step': obs['observation_step'],
                 'action_step': obs['action_step'], 'note': note or old.get('note','')}
        if old.get('fingerprint') and old['fingerprint'] != sig:
            entry['superseded'] = {k:old[k] for k in ['fingerprint','frame_id','observed_step']}
            corrected.append(p)
            # A new identity supersedes negative comparisons involving the old one.
            data['nonmatches'] = {key: value for key,value in data['nonmatches'].items()
                                  if p not in key.split('|')}
        crop.save(_crop_path(agent,p), format='PNG')
        data['cards'][p] = entry
        stored.append(p)
    logged_removed = [p for p in obs['removed'] if p not in forgotten]
    for p in logged_removed:
        entry = data['cards'].setdefault(p,{'observed_step':obs['observation_step'],
            'action_step':obs['action_step'],'frame_id':agent.current_frame_id,'note':note})
        entry.update(status='removed', removed_step=obs['observation_step'],
                     removal_reason=obs['outcome'])
    synchronized = []
    if sync_empty:
        eligibility = board_eligibility(agent, data, grid)
        for p in eligibility['visual_empty']:
            if p in forgotten or data['cards'].get(p,{}).get('status') == 'removed':
                continue
            entry = data['cards'].setdefault(p, {})
            entry.update(status='removed', removed_step=obs['observation_step'],
                         removal_reason='current_pixels_equal_confirmed_empty_template')
            synchronized.append(p)
    if (obs['outcome'] == 'mismatch' and len(obs['revealed']) == 2 and
            not set(obs['revealed']).intersection(forgotten)):
        data['nonmatches'][_pair(*obs['revealed'])] = obs['observation_step']
    data['last_log'] = {k:obs[k] for k in ['observation_step','action_step','outcome','pending','removed']}
    _persist(agent, data)
    return {'stored': stored, 'corrected': corrected, 'forgotten': forgotten,
            'removed': logged_removed, 'last_log': data['last_log'],
            'learned_empty_templates':learned_empty, 'synchronized_empty':synchronized,
            'grid_method': grid['method'], 'files': ['card_memory.json','cards_index.txt'],
            'geometry': {'id':geometry_id(grid)[:16], 'changed':grid_changed,
                'bounds_hint_kept_existing_grid': bool(grid_bounds is not None and
                    previous_grid and not recalibrate and not grid_changed),
                'identity_requires_reobserve': sorted(p for p,e in data['cards'].items()
                    if e.get('status')=='active' and not compatible_entry(p,e,grid)),
                'note':'Existing geometry is stable during ordinary STORE. Elect recalibrate=true to replace it; incompatible identities remain available for recall but require a new observed reveal for matching.'},
            'remembered_count': len(data['cards'])}


def query_cards(agent, position=None, limit=8, positions=None):
    if positions is not None:
        if len(positions) != 1 or (position is not None and position != positions[0]):
            raise ValueError('positions is a single-position alias; use position for one target')
        position = positions[0]
    data = load_ledger(agent)
    obs = agent.card_observation
    try:
        grid = _grid(agent, data)
    except ValueError as exc:
        return {'status': 'grid_unavailable', 'reason':str(exc), 'next_flip':obs.get('next_flip'),
                'pending':obs.get('pending'), 'stored':[_public(p,e) for p,e in sorted(data['cards'].items())]}
    cards = data['cards']
    eligibility = board_eligibility(agent, data, grid)
    removed = eligibility['removed']
    incompatible = sorted(p for p,e in cards.items() if e['status']=='active'
                          and p not in removed and not compatible_entry(p,e,grid))
    active = {p:e for p,e in cards.items() if e['status']=='active' and p not in removed
              and compatible_entry(p,e,grid)}
    excluded = set(removed) | ({obs['pending']} if obs.get('pending') else set())
    # Current visible reveal pixels supplement, but are NOT written to memory.
    target = position or obs.get('pending')
    matches, neighbors, conflicts = [], [], []
    if target:
        if target not in grid['boxes']:
            raise ValueError('Coordinate outside observed grid')
        if target in obs['revealed'] and target not in removed:
            crop = agent.current_image.crop(grid['boxes'][target])
            sig, target_feature = fingerprint(crop), feature(crop)
            source = {'frame_id':agent.current_frame_id, 'observed_step':obs['observation_step'], 'saved':False}
        elif target in active and _crop_path(agent,target).exists():
            with Image.open(_crop_path(agent,target)) as image:
                sig, target_feature = active[target]['fingerprint'], feature(image)
            source = {'frame_id':active[target]['frame_id'], 'observed_step':active[target]['observed_step'], 'saved':True}
        else:
            sig, target_feature, source = None, None, None
        if sig is not None:
            for p,e in active.items():
                if p==target or p in excluded:
                    continue
                if not _crop_path(agent,p).exists():
                    continue
                with Image.open(_crop_path(agent,p)) as image:
                    d = distance(target_feature,feature(image))
                row = _public(p,e) | {'distance':round(d,6),'exact_pixels':sig==e['fingerprint']}
                if _pair(target,p) in data['nonmatches']:
                    if row['exact_pixels']:
                        conflicts.append(row | {'reason':'Exact crop conflicts with observed mismatch; inspect/correct'})
                    continue
                neighbors.append(row)
                if row['exact_pixels']:
                    matches.append(row)
    else:
        source = None
    groups = defaultdict(list)
    for p,e in active.items():
        if e.get('fingerprint') and p not in excluded:
            groups[e['fingerprint']].append(p)
    pairs = []
    for group in groups.values():
        for i,a in enumerate(sorted(group)):
            for b in sorted(group)[i+1:]:
                if _pair(a,b) not in data['nonmatches']:
                    pairs.append([a,b])
                else:
                    conflicts.append({'positions':[a,b], 'reason':'Identical stored crops contradict mismatch'})
    # The list is a retrieval result, not a scripted route or action selection.
    unknown = [p for p in grid['boxes'] if (p not in cards or p in incompatible)
               and p not in excluded]
    eligible = [p for p in grid['boxes'] if p not in excluded]
    return {'next_flip':obs['next_flip'], 'pending':obs['pending'],
            'pending_exact_mates':[r['position'] for r in matches] if target==obs.get('pending') else [],
            'exact_matches':matches, 'known_exact_pairs':pairs[:limit],
            'eligible_unremembered':unknown[:limit], 'unremembered_count':len(unknown),
            'eligible_known_exact_pairs':pairs[:limit],
            'geometry':{'id':geometry_id(grid)[:16], 'method':grid['method'],
                        'identity_requires_reobserve':incompatible},
            'eligibility':{'current_visual_empty_count':len(eligibility['visual_empty']),
                          'stale_active_excluded':eligibility['stale_active'],
                          'template_conflicts':eligibility['template_conflicts'],
                          'template_available':bool(eligibility['references'])},
            'target':target,
            'target_source':source,
            'nearest_nonconfirmed':sorted(neighbors,key=lambda r:(r['distance'],r['position']))[:limit],
            'conflicts':conflicts[:limit], 'removed':removed,
            'unremembered':unknown, 'eligible_positions':eligible,
            'active_count':len(active), 'last_log':data['last_log'],
            'warning':'Current empty-slot evidence vetoes stale memory. Choose only from eligible_positions. Exact pairs refer to two eligible cells; similarity alone is unconfirmed. QUERY makes no persistent corrections; elect STORE in LOG to synchronize.'}


def recall_cards(agent, positions=None):
    data = load_ledger(agent)
    if positions is None:
        positions = [p for p,e in sorted(data['cards'].items(),
                     key=lambda item: (-item[1].get('observed_step',-1),item[0]))
                     if e.get('frame_id') and e.get('bbox')][:4]
    positions = list(dict.fromkeys(positions))
    if not positions:
        return {'recalled':[], 'reason':'No saved card crops yet; STORE_CARD_VISUAL is an explicit LOG operation'}
    ops, aliases, recalled = [], [], []
    for i,p in enumerate(positions):
        entry = data['cards'].get(p)
        if not entry or not entry.get('frame_id') or not entry.get('bbox'):
            raise ValueError('No saved source crop for '+p)
        # Leave space above the picture for the host's tile label. The renderer
        # currently supports shrinking only, so do not claim to enlarge crops.
        prefix = 'c'+str(i)
        box = list(entry['bbox'])
        box[1] = max(0, box[1]-20)
        ops.extend([{'op':'frame','id':entry['frame_id'],'as':prefix+'f'},
                    {'op':'crop','src':prefix+'f','box':box,'as':prefix+'b'}])
        aliases.append(prefix+'b')
        recalled.append(_public(p,entry))
    # Tile supports labelled source images without additional annotation ops.
    ops.append({'op':'tile','srcs':aliases,'cols':min(4,len(aliases)),
                'labels':[p+' @'+str(data['cards'][p]['observed_step']) for p in positions],'as':'out'})
    agent.visual_request = {'ops':ops,'output':'out'}
    return {'recalled':recalled, 'delivery':'Host attaches these historical crops to the next model request'}


def install_cards(agent, registry):
    obj = {'type':'object','additionalProperties':False}
    coords = {'type':'array','items':{'type':'string','maxLength':2},'maxItems':676}
    registry.register(Tool('STORE_CARD_VISUAL',
        'Explicitly save observed card identities and prior-action outcomes. Use {} after reveals, matches AND invalid flips. Match writes update removal status; later observations of elected tombstones ground empty references. sync_empty=true elects reconciliation. forget deletes chosen entries. clear_empty_templates discards references. Ordinary writes preserve native crop geometry: grid_bounds is an initialization fallback. Only elected recalibrate=true replaces an established grid, using measured geometry or supplied [left,top,right,bottom]; old incompatible identities then require reobservation and remain recallable.',
        {**obj,'properties':{'positions':coords,'forget':coords,'note':{'type':'string','maxLength':500},
            'grid_bounds':{'type':'array','items':{'type':'integer','minimum':0},'maxItems':4},
            'sync_empty':{'type':'boolean'}, 'clear_empty_templates':{'type':'boolean'},
            'recalibrate':{'type':'boolean'}},'required':[]},
        lambda **kwargs:store_cards(agent,**kwargs),('LOG',),'memory_write','metadata'))
    registry.register(Tool('QUERY_CARD_VISUAL',
        'Read saved identities with current-board eligibility and crop-geometry checks. {} retrieves pending_exact_mates and known_exact_pairs first, then eligible_unremembered. Similarity alone is unconfirmed. Incompatible saved geometry is flagged for reobservation, never silently compared or persistently corrected. position queries one card; positions:[coord] is an alias. Choose your own ACTION.',
        {**obj,'properties':{'position':{'type':'string','maxLength':2},
                            'positions':{**coords,'maxItems':1},
                            'limit':{'type':'integer','minimum':1,'maximum':16}},'required':[]},
        lambda **kwargs:query_cards(agent,**kwargs),('PLAN',),'memory_read','metadata'))
    registry.register(Tool('RECALL_CARD_VISUAL',
        'Recall isolated, labelled historical card crops by positions through the host visual channel. {} recalls up to four most recently saved cards; specify up to eight positions for comparison. Metadata is provenance, not proof of current eligibility.',
        {**obj,'properties':{'positions':{**coords,'maxItems':8}},'required':[]},
        lambda **kwargs:recall_cards(agent,**kwargs),('LOG','PLAN'),'memory_read','image'))
