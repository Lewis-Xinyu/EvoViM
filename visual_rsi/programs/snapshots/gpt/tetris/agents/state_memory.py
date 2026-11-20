"""Explicit, replaceable memory records and optional numeric visual measurements.

No action selection, OCR, game-specific geometry, colors or automatic snapshots.
All durable writes below are initiated by model-invoked tools.
"""
import json
import math
import re

import numpy as np

FILE = 'state_memory.json'
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def empty_store():
    return {'version': 1, 'grid_config': None, 'entries': {}}


def load(agent):
    raw = agent.memory.read(FILE)
    if raw.startswith('[ERROR:') or raw == '[FILE IS EMPTY]':
        return empty_store()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('Unsupported state memory schema; use WRITE to repair the file')
    if not isinstance(data.get('entries'), dict) or 'grid_config' not in data:
        raise ValueError('State memory needs entries and grid_config; use WRITE to repair the file')
    return data


def persist(agent, data):
    # One ordinary episode memory file: legacy READ/SEARCH/WRITE remain useful.
    return agent.memory.write(FILE, json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def key_name(key):
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,48}', key):
        raise ValueError('key must contain 1-48 letters, digits, underscores, dots or hyphens')
    return key


def coordinate(value, cols, rows):
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(v, int) or isinstance(v, bool) for v in value) or
            not 0 <= value[0] < cols or not 0 <= value[1] < rows):
        raise ValueError('cell coordinates are [column,row], zero based, inside the grid')
    return value


def sample_grid(image, cfg):
    """Read central patch medians; never interpret a color as an object."""
    if image is None:
        raise ValueError('No current image')
    width, height = image.size
    if [width, height] != cfg['image_size']:
        raise ValueError('Image dimensions changed; reconfigure the grid from this observation')
    left, top, right, bottom = cfg['bbox']
    cols, rows = cfg['columns'], cfg['rows']
    pixels = np.asarray(image.convert('RGB'))
    dx, dy = (right-left)/cols, (bottom-top)/rows
    result = []
    for y in range(rows):
        row = []
        for x in range(cols):
            cx, cy = left+(x+0.5)*dx, top+(y+0.5)*dy
            # Central half of each cell excludes common grid lines. This is a
            # dimensionless sampling rule, not a fitted color/pixel threshold.
            x0, x1 = max(0, int(cx-dx/4)), min(width, max(int(cx-dx/4)+1, int(cx+dx/4)))
            y0, y1 = max(0, int(cy-dy/4)), min(height, max(int(cy-dy/4)+1, int(cy+dy/4)))
            rgb = np.median(pixels[y0:y1, x0:x1].reshape(-1, 3), axis=0)
            row.append([int(v) for v in rgb])
        result.append(row)
    return result


def configure_grid(agent, bbox=None, rows=None, columns=None, empty_cell=None, ghost_cell=None):
    data = load(agent)
    image = getattr(agent, 'current_image', None)
    if bbox is None:
        return {'configuration': data['grid_config'],
                'image_size': list(image.size) if image is not None else None,
                'help': 'To configure, provide bbox [left,top,right,bottom] in current image pixels, rows, columns and an empty_cell [column,row]. Optional ghost_cell identifies a non-solid preview by its sampled RGB. No geometry or color is inferred for you.'}
    if image is None:
        raise ValueError('No current image')
    width, height = image.size
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for v in bbox) or
            not 0 <= bbox[0] < bbox[2] <= width or not 0 <= bbox[1] < bbox[3] <= height):
        raise ValueError('bbox must be inside the current image')
    if (not isinstance(rows, int) or isinstance(rows, bool) or not isinstance(columns, int) or isinstance(columns, bool)
            or not 1 <= rows <= 64 or not 1 <= columns <= 64 or rows*columns > 2048):
        raise ValueError('rows and columns must be 1-64; at most 2048 cells')
    coordinate(empty_cell, columns, rows)
    if ghost_cell is not None:
        coordinate(ghost_cell, columns, rows)
    cfg = {'bbox': bbox, 'rows': rows, 'columns': columns, 'image_size': [width, height],
           'configured_step': agent.step_count, 'frame_id': agent.current_frame_id}
    grid = sample_grid(image, cfg)
    cfg['empty_rgb'] = grid[empty_cell[1]][empty_cell[0]]
    cfg['ghost_rgb'] = grid[ghost_cell[1]][ghost_cell[0]] if ghost_cell is not None else None
    if cfg['ghost_rgb'] == cfg['empty_rgb']:
        raise ValueError('ghost and empty samples coincide; inspect the reference cells')
    data['grid_config'] = cfg
    persist(agent, data)
    return {'configuration': cfg, 'warning': 'Model-specified geometry and role samples are unverified. Inspect READ_GRID against the screenshot. Reconfiguring does not rewrite older snapshots.'}


def current_snapshot(agent, data=None):
    data = data if data is not None else load(agent)
    cfg = data['grid_config']
    if cfg is None:
        return None
    return {'step': agent.step_count, 'frame_id': agent.current_frame_id,
            'config': cfg, 'rgb': sample_grid(agent.current_image, cfg)}


def grid_report(snapshot, active_cells=None):
    cfg, grid = snapshot['config'], snapshot['rgb']
    rows, cols = cfg['rows'], cfg['columns']
    empty, ghost = cfg['empty_rgb'], cfg['ghost_rgb']
    colors = sorted({tuple(c) for row in grid for c in row if c != empty and c != ghost})
    labels = {rgb: SYMBOLS[i] if i < len(SYMBOLS) else '?' for i, rgb in enumerate(colors)}
    active = set()
    for cell in active_cells or []:
        coordinate(cell, cols, rows)
        active.add(tuple(cell))
    solid = [[c != empty and c != ghost and (x,y) not in active for x,c in enumerate(row)] for y,row in enumerate(grid)]
    symbolic = [''.join('.' if c == empty else '~' if c == ghost else labels[tuple(c)] for c in row) for row in grid]
    # Same-RGB components are measurements, not assertions about active pieces.
    seen, components = set(), []
    for y in range(rows):
        for x in range(cols):
            if (x,y) in seen or grid[y][x] == empty or grid[y][x] == ghost:
                continue
            color, stack, cells = grid[y][x], [(x,y)], []
            while stack:
                cx, cy = stack.pop()
                if (cx,cy) in seen or grid[cy][cx] != color:
                    continue
                seen.add((cx,cy))
                cells.append([cx,cy])
                stack.extend((nx,ny) for nx,ny in ((cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1))
                             if 0 <= nx < cols and 0 <= ny < rows and (nx,ny) not in seen)
            components.append({'symbol': labels[tuple(color)], 'cells': sorted(cells, key=lambda c:(c[1],c[0]))})
    answer = {'observed_step': snapshot['step'], 'frame_id': snapshot['frame_id'],
              'coordinates': '[column,row], zero based; row 0 at top; . = sampled empty; ~ = sampled ghost',
              'grid': symbolic, 'palette': {label: list(rgb) for rgb,label in labels.items()},
              'components': components[:32], 'component_count': len(components),
              'nonempty_nonghost_count': sum(c != empty and c != ghost for row in grid for c in row),
              'role_warning': 'Palette letters are colors only and may change between frames; compare RGB values. Identify the active piece from current pixels; connected components can merge settled pieces. Ghost is excluded only if configured. Objects outside the bbox, including a spawning piece above the grid, are not measured.'}
    if active_cells is not None:
        heights, holes = [], []
        for x in range(cols):
            occupied = [y for y in range(rows) if solid[y][x]]
            first = min(occupied) if occupied else rows
            heights.append(rows-first)
            holes.append(sum(not solid[y][x] for y in range(first, rows)))
        answer['after_excluding_model_active_cells'] = {'active_cells': active_cells, 'column_heights': heights,
            'holes_per_column': holes, 'row_fills': [sum(row) for row in solid],
            'bottom_gaps': [x for x in range(cols) if not solid[-1][x]],
            'warning': 'Only valid as settled-board metrics if your exclusion list and empty/ghost references are correct.'}
    return answer


def compare_snapshots(before, after):
    if before is None or after is None:
        return {'available': False, 'reason': 'No captured grid for this entry or no current configuration'}
    if before['config'] != after['config']:
        return {'available': False, 'reason': 'Configurations differ; recalibration invalidates numeric comparison'}
    changes = []
    for y,(old,new) in enumerate(zip(before['rgb'],after['rgb'])):
        for x,(a,b) in enumerate(zip(old,new)):
            if a != b:
                changes.append({'cell': [x,y], 'before_rgb': a, 'now_rgb': b})
    return {'available': True, 'from_step': before['step'], 'to_step': after['step'],
            'consecutive': before['step'] == after['step']-1,
            'changed_cells': len(changes), 'changes': changes[:64],
            'warning': 'Color changes measure visible change, not progress, locking or score. A nonconsecutive comparison cannot isolate the previous action.'}


def read_grid(agent, active_cells=None, compare_key=None):
    data = load(agent)
    snap = current_snapshot(agent, data)
    if snap is None:
        return {'available': False, 'help': 'Use GRID_CONFIG from this screenshot first, or use normal image observation and text memory.'}
    answer = grid_report(snap, active_cells)
    if compare_key is not None:
        entry = data['entries'].get(key_name(compare_key))
        answer['comparison'] = compare_snapshots(entry.get('grid') if entry else None, snap)
    return answer


def put(agent, key, text, kind, capture_grid=False, pin=False):
    data = load(agent)
    key_name(key)
    if kind == 'result' and (getattr(agent, '_memory_phase', None) != 'LOG' or
                             agent.step_count <= 0 or getattr(agent, '_observed_previous_action', None) is None):
        raise ValueError('A result requires a real previous action and its observed successor in LOG')
    if not isinstance(text, str) or not text.strip() or len(text) > 1200:
        raise ValueError('text must contain 1-1200 characters')
    if key not in data['entries'] and len(data['entries']) >= 32:
        raise ValueError('32-entry limit; explicitly FORGET_STATE an obsolete entry first')
    if pin and sum(bool(e.get('pin')) for k,e in data['entries'].items() if k != key) >= 4:
        raise ValueError('At most 4 pinned entries; explicitly unpin or forget an old entry')
    old = data['entries'].get(key)
    revision = old.get('revision', 0)+1 if old else 1
    prior = ([{k:v for k,v in old.items() if k not in ('grid','superseded')}]+old.get('superseded', []))[:2] if old else []
    entry = {'key': key, 'kind': kind, 'text': text.strip(), 'observed_step': agent.step_count,
             'frame_id': agent.current_frame_id, 'revision': revision, 'pin': pin,
             'superseded': prior, 'grid': current_snapshot(agent, data) if capture_grid else None}
    if kind == 'result':
        entry['action_step'] = agent.step_count-1
        entry['action'] = agent._observed_previous_action
    data['entries'][key] = entry
    persist(agent, data)
    return {'saved': key, 'revision': revision, 'kind': kind, 'observed_step': agent.step_count,
            'replaces_revision': old['revision'] if old else None, 'grid_captured': entry['grid'] is not None,
            'warning': 'Model-authored claim; the snapshot is evidence, not confirmation of the text.'}


def recall(agent, key=None, compare=False, show_image=False):
    data = load(agent)
    if key is None:
        return {'entries': [{'key': k, 'kind': e['kind'], 'revision': e['revision'],
                             'observed_step': e['observed_step'], 'pin': e['pin'], 'has_grid': e['grid'] is not None}
                            for k,e in sorted(data['entries'].items())]}
    entry = data['entries'].get(key_name(key))
    if entry is None:
        return {'found': False, 'key': key, 'known_keys': sorted(data['entries'])}
    answer = {k:v for k,v in entry.items() if k != 'grid'}
    answer['age_steps'] = agent.step_count-entry['observed_step']
    if entry.get('grid'):
        answer['saved_grid'] = grid_report(entry['grid'])
    if compare:
        answer['comparison'] = compare_snapshots(entry.get('grid'), current_snapshot(agent, data))
    if show_image and entry.get('frame_id'):
        # Invoked only by RECALL_STATE's handler. The host owns all pixels.
        agent.visual_request = {'ops': [
            {'op': 'frame', 'id': entry['frame_id'], 'as': 'saved'},
            {'op': 'resize', 'src': 'saved', 'max_side': 1024, 'as': 'out'}], 'output': 'out'}
    return answer


def forget(agent, key):
    data = load(agent)
    key_name(key)
    removed = data['entries'].pop(key, None)
    persist(agent, data)
    return {'removed': removed is not None, 'key': key}


def brief(agent):
    """Bounded default index + ONLY explicitly pinned model-written text."""
    try:
        data = load(agent)
        lines = ['STATE MEMORY (claims may be stale; current observation takes precedence):']
        pinned = 0
        for key,e in sorted(data['entries'].items())[:32]:
            lines.append('{}: {} rev {} observed {} age {}{}'.format(key, e['kind'], e['revision'],
                e['observed_step'], agent.step_count-e['observed_step'], ' pinned' if e['pin'] else ''))
            if e['pin'] and pinned < 4:
                lines.append(e['text'][:600])
                pinned += 1
        if not data['entries']:
            lines.append('(empty; use SAVE_STATE to keep a placement intention or working observation)')
        lines.append('Grid configured: {}. Current image size: {}.'.format(data['grid_config'] is not None,
                     list(agent.current_image.size) if getattr(agent,'current_image',None) is not None else None))
        return '\n'.join(lines)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return 'State memory needs repair with legacy WRITE: '+str(exc)
