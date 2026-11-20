"""Explicit, episode-local region memory and numerical appearance retrieval.

Only handlers invoked by the model store/retrieve entries. Pixels stay on disk;
historical pixels are delivered exclusively by host frame/crop/tile programs.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares


def pixel_box(box, size):
    """Convert model-selected normalized coordinates, validating before I/O."""
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('box must be four finite fractions [left,top,right,bottom]')
    if np.any(values < 0) or np.any(values > 1) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError('box must be ordered within [0,1]')
    w, h = size
    result = [int(math.floor(values[0]*w)), int(math.floor(values[1]*h)),
              int(math.ceil(values[2]*w)), int(math.ceil(values[3]*h))]
    if result[2]-result[0] < 4 or result[3]-result[1] < 4:
        raise ValueError('region too small to compare: choose a larger visible region')
    return result


def foreground_box(image):
    """Optional query computation: trim a uniform surrounding matte, no colors assumed."""
    a = np.asarray(image.convert('RGB'), dtype=float)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    background = np.median(border, axis=0)
    # Robust noise estimate with a single 8-bit quantization-unit floor.
    tolerance = np.maximum(1., 4.*np.median(np.abs(border-background), axis=0))
    mask = np.any(np.abs(a-background) > tolerance, axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return [0, 0, image.width, image.height]
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    if min(box[2]-box[0], box[3]-box[1]) < 4:
        return [0, 0, image.width, image.height]
    return box


def normalized(a):
    """Per-channel centering/scaling removes affine changes of illumination."""
    b = a.reshape(-1, 3).astype(float)
    b -= b.mean(axis=0)
    norm = np.sqrt(np.sum(b*b, axis=0))
    return b / np.maximum(norm, 1e-8)


def appearance_match(source, query):
    """Fit scale, translation and rotation; return evidence, never an action.

    Numerical resolutions and search ranges are generic algorithm parameters,
    chosen before evidence fitting. No episode-specific coordinates or palette.
    """
    side = 64
    a = np.asarray(source.convert('RGB').resize((side, side), Image.Resampling.BILINEAR), dtype=float)
    b = np.asarray(query.convert('RGB').resize((side, side), Image.Resampling.BILINEAR), dtype=float)
    a = gaussian_filter(a, (1., 1., 0.))
    b = gaussian_filter(b, (1., 1., 0.))
    # Inner query support tolerates rotated matte corners and uncertain boxes.
    axis = np.linspace(-.32, .32, 28)
    v, u = np.meshgrid(axis, axis, indexing='ij')
    q = np.stack([map_coordinates(b[:, :, c], [(v+.5)*(side-1), (u+.5)*(side-1)], order=1)
                  for c in range(3)], axis=-1)
    target = normalized(q)
    if np.linalg.norm(target) < .5:
        return {'similarity': 0., 'fit': None, 'warning': 'query has little spatial variation'}

    def sample(p):
        sx, sy, angle, dx, dy = p
        cosine, sine = np.cos(angle), np.sin(angle)
        x = .5 + dx + sx*cosine*u - sy*sine*v
        y = .5 + dy + sx*sine*u + sy*cosine*v
        samples = np.stack([map_coordinates(a[:, :, c], [y*(side-1), x*(side-1)],
                                            order=1, mode='nearest') for c in range(3)], axis=-1)
        return normalized(samples), x, y

    def residual(p):
        samples, x, y = sample(p)
        # Discourage degenerate fits that explain a query with clamped borders.
        outside = np.maximum(0., -x) + np.maximum(0., x-1.) + np.maximum(0., -y) + np.maximum(0., y-1.)
        return np.concatenate(((samples-target).ravel(), 2.*outside.ravel()))

    seeds = []
    for scale in (.7, 1., 1.3):
        for angle in np.deg2rad([-30., -15., 0., 15., 30.]):
            for dx in (-.15, 0., .15):
                for dy in (-.15, 0., .15):
                    p = np.array([scale, scale, angle, dx, dy])
                    r = residual(p)
                    seeds.append((float(r@r), p))
    bounds = ([.35, .35, -math.pi/3, -.4, -.4], [1.8, 1.8, math.pi/3, .4, .4])
    solutions = []
    for _, p in sorted(seeds, key=lambda item:item[0])[:3]:
        fit = least_squares(residual, p, bounds=bounds, max_nfev=45, diff_step=.002)
        samples, x, y = sample(fit.x)
        score = float(np.sum(samples*target)/3.)
        coverage = float(np.mean((x>=0)&(x<=1)&(y>=0)&(y<=1)))
        solutions.append((score*coverage, score, coverage, fit.x))
    _, score, coverage, p = max(solutions, key=lambda item:item[0])
    return {'similarity': round(score*coverage, 5), 'coverage': round(coverage, 5),
            'fit': {'scale_x':round(float(p[0]), 4), 'scale_y':round(float(p[1]), 4),
                    'rotation_degrees':round(float(np.rad2deg(p[2])), 2),
                    'offset_x':round(float(p[3]), 4), 'offset_y':round(float(p[4]), 4)}}


class PatchMemory:
    def __init__(self, root):
        self.root = Path(root)
        self.pixels = self.root/'visual'/'patches'
        self.pixels.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'patch_index.json'
        if not self.path.exists():
            self.write({'schema':1, 'next_id':1, 'entries':{}})

    def read(self):
        # Read on each call, so checkpoint restoration cannot leave a stale cache.
        return json.loads(self.path.read_text(encoding='utf-8'))

    def write(self, data):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path)

    def save(self, image, frame_id, step, regions):
        if image is None or frame_id is None:
            raise ValueError('No current host frame available')
        if not regions:
            return {'saved':[], 'hint':'Choose regions to store; no entry was created'}
        prepared = []
        for region in regions:
            key = region['key'].strip()
            if not key:
                raise ValueError('Every region needs a nonempty observed place/object key')
            prepared.append((region, key, pixel_box(region['box'], image.size)))
        data = self.read()
        saved = []
        for region, key, box in prepared:
            duplicate = next((e for e in data['entries'].values() if e['active'] and
                              e['frame_id']==frame_id and e['key']==key and e['box']==box and
                              e['note']==region.get('note','')), None)
            if duplicate:
                saved.append(duplicate)
                continue
            patch_id = 'patch_%04d' % data['next_id']
            data['next_id'] += 1
            image.crop(box).save(self.pixels/(patch_id+'.png'))
            entry = {'id':patch_id, 'key':key, 'note':region.get('note',''), 'active':True,
                     'confidence':region.get('confidence','observed'), 'step':int(step),
                     'frame_id':frame_id, 'box':box, 'image_size':list(image.size), 'revisions':[]}
            data['entries'][patch_id] = entry
            saved.append(entry)
        self.write(data)
        return {'saved':saved}

    def edit(self, patch_id, key=None, note=None, active=None, confidence=None):
        data = self.read()
        if patch_id not in data['entries']:
            return {'error':'unknown patch_id', 'known_ids':list(data['entries'])}
        entry = data['entries'][patch_id]
        changes = {k:v for k,v in {'key':key,'note':note,'active':active,'confidence':confidence}.items()
                   if v is not None and entry[k]!=v}
        if key is not None and not key.strip():
            raise ValueError('key cannot be empty')
        if changes:
            entry['revisions'].append({k:entry[k] for k in changes})
            entry.update(changes)
            self.write(data)
        return entry

    def match(self, image, box=None, limit=5, key_contains=''):
        if image is None:
            raise ValueError('No current image')
        bounds = pixel_box(box, image.size) if box is not None else foreground_box(image)
        query = image.crop(bounds)
        entries = [e for e in self.read()['entries'].values() if e['active'] and
                   key_contains.casefold() in (e['key']+' '+e['note']).casefold()]
        results = []
        for entry in entries:
            with Image.open(self.pixels/(entry['id']+'.png')) as patch:
                score = appearance_match(patch, query)
            results.append({**{k:entry[k] for k in ('id','key','note','confidence','step','frame_id')}, **score})
        results.sort(key=lambda e:e['similarity'], reverse=True)
        # The model's exact key defines identity/location grouping. Multiple
        # saved viewpoints of that same key must not create a false rival.
        grouped = {}
        for result in results:
            key = result['key']
            if key not in grouped:
                grouped[key] = {**result, 'supporting_ids':[]}
            grouped[key]['supporting_ids'].append(result['id'])
        ranked = list(grouped.values())
        gap = ranked[0]['similarity']-ranked[1]['similarity'] if len(ranked)>1 else None
        return {'query_box_pixels':bounds, 'compared':len(entries), 'distinct_keys':len(ranked),
                'matches':ranked[:limit],
                'top_gap':None if gap is None else round(gap,5),
                'interpretation':'Similarity is spatial correlation after alignment, not probability. '
                'There is always a top rank, even for unseen objects. Check absolute agreement, gap, '
                'coverage, source location and memory completeness. Weak/close scores are inconclusive. '
                'Only the LLM decides identity, absence, correction and action.'}

    def catalog(self):
        entries = list(self.read()['entries'].values())
        active = [e for e in entries if e['active']]
        rows = [f"{e['id']} | {e['key']} | step {e['step']} | {e['confidence']}" for e in active[-32:]]
        return f"REGION MEMORY: {len(active)} active / {len(entries)} total\n" + ('\n'.join(rows) or '(empty)')

    def recall_program(self, patch_ids, include_context=False):
        data = self.read()
        if not patch_ids:
            return None, {'hint':'Choose patch_ids from MATCH_PATCHES or region catalog; no image requested'}
        selected = []
        for patch_id in patch_ids:
            if patch_id not in data['entries']:
                raise ValueError('unknown patch_id: '+patch_id)
            selected.append(data['entries'][patch_id])
        ops, aliases = [], []
        for i, entry in enumerate(selected):
            frame, crop, small = f'f{i}', f'c{i}', f's{i}'
            ops.append({'op':'frame','id':entry['frame_id'],'as':frame})
            if include_context:
                ops.append({'op':'resize','src':frame,'max_side':512,'as':small})
            else:
                ops += [{'op':'crop','src':frame,'box':entry['box'],'as':crop},
                        {'op':'resize','src':crop,'max_side':256,'as':small}]
            aliases.append(small)
        ops.append({'op':'tile','srcs':aliases,'cols':min(3,len(aliases)),
                    'labels':[e['id']+' '+e['key'][:24] for e in selected], 'as':'out'})
        return {'ops':ops,'output':'out'}, {'entries':selected,
                'delivery':('historical source context' if include_context else 'historical crops')+' on next model request'}


def install_patches(agent, registry):
    from registry import Tool
    obj = {'type':'object','additionalProperties':False}
    string = {'type':'string','maxLength':160}
    box = {'type':'array','items':{'type':'number','minimum':0,'maximum':1},'maxItems':4}
    region = {**obj,'properties':{'key':string, 'box':box, 'note':{'type':'string','maxLength':1000},
              'confidence':{'type':'string','enum':['observed','uncertain']}},'required':['key','box']}

    def store(regions):
        return PatchMemory(agent._memory_dir).save(agent.current_image, agent.current_frame_id,
                                                  agent.step_count, regions)

    def match(box=None, limit=5, key_contains=''):
        return PatchMemory(agent._memory_dir).match(agent.current_image, box, limit, key_contains)

    def recall(patch_ids, include_context=False):
        program, result = PatchMemory(agent._memory_dir).recall_program(patch_ids, include_context)
        if program is not None:
            agent.visual_request = program
        return result

    def edit(patch_id, key=None, note=None, active=None, confidence=None):
        return PatchMemory(agent._memory_dir).edit(patch_id,key,note,active,confidence)

    specs = [
        Tool('STORE_PATCHES', 'Explicitly save selected CURRENT image regions with observed place/object keys. '
             'box=[left,top,right,bottom] as fractions of full image dimensions, excluding frames/occluders. '
             'Batch all distinctive visible objects, with their visible landmark and side in key. '
             'This records current evidence, never a future action outcome. Empty regions stores nothing.',
             {**obj,'properties':{'regions':{'type':'array','items':region,'maxItems':16}},'required':['regions']},
             store, ('LOG','PLAN'),'memory_write','metadata'),
        Tool('MATCH_PATCHES', 'Retrieve numeric spatial appearance matches between CURRENT query and LLM-saved regions. '
             'Optional box uses full-image fractions; omit to trim a uniform background automatically. '
             'Returns ranked distinct keys (best saved view each), correlations, fit and top-gap, no automatic identity/absence/action. '
             'Compare texture arrangement, not color. key_contains optionally filters memory.',
             {**obj,'properties':{'box':box,'limit':{'type':'integer','minimum':1,'maximum':16},
                                 'key_contains':string},'required':[]}, match, ('PLAN',),'memory_read','metadata'),
        Tool('RECALL_PATCHES', 'Request labelled historical crops via host on the next model request. '
             'Use saved patch_ids to inspect disputed matches. include_context=true shows full source frames '
             'to verify landmark/place labels rather than trusting captions. Empty list requests nothing.',
             {**obj,'properties':{'patch_ids':{'type':'array','items':string,'maxItems':9},
                                 'include_context':{'type':'boolean'}},'required':['patch_ids']},
             recall, ('LOG','PLAN'),'memory_read','image'),
        Tool('EDIT_PATCH', 'Correct key/note/confidence or deactivate/reactivate a saved region after inspecting evidence. '
             'Prior values remain in revisions. active=false removes the entry from retrieval without destroying provenance.',
             {**obj,'properties':{'patch_id':string,'key':string,'note':{'type':'string','maxLength':1000},
                                 'active':{'type':'boolean'},'confidence':{'type':'string','enum':['observed','uncertain']}},
              'required':['patch_id']}, edit, ('LOG','PLAN'),'memory_write','metadata'),
    ]
    for spec in specs:
        registry.register(spec)
