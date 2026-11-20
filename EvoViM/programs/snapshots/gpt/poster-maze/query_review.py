"""Model-invoked query workspaces, shared between LOG and PLAN.

This is a cache and an inspection/assessment ledger for an explicit retrieval,
not an automatic navigation archive or an identity classifier. All visual
requests originate in ReviewMemory.recall, a registered memory tool handler.
"""
import hashlib
import json
from pathlib import Path

from pattern_registration import content_regions, pixel_box
from scene_memory import SceneMemory


def source_digest(agent):
    path = Path(agent._memory_dir) / 'scene_records.json'
    records = path.read_text() if path.exists() else ''
    index = json.dumps(agent.visual.data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((records + index).encode()).hexdigest()


def review_status(agent):
    """Bounded derived context from a previously requested review, without writes."""
    path = Path(agent._memory_dir) / 'query_review.json'
    if not path.exists() or not agent.public_observation.get('question'):
        return None
    data = json.loads(path.read_text())
    if data.get('frame_id') != agent.current_frame_id or data.get('step') != agent.step_count:
        return None
    if data.get('source_digest') != source_digest(agent):
        return {'source_changed': True, 'instruction': 'Saved evidence changed. Retrieve again before using old comparisons or assessments.'}
    return summary(data)


def summary(data):
    comparison = data['comparison']
    ranked = comparison.get('matches', [])
    return {
        'query_step': data['step'], 'query_region': data['query_region'],
        'gallery_requests': data['gallery_requests'],
        'full_frame_requests': data['requested'],
        'model_assessments': list(data['assessments'].values())[-8:],
        'best_bounded_similarity': ranked[0]['similarity'] if ranked else None,
        'bounded_top_two_gap': (ranked[0]['similarity'] - ranked[1]['similarity']) if len(ranked) > 1 else None,
        'top_candidates': [compact(row) for row in ranked[:5]],
        'interpretation': 'Counts are requests, not proof of delivery or identity. Assessments are your query-specific claims. Rankings and gaps are not probabilities, recognition cutoffs, or answers. A nearest candidate exists even for an unseen pattern. Repeating an unchanged view adds no evidence.'
    }


def compact(view):
    return {key: view[key] for key in ('image_id', 'step', 'region', 'similarity',
            'rotation', 'scale_x', 'scale_y', 'overlap', 'frame_clipped', 'location') if key in view}


def flattened(comparison):
    """Include grouped alternate views so a requested image keeps its own score."""
    rows = []
    for top in comparison.get('matches', []) + comparison.get('partial_matches', []):
        rows.append(top)
        for other in top.get('other_views_at_wall', []):
            rows.append(other)
        # other_sightings omit their own region. Do not copy a representative's
        # pixel box or transform to another frame merely because descriptors
        # were deduplicated. Its original full frame remains retrievable.
    return rows


class ReviewMemory(SceneMemory):
    def __init__(self, agent):
        super().__init__(agent)
        self.review_path = Path(agent._memory_dir) / 'query_review.json'

    def _persist_review(self, data):
        temporary = self.review_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(self.review_path)

    def _workspace(self, region):
        agent = self.agent
        if agent.current_image is None or not agent.current_frame_id:
            raise ValueError('No public query frame')
        query_region = region if region is not None else content_regions(agent.current_image)[0]
        pixel_box(agent.current_image, query_region)
        digest = source_digest(agent)
        old = json.loads(self.review_path.read_text()) if self.review_path.exists() else None
        same_query = bool(old and old.get('frame_id') == agent.current_frame_id and
                          old.get('step') == agent.step_count and old.get('query_region') == query_region)
        if same_query and old.get('source_digest') == digest:
            return old, True
        # Registration is unchanged. Request all eight permitted ranked wall
        # groups once so a later broader review does not repeat the search.
        comparison = super().recall(mode='visual', region=query_region, limit=8)
        # The parent's pending three-column image is replaced by this handler's
        # query gallery. Its old legend must not become a false delivery claim
        # in the persistent comparison cache.
        comparison.pop('attached_order', None)
        comparison['interpretation'] = (
            'Local registration scores are not probabilities or identity decisions. '
            'Bounded crops lie inside a frame, not proof of a complete panel; partial crops '
            'touch an edge and may correlate with unrelated repeated blocks. Equal pixels '
            'and repeated sightings are not independent corroboration. Wall candidates '
            'are separate conditional spatial inferences or model claims; unknown stays unknown. '
            'This cached comparison records computation, not image delivery. Each tool reply '
            'has its own requested attached_order in row-major order, including the query.')
        data = {'version': 1, 'frame_id': agent.current_frame_id, 'step': agent.step_count,
                'query_region': query_region, 'source_digest': digest, 'comparison': comparison,
                'requested': {}, 'gallery_requests': 0, 'assessments': {}, 'revisions': []}
        if same_query:
            data['requested'] = old['requested']
            data['gallery_requests'] = old['gallery_requests']
            data['revisions'] = old.get('revisions', []) + [{
                'reason': 'saved evidence changed; previous assessments invalidated',
                'source_digest': old['source_digest'], 'assessments': old['assessments']}]
        return data, False

    def _gallery(self, data, selected=None, limit=5):
        """Query and alternatives in one host-rendered image, labels in text."""
        agent = self.agent
        ops = [{'op': 'frame', 'id': agent.current_frame_id, 'as': 'q'},
               {'op': 'crop', 'src': 'q', 'box': pixel_box(agent.current_image, data['query_region']), 'as': 'qc'},
               {'op': 'resize', 'src': 'qc', 'max_side': 320, 'as': 'qt'}]
        aliases, legend = ['qt'], [{'cell': 1, 'kind': 'current query', 'step': agent.step_count}]
        candidates = data['comparison'].get('matches', [])[:limit]
        if len(candidates) < limit:
            candidates += data['comparison'].get('partial_matches', [])[:limit-len(candidates)]
        if selected is None:
            views = candidates
        else:
            # Full scene plus its best stored crop and the best alternative.
            # The original frame is preserved in full, not cropped away.
            entry = agent.visual._entry(selected)
            ops.extend([{'op': 'frame', 'id': entry['frame_id'], 'as': 'selected'},
                        {'op': 'resize', 'src': 'selected', 'max_side': 640, 'as': 'selected_full'}])
            aliases.append('selected_full')
            legend.append({'cell': 2, 'kind': 'selected full frame', 'image_id': selected, 'step': entry['step']})
            rows = [r for r in flattened(data['comparison']) if r['image_id'] == selected]
            chosen = max(rows, key=lambda r: r['similarity']) if rows else None
            views = ([chosen] if chosen else []) + [r for r in candidates if r['image_id'] != selected][:1]
        for i, view in enumerate(views):
            # Alternate grouped sightings may inherit a different frame's box;
            # recompute it from that image's own normalized region and size.
            entry = agent.visual._entry(view['image_id'])
            from PIL import Image
            with Image.open(Path(agent._memory_dir) / entry['path']) as frame:
                box = pixel_box(frame, view['region'])
            root, crop, thumb = 'r%s' % i, 'c%s' % i, 't%s' % i
            ops.extend([{'op': 'frame', 'id': entry['frame_id'], 'as': root},
                        {'op': 'crop', 'src': root, 'box': box, 'as': crop},
                        {'op': 'resize', 'src': crop, 'max_side': 320, 'as': thumb}])
            aliases.append(thumb)
            legend.append({'cell': len(aliases), 'kind': 'historical crop', **compact(view)})
        # At most six default cells (3 columns) or four selected cells
        # (2 columns). Max sides remain <=1280 even with a 640px full frame.
        ops.append({'op': 'tile', 'srcs': aliases, 'cols': min(len(aliases), 2 if selected else 3), 'as': 'out'})
        agent.visual_request = {'ops': ops, 'output': 'out'}
        return legend

    def recall(self, image_id=None, mode='auto', region=None, limit=5, assessments=None):
        if not self.agent.public_observation.get('question') or mode == 'places':
            if assessments:
                raise ValueError('Assessments belong to the current question only')
            return super().recall(image_id=image_id, mode=mode, region=region, limit=limit)
        if image_id is not None:
            self.agent.visual._entry(image_id)
        for assessment in assessments or []:
            self.agent.visual._entry(assessment['image_id'])
        data, cached = self._workspace(region)
        # Only explicit model parameters create/replace identity assessments.
        for assessment in assessments or []:
            key = assessment['image_id']
            self.agent.visual._entry(key)
            if key in data['assessments']:
                data['revisions'].append({'image_id': key, 'old_assessment': data['assessments'][key]})
                data['assessments'].pop(key)
            data['assessments'][key] = dict(assessment)
        if image_id is None:
            data['gallery_requests'] += 1
        else:
            data['requested'][image_id] = data['requested'].get(image_id, 0) + 1
        # Keep the original comparison return contract, including provenance.
        result = dict(data['comparison'])
        result['query_region'] = data['query_region']
        result['matches'] = result.get('matches', [])[:limit]
        result['partial_matches'] = result.get('partial_matches', [])[:min(limit, 3)]
        if image_id is not None:
            original = super().recall(image_id=image_id)
            result['selected_frame'] = original
            # Carry the comparison forward without reprinting every grouped
            # sighting on every full-frame inspection. Spatial support stays
            # attached to each candidate; the complete search is cached.
            result['matches'] = [compact(row) for row in result['matches']]
            result['partial_matches'] = [compact(row) for row in result['partial_matches']]
            rows = [r for r in flattened(data['comparison']) if r['image_id'] == image_id]
            result['selected_comparisons'] = [compact(r) for r in sorted(rows, key=lambda r: r['similarity'], reverse=True)[:3]]
            result['selected_comparison_scope'] = 'Only retained ranked candidates are listed; an empty list is unranked, not evidence of absence. Use the full frame or a corrected region.'
        result['attached_order'] = self._gallery(data, image_id, min(limit, 5))
        self._persist_review(data)
        result['review'] = summary(data)
        result['comparison_reused'] = cached
        result['next_decision'] = ('The tile includes this query and historical alternatives in row-major order. '
            'Check the exact arrangement, not just shared colors or a best rank. If none is the same pattern, '
            'answer not_seen is a valid decision. Record a mismatch/uncertainty with assessments if helpful. '
            'Re-request an unchanged frame only to test a specific unresolved feature. '
            'Only ACTION commits your answer; retrieval does not.')
        return result
