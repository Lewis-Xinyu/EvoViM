"""Original visual tools with bundled public provenance and numeric retrieval."""
from registry import Tool


def install_visual(agent, registry, scratch=None, event=None):
    from agents.visual_memory import VisualMemoryStore
    from query_review import ReviewMemory
    agent.visual = VisualMemoryStore(scratch or agent._memory_dir, event)
    agent.visual_request = None
    scenes = ReviewMemory(agent)
    obj = {'type': 'object', 'additionalProperties': False}
    string = {'type': 'string'}
    region = {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'maxItems': 4}
    wall = {**obj, 'properties': {'row': {'type': 'integer', 'minimum': 0},
                                  'col': {'type': 'integer', 'minimum': 0},
                                  'side': {'type': 'string', 'enum': list('NESW')}},
            'required': ['row', 'col', 'side']}
    specs = [
        Tool('SAVE_FRAME', 'Explicitly retain this navigation observation: original image plus current public pose, legal exits and observed previous transition in one record. Omit all arguments for adaptive crop proposals; label/note/region/wall are optional. No need to know a distant wall yet. A wall parameter is your verified claim, not the observer pose. Repeated calls in one frame update it.',
             {**obj, 'properties': {'label': string, 'note': string, 'region': region, 'wall': wall}, 'required': []},
             scenes.save, ('LOG', 'PLAN'), 'memory_write', 'metadata'),
        Tool('RECALL_IMAGE', 'With {} at a question: compare saved patterns and open a persistent query workspace shared by LOG/PLAN, with rankings, request counts and simultaneous query/alternative crops. image_id adds its full frame alongside the query and alternatives. Optional assessments records or corrects YOUR same_pattern/different_pattern/uncertain/unreviewed judgments for this query; these are claims, never automatic answers. Unchanged comparisons are cached; evidence edits invalidate them. mode=places retrieves saved exits without images. Scores are not probabilities. Repeated views add no evidence: decide whether an exact match exists, then commit ACTION (not_seen is valid).',
             {**obj, 'properties': {'image_id': string, 'mode': {'type': 'string', 'enum': ['auto', 'places', 'visual']},
                                    'region': region, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 8},
                                    'assessments': {'type': 'array', 'maxItems': 8, 'items': {
                                        **obj, 'properties': {'image_id': string,
                                            'verdict': {'type': 'string', 'enum': ['same_pattern', 'different_pattern', 'uncertain', 'unreviewed']},
                                            'note': {'type': 'string', 'maxLength': 500}},
                                        'required': ['image_id', 'verdict']}}}, 'required': []},
             scenes.recall, ('LOG', 'PLAN'), 'memory', 'metadata+image'),
        Tool('SEARCH_IMAGE', 'Search saved visual labels/notes; original substring search.',
             {**obj, 'properties': {'keyword': string}, 'required': ['keyword']}, agent.visual.search,
             ('LOG', 'PLAN'), 'memory_read', 'metadata'),
        Tool('ANNOTATE_IMAGE', 'Correct a saved label/note, crop region, or model-verified wall. clear_wall=true retracts a wall claim. Corrections replace old values with revision provenance; inspect pixels before claiming a location.',
             {**obj, 'properties': {'image_id': string, 'label': string, 'note': string, 'region': region,
                                    'wall': wall, 'clear_wall': {'type': 'boolean'}}, 'required': ['image_id']},
             scenes.annotate, ('LOG', 'PLAN'), 'memory_write', 'metadata'),
        Tool('DELETE_IMAGE', 'Delete a selected saved image and its observation bundle; its derived place facts disappear from retrieval unless another saved frame supports them.',
             {**obj, 'properties': {'image_id': string}, 'required': ['image_id']}, scenes.delete,
             ('LOG', 'PLAN'), 'memory_write', 'metadata'),
    ]
    for tool in specs:
        registry.register(tool)
