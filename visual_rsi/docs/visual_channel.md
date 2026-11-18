# Derived visual channel contract

Only the trusted host owns original frames. On line M, a candidate may set
`agent.visual_request` inside a handler invoked by the model. The host renders
the JSON program from its own copies of current or earlier frames. Candidate
code never sends pixel bytes. Lines T and V cannot request historical images.

Allowed operations are `frame`, `crop`, `resize`, `grayscale`, `quantize`, `diff`,
`tile`, `annotate`, `canvas`, and `draw`. A program has at most 32 operations.
Output sides are at most 1536 px. Tiles have at most 16 frames. Historical
image deliveries are counted and reported, with no per-phase or per-episode
quantity cap. Annotation mark text is at most 40 characters. Future or unknown frames, invalid
aliases, out of bounds regions, and requests with no next model call are rejected.

`canvas` starts a blank image without a frame. Its positive grid dimensions and
`cell_px` must keep each output side within 1536 px. `draw` adds primitives
without a separate count cap to a canvas or existing image. Coordinates
are grid cells. For a frame or other image without canvas grid metadata, the
host uses 32 px cells. A primitive is a `cell`, `agent`, `label`, or `path`.
Path length and label text have no separate length cap.
Cell colors must be one of `wall`, `floor`, `visited`, `unknown`, `door`, `key`,
`goal`, `danger`, `agent`, `object`, `water`, or `other`. RGB values are rejected.
The host counts labels containing a legal action name in `canvas_action_labels`;
the count is an annotation and does not reject the image.

For example, an M-line tool can set:

```json
{"ops":[
  {"op":"canvas","grid":[9,9],"cell_px":32,"as":"m"},
  {"op":"draw","src":"m","prims":[
    {"type":"cell","at":[2,3],"color":"wall"},
    {"type":"agent","at":[4,4],"facing":"N"},
    {"type":"label","at":[5,4],"text":"red key"},
    {"type":"path","pts":[[4,4],[5,4]]}
  ],"as":"out"}
],"output":"out"}
```
