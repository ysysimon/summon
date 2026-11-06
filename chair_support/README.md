# Supported-chair geometry backend

This backend fits labelled chair surfaces to an unchanged motion. ContactFormer
predictions remain unchanged; role assignment is a fitting diagnostic only.
The main repository worker owns model inference and procedural asset generation.

Run the isolated SUMMON Python environment from the repository root:

```powershell
environments\summon\.venv\Scripts\python.exe external\summon\supported_chair.py `
  --geometry <dense-manifest.json> --vertices-path <world-655.npy> `
  --contacts <contact-labels.npy> --frame-indices <source-frame-indices.npy> `
  --candidate-source cad --models-dir <models-directory> --output <new-fit-dir>
```

`--estimate-only` writes body/phase diagnostics and `estimate.json`, without
fitting an asset. For generated chairs, pass `--candidate-source infinigen` and
`--assets-manifest candidates.json`, containing `{"assets":["path/to/asset.json"]}`.
An Infinigen asset is optimized with rigid XY translation and yaw only. CAD
assets also permit uniform scale in [0.9, 1.1]. Neither branch changes the human.

## Inputs and annotations

The dense manifest's `fitting_data` identifies `world`, `joints`, `joint_names`,
and `faces` files, with metre units and Z-up coordinates. `source_fps` is required.
The worker validates input shapes, topology and hashes before launching this
backend. The frozen SMPL-X anatomy annotation additionally checks the exact
10475-vertex template and int64 face-array hash. The same vertex IDs are used
for original, mirrored and transition clips. The annotation's provenance is
recorded in `annotations/body_regions_smplx_v1.json`; it is a one-time inspected draft,
not a learned anatomical classifier.

An asset contains `asset.json`, `mesh.obj`, and `parts.npz`. Its opening points
along +Y, its up axis is Z or +Z, and its units are metres. Parts name actual mesh
faces: `seat_face_indices` and `backrest_face_indices` (aliases `seat_faces` and
`backrest_faces` accepted), with optional `front_edge_face_indices`,
`front_edge_vertices` and `floor_vertices`. `face_part_ids` uses 0 for seat,
1 for back and 2 for remaining geometry. The seat annotation must select mostly
upward surfaces; the back annotation must select mostly forward surfaces.

Eight bundled CAD assets have source-hash-bound height/orientation predicates in
`annotations/cad_chair_annotations_v1.json`. The converted bundle exports actual face
indices and `annotated.ply` for inspection. These predicates are functional
surface annotations, not semantic segmentation of every face in a chair.

Physical instances use optional integer `face_instance_ids` in `parts.npz`,
one per original face (`-1` means unknown). The asset manifest must bind both
the source mesh and parts file by SHA-256. A group ID supplies identity only.
`convex_envelope_ids` lists complete authored instances whose vertex hull is a
trusted outer bound. Alternatively, `instance_bound_ids` and `instance_bounds`
declare a finite local bounding box of shape (K, 2, 3) per listed instance,
including any missing geometry. Box-only declarations permit separation tests.

## Decisions and limits

Seated phases require sustained chair predictions, low pelvis velocity,
approximately horizontal thighs, and bent knees. `phase_evidence.npz` contains
the per-source-frame measurements, and `phase.json` contains accepted intervals.
Predicted floor contacts estimate the floor height. No floor evidence is an
explicit error, not a silent zero-height assumption.

Downward-facing fixed anatomical regions supply a lower envelope on a 2 cm XY
grid. Coverage uses equal projected-area cells; distance quantiles use cell
surface-area estimates. Each region and sampled frame is normalized independently
in the support objective. Contact-role distances also normalize each source
frame independently. The mesh seat is ray-tested at these points; a chair bounding box or infinite plane
cannot satisfy coverage. A curved body surface may touch near its low part:
the method does **not** require every buttock vertex to lie on one plane.

Hard acceptance evaluates every stable source frame: all four support regions
need at least 90% area coverage; their area-weighted 20th-percentile gaps must
be at most 15 mm for buttocks and 20 mm for proximal thighs; seat intrusion must
be at most 5 mm. The backrest must remain behind the pelvis, heading error at
most 20 degrees, and ground error at most 5 mm. These are engineering defaults,
not calibrated perception probabilities. `estimate.seat_profile` is only an
initial sloping-plane proxy for procedural generation; `front_dip` controls the
native Bezier handles, so the generated seat must be measured and validated.

Coarse search and Powell optimization are followed by COBYLA repair against
explicit geometry constraints with an initial 0.5 mm internal margin. If that
extra margin is infeasible, subsequent repair may reduce it to 0.05 mm and then
0.005 mm for support; collision guards retain at least 0.1 mm for the audit's
distance bounds. Final acceptance limits never change, and all stages are logged. Fixed human
signed-distance scenes at guard frames guide the optimizer away from body
penetration. The complete audit supplies worst source frames and exact surface
witnesses for repair. A witness is transformed into the chair's local coordinates
and checked at every guard frame, including existing frames. Worst support
frames also join the active constraints. Each new exact witness triggers one
streamed guard-distance scan over every source frame; the three most dangerous
frames not already active are added, with the complete risk curve exported as
NPZ. This scan guides optimization and does not certify geometry. There are at most eight validation
rounds, with at most six support and six collision repairs inside that total
budget. These guidance samples never establish
final collision acceptance. `optimization.json` records active frames,
constraint violations, optimizer status and every repair round.

Only candidates passing those gates reach collision verification over **every
source frame**. The visible body is unchanged. A centroid fan closes each open
body boundary for the signed-distance collision proxy, with cap counts recorded.
Both complete surfaces are adaptively subdivided against solid distance bounds;
surface samples supply lower bounds, and cell coverage supplies upper bounds.
The audit includes containment without any surface crossing. Closed furniture
components establish reverse penetration lower bounds. Within one authored
instance, unique coincident borders may be stitched. A certified single hull
can contribute to the upper bound when its surface is within 2 mm of the
original triangles at the largest permitted scale. Unresolved components
retain their raw surface checks. A measured original penetration over 5 mm rejects the candidate; a pass
requires the global upper bound to stay within 5 mm. Edge/triangle tests report
all segment crossings for diagnostics. Crossings alone do not reject shallow
contact.
This is geometric validation, not a gravity/friction or deformable-body simulation.

## Outputs

`fit_result.json` records all candidates, metrics, rejection reasons, and a best
candidate only when one qualifies. At most three qualified candidates have
`retained:true`; other physically qualified records remain distinguishable from
failures. Each fitted candidate retains its mesh and parts, support metrics per
source frame, and collision diagnostics when evaluated. Invalid individual
candidate assets are rejected while other candidates continue.

Viewer diagnostics include `body_regions.npz` (fixed vertex indices),
`support_indices.npz` (source frame indices and padded lower-envelope IDs),
`body_forward.npy`, and `pelvis.npy`. Contact roles are in `contact_roles.json`.
Exit status is 0 for success or successful estimate, 2 for no valid candidate or
no stable seated interval, and nonzero with `status:error` for malformed inputs.

Run the acceptance tests with:

```powershell
environments\summon\.venv\Scripts\python.exe -m unittest discover `
  -s external\summon\tests -p test_supported_chair.py -v
```

Tests use real small triangle meshes to cover finite support footprint, curved
contact, narrow/floating seats, reversed/sideways/raised chairs, shallow and deep
leg intersections, edge crossings with outside endpoints, caps, and invalid
part annotations.
