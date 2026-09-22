# 3CHAKSHU — SIH 2026 Idea Submission

**SIH26158** · Single-Pass Drone Video to Accurate 3D Model Generation System
NTRO · Drone/Robotics · Software

Layout per slide: **4:3 diagram on the left (~60%), bullets on the right (~40%).**
Template is 16:9, so a 4:3 graphic leaves a clean text column.

---

## SLIDE 1 — TITLE

```
Problem Statement ID     SIH26158
Problem Statement Title  Single-Pass Drone Video to Accurate 3D Model Generation System
Theme                    Drone / Robotics
PS Category              Software
Team ID                  <team ID>
Team Name                <registered name>
```

**Tagline:**
> **3CHAKSHU** — accurate, georeferenced 3D models from a single drone pass, in minutes, on an offline field laptop.

---

## SLIDE 2 — PROPOSED SOLUTION

**Diagram:** `43_capture_geometry.svg`

**Right column:**

**The problem is geometry, not speed**
- One pass → all cameras on a straight line
- Overlap is already 93.7% — more frames add nothing
- B/H = 0.064 → 3.7° triangulation angle
- Depth weak, roll barely observable
- Only one side of every structure is seen

**Why existing tools fail**
- Photogrammetry needs crossing passes
- On one strip it is ill-conditioned — it does not converge

**Our solution**
- Offline desktop app: video + GPS → georeferenced 3D model
- Reconstructs terrain, facades, rooftops, roads, vegetation
- Outputs OBJ · PLY · GLB · LAS · GeoTIFF
- For visualization, measurement and analysis

---

## SLIDE 3 — TECHNICAL APPROACH

**Diagram:** `43_pipeline.svg`

**Right column:**

**Method**
- **Select** — keyframes by GPS baseline in metres, blur-rejected
- **Reconstruct** — pose-free model, no prior poses needed
- **Refine** — bundle adjustment, GPS + gravity priors
- **Quantify** — covariance → per-vertex trust tier
- **Deliver** — exports + accuracy report + refly plan

**Stack**

| Layer | Choice |
|---|---|
| Backbone | MapAnything (Apache-2.0) |
| Alternatives | VGGT · π³ · Depth Anything 3 |
| Optimisation | Sparse LM (SciPy) |
| Fusion | Confidence-weighted TSDF |
| Interface | PySide6 + VTK — native, offline |
| Geospatial | pyproj · laspy · GDAL |

**Progressive reconstruction**
- Geometry appears **while processing continues**
- Coarse model on screen before the run finishes
- Operator can abort a bad capture in seconds
- Enables refly **while still on site**
- Directly answers PS challenge 6 — near-real-time

**Design decisions**
- Swappable backbone — no model lock-in
- No training — pretrained inference only
- Windowed — memory bounded by window, not video length

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

**Diagram:** `17_gaps_roadmap.svg` *(or `13_ps_requirements.svg`)*

**Right column:**

**Published evidence**

| Finding | Source |
|---|---|
| +50% completeness over COLMAP, sparse aerial | arXiv:2507.14798 |
| UAV adaptation: Ray Error −84.2%, ATE −76.0% | arXiv:2605.17942 |
| 2,000 images in ~5 min, F@5 = 0.877 | arXiv:2608.28288 |

**Challenges → strategy**

| Challenge | Strategy |
|---|---|
| Limited viewing angles | Pose-free initialisation |
| Motion blur | Sharpness-scored selection |
| Dynamic objects | Segmentation masking |
| GPS noise | GPS as soft prior in BA |
| Occluded surfaces | Tagged INFERRED, refly planned |
| Metric without GCP | GPS + gravity priors in one solve |

**Feasible because**
- No training, no dataset collection, no cluster
- Single offline installer, GPU optional
- Public UAV benchmarks exist for validation

---

## SLIDE 5 — IMPACT AND BENEFITS

**Diagram:** `43_impact.svg`

**Right column:**

**Applications**
- Border mapping · military reconnaissance
- Disaster damage assessment
- Bridge, tower, dam, corridor inspection
- Urban planning · digital twins
- Construction progress · earthwork volume
- Archaeological documentation

**Benefits**

| Dimension | Benefit |
|---|---|
| Operational | One pass replaces a grid — fewer sorties |
| Decision safety | Never act on unobserved geometry |
| Economic | Open stack, existing hardware |
| Security | Air-gapped; footage never transmitted |
| Strategic | Indigenous, no foreign cloud |

**Accuracy, stated precisely**

| | GPS | RTK |
|---|---|---|
| Relative | < 1 m | < 1 m |
| Absolute | 2–5 m | 1–3 cm |

GPS bias is a sensor limit, not an algorithm limit.

### Limitations we acknowledge — and how we address each

| Limitation | Root cause | Our response |
|---|---|---|
| Far facades never imaged | One pass sees one side | Tag INFERRED · exclude from measurement · plan corrective pass |
| Absolute accuracy capped at 2–5 m | GPS receiver bias, not algorithmic | Report relative and absolute apart · support RTK/PPK · optional 1–2 GCPs |
| Depth precision degrades with altitude | Error scales as z²/(baseline × focal) | Baseline set to 20–30% of altitude · higher working resolution |
| Dense vegetation reconstructs poorly | Canopy has no stable surface | Semantic classification · canopy marked low-confidence, not measured |
| Dynamic objects corrupt geometry | Violates the static-scene assumption | Segmentation masking before fusion |
| Long flights strain field hardware | Memory and time grow with coverage | Windowed processing · quality profiles · progressive output |
| Illumination shifts across a flight | Sun angle and exposure drift | Exposure scoring · colour harmonisation across keyframes |
| Model quality varies by capture pattern | Nadir, oblique and orbit differ geometrically | Flight profile auto-detected · alignment strategy chosen to match |

> Every one of these is a documented failure mode of the domain, not a surprise. Naming them and pairing each with a mitigation is how a system earns trust in an operational setting.

---

## SLIDE 6 — RESEARCH AND REFERENCES

**Diagram:** `43_novelty.svg` *(or `07_research_gap.svg`)*

**Right column:**

**Research gap**
> Photogrammetry requires 70–80% overlap across multiple passes and fails on single strips. Pose-free feed-forward models reconstruct from unposed frames but produce scale-free, non-georeferenced output. **No system combines the two with per-vertex confidence reporting.**

**References**

| Work | ID |
|---|---|
| MapAnything: Feed-Forward Metric 3D Reconstruction | arXiv:2509.13414 |
| π³: Permutation-Equivariant Visual Geometry | arXiv:2507.13347 |
| VGGT: Visual Geometry Grounded Transformer | CVPR 2025 |
| UAVFF3D: Geometry-Aware UAV Benchmark | arXiv:2605.17942 |
| GeoFF3D: Coordinate-Anchored UAV Mapping | arXiv:2608.28288 |
| DUSt3R/MASt3R/VGGT on Aerial Blocks | arXiv:2507.14798 |
| UAV3DCrop: Multi-Angle UAV Benchmark | arXiv:2608.06404 |
| Uncertainty Quantification for UAV Photogrammetry | ISPRS J. |
| Covariance Propagation & Next Best View | ECCV 2012 |
| RTK/PPK UAV Accuracy Assessment | Geocarto 2023 |
| UseGeo: UAV dataset with LiDAR ground truth | ISPRS Open J. 2024 |

**Ours vs published**
- Published: reconstruction, BA, fusion — all cited
- Ours: trust layer + corrective flight planning

---

# 4:3 diagram set

| File | Slide | Visual content |
|---|---|---|
| `43_capture_geometry.svg` | 2 | Drone, frustums, seen/unseen walls, 3 big numbers |
| `43_pipeline.svg` | 3 | 5 stages, colour-coded, trust tiers shown |
| `43_progressive.svg` | 3 (alt) | Live build-up: 3 viewport states, timeline, why it matters |
| `17_gaps_roadmap.svg` | 4 | Gap → why hard → mechanism |
| `43_impact.svg` | 5 | Today vs with · 4 sectors · 4 benefit numbers |
| `43_novelty.svg` | 6 | Mesh comparison · mechanism chain · refly · accuracy table |

**Spares:** `13_ps_requirements` · `07_research_gap` · `11_competitive_matrix` · `09_scalability_submaps` · `10_ui_workflow` · `15_node_graph_ui`

**Insert:** Insert → Pictures → From File. SVG stays vector through PDF export.

---

# Talk track

**The three sentences that matter**

1. *"A single pass is a degenerate geometry, not just fewer photographs — depth and tilt are weakly observable, so photogrammetry doesn't run slowly here, it fails to converge."*
2. *"Feed-forward reconstruction for robustness where SfM can't converge, then bundle adjustment with GPS and gravity priors for the accuracy the model can't give."*
3. *"The same solve that gives accuracy gives us per-point uncertainty — so the model reports which parts of itself were measured."*

| Do | Don't |
|---|---|
| Lead with the geometry insight | Quote an accuracy figure in metres |
| Name what fails and why | Claim published work as yours |
| Cite papers — judges check | Propose cloud (org is air-gapped) |
