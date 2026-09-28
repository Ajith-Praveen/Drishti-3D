import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide6 as S
R = "/Users/ajith/sih/output/slides/slide6/assets"
E.export(lambda p: S.timeline(p), (K.W, K.H), f"{R}/s6_research_timeline")
E.export(lambda p: S.listing(p, 0.45, 4.74, 4.1, 2.14, S.TEAL, "globe", "Datasets & ground truth", S.DATASETS), (K.W, K.H), f"{R}/s6_datasets")
E.export(lambda p: S.listing(p, 4.65, 4.74, 4.1, 2.14, S.ORANGE, "chart", "Our evidence", S.EVIDENCE, S.ORANGE.darker(135)), (K.W, K.H), f"{R}/s6_evidence")
E.export(lambda p: S.links(p, 8.85, 4.74, 4.05, 2.14), (K.W, K.H), f"{R}/s6_project_links")
