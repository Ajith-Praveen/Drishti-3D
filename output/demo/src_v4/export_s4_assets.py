import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide4 as S
R = "/Users/ajith/sih/output/slides/slide4/assets"
E.export(lambda p: S.screens(p), (K.W, K.H), f"{R}/s4_screenshots_strip")
E.export(lambda p: S.feasible(p), (K.W, K.H), f"{R}/s4_why_ours_is_more_feasible")
E.export(lambda p: S.proof(p), (K.W, K.H), f"{R}/s4_proof_in_numbers")
E.export(lambda p: S.risks(p), (K.W, K.H), f"{R}/s4_risks_and_strategies")
