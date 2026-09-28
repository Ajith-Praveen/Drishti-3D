import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide4_validation as S
R = "/Users/ajith/sih/output/slides/slide4/assets"
E.export(lambda p: S.draw_table(p), (K.W, K.H), f"{R}/s4_real_flight_validation")
E.export(lambda p: S.draw_failure(p), (K.W, K.H), f"{R}/s4_failure_aware")
