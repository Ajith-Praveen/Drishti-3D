import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide5v2 as S
R = "/Users/ajith/sih/output/slides/slide5/assets"
E.export(lambda p: S.who(p), (K.W, K.H), f"{R}/s5_who_it_serves")
E.export(lambda p: S.today_vs(p), (K.W, K.H), f"{R}/s5_present_gaps")
E.export(lambda p: S.kpis(p), (K.W, K.H), f"{R}/s5_benefits")
E.export(lambda p: S.timeline(p), (K.W, K.H), f"{R}/s5_mission_timeline")
