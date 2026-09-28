import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide5 as S
R = "/Users/ajith/sih/output/slides/slide5/assets"
E.export(lambda p: S.hub(p), (K.W, K.H), f"{R}/s5_applications_hub")
E.export(lambda p: S.before_after(p), (K.W, K.H), f"{R}/s5_before_after")
E.export(lambda p: S.awareness(p), (K.W, K.H), f"{R}/s5_live_awareness")
E.export(lambda p: S.benefits(p), (K.W, K.H), f"{R}/s5_benefits")
E.export(lambda p: S.result(p), (K.W, K.H), f"{R}/s5_real_result")
