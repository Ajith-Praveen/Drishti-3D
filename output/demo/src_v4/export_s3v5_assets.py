import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide3_v5 as S
R = "/Users/ajith/sih/output/slides/slide3/assets"
E.export(lambda p: S.cards(p, 1.52, 1.4), (K.W, K.H), f"{R}/techstack")
E.export(lambda p: S.architecture(p), (K.W, K.H), f"{R}/architecture")
E.export(lambda p: S.prototype(p), (K.W, K.H), f"{R}/working_prototype")
