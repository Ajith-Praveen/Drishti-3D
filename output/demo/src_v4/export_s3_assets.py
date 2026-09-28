import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide3_v4 as S
R = "/Users/ajith/sih/output/slides/slide3/assets"
E.export(lambda p: S.cards(p, 1.56, 1.3), (K.W, K.H), f"{R}/techstack_cards")
E.export(lambda p: S.architecture(p), (K.W, K.H), f"{R}/architecture_readable")
E.export(lambda p: S.prototype(p), (K.W, K.H), f"{R}/working_prototype")
