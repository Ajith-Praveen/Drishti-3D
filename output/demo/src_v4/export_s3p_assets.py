import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import slide3_proto as S
E.export(lambda p: S.prototype(p, 10.7, 2.5, 12.9, 2.5 + 0.5 + (2.2 - 0.16) * 9 / 16 + 0.1 + 3 * 0.27 + 0.12), (K.W, K.H),
         "/Users/ajith/sih/output/slides/slide3/assets/working_prototype")
