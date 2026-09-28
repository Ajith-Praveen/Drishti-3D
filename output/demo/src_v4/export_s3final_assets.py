import sys
sys.path.insert(0, "/tmp/demo4")
import export_assets as E
import slidekit as K
import arch, techstack_v as TV
import slide3_final as S
R = "/Users/ajith/sih/output/slides/slide3/assets"
E.export(lambda p: arch.draw(p, title=False), (arch.W, arch.H), f"{R}/architecture", dpi=400)
E.export(lambda p: TV.draw(p), (TV.W, TV.H), f"{R}/techstack_vertical", dpi=560)
E.export(lambda p: S.prototype(p, 0.45, 2.0, S.COL_W), (K.W, K.H), f"{R}/working_prototype")
E.export(lambda p: arch.draw(p, title=False), (arch.W, arch.H), "/Users/ajith/sih/output/slides/DRISHTI-3D_architecture", dpi=400)
