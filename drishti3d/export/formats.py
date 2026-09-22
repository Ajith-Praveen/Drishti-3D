"""Deliverable point-cloud / mesh file formats: PLY, OBJ, glTF-binary (GLB), LAS, XYZ.

The one rule every writer in this module follows: **confidence is an
exported channel, not a debugging artifact that stays inside this
process**. A model that looks clean in the desktop app but silently loses
its trust layer the moment it's handed to a GIS tool or a third-party
viewer would defeat the entire point of DRISHTI-3D, so every format here
that has any reasonable way to carry a custom per-point/per-vertex scalar
does so:

- PLY: a plain ``confidence`` scalar vertex property (every PLY reader
  already knows how to expose "extra" vertex properties).
- LAS: a proper LAS **extra bytes** dimension (the standard LIDAR-world
  mechanism for exactly this kind of custom per-point channel), plus a
  second one for covariance trace when covariance is present.
- glTF/GLB: a custom ``_CONFIDENCE`` vertex attribute -- the underscore
  prefix is glTF's own spec-sanctioned way of adding an application
  -specific attribute without claiming a reserved semantic name.
- OBJ has no mechanism for arbitrary per-vertex scalars at all (it barely
  has one for per-vertex colour, itself a non-standard-but-widely-
  supported extension) -- confidence is simply not exportable there, and
  ``export_obj`` does not pretend otherwise.

``export_ply``/``export_las``/``export_glb`` are all hand-rolled against
their format specs (no ``plyfile``/``pygltflib`` dependency): PLY and GLB
because they're simple enough to not warrant a new dependency, LAS because
``laspy`` is already a project dependency. ``read_ply``/``read_glb`` are
included alongside their writers specifically so round-trips (confidence
included) can be verified without any extra library.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import laspy
import numpy as np

from drishti3d.types import PointCloud

__all__ = [
    "as_geometry",
    "export_fbx",
    "export_glb",
    "export_las",
    "export_obj",
    "export_obj_textured",
    "export_ply",
    "export_xyz",
    "read_glb",
    "read_ply",
]

MeshLike = tuple  # (vertices, faces) or (vertices, faces, colors, confidence)


def as_geometry(
    pc_or_mesh: PointCloud | MeshLike,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Normalize a ``PointCloud`` or ``(vertices, faces[, colors, confidence])`` tuple.

    Returns ``(xyz, faces_or_None, rgb_or_None, confidence_or_None)``.
    """
    if isinstance(pc_or_mesh, PointCloud):
        return pc_or_mesh.xyz, None, pc_or_mesh.rgb, pc_or_mesh.confidence
    if isinstance(pc_or_mesh, tuple):
        if len(pc_or_mesh) == 2:
            vertices, faces = pc_or_mesh
            return np.asarray(vertices), np.asarray(faces), None, None
        if len(pc_or_mesh) == 4:
            vertices, faces, colors, confidence = pc_or_mesh
            return np.asarray(vertices), np.asarray(faces), colors, confidence
        if len(pc_or_mesh) == 6:
            # (vertices, faces, colors, confidence, semantic_class,
            # semantic_confidence) -- the semantic pair is read by
            # `semantic_of`, not here, so this function's 4-tuple return
            # contract stays exactly as every existing caller expects.
            vertices, faces, colors, confidence = pc_or_mesh[:4]
            return np.asarray(vertices), np.asarray(faces), colors, confidence
        raise ValueError(
            "mesh tuple must be (vertices, faces), (vertices, faces, colors, confidence), "
            "or (vertices, faces, colors, confidence, semantic_class, semantic_confidence)"
        )
    raise TypeError(f"unsupported geometry type: {type(pc_or_mesh)!r}")


def _quantize_unit(values: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` float array to ``uint8`` ``[0, 255]``.

    Used for ``semantic_confidence`` in PLY and LAS. A byte is enough: the
    vote ratio is a coarse agreement measure, and storing it as float32
    would add 3 bytes per point to every export to preserve precision the
    quantity does not have. Values outside ``[0, 1]`` are clipped rather
    than wrapped -- silently aliasing 1.2 to 51 would be worse than
    saturating it.
    """
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).__mul__(255.0).round().astype(np.uint8)


def semantic_of(
    pc_or_mesh: PointCloud | MeshLike,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(semantic_class, semantic_confidence)``, or ``(None, None)``.

    Deliberately a separate accessor rather than two more elements on
    ``as_geometry``'s tuple: that function's 4-tuple shape is relied on by
    every writer and by external callers, and widening it would break them
    all to serve a channel that only some formats can carry anyway. Mesh
    tuples have no semantic slot at all, so they always answer
    ``(None, None)`` -- meshes reach the writers through ``PointCloud``
    when labels matter (see ``pipeline.stages.FusionStage``).
    """
    if isinstance(pc_or_mesh, PointCloud):
        return pc_or_mesh.semantic_class, pc_or_mesh.semantic_confidence
    if isinstance(pc_or_mesh, tuple) and len(pc_or_mesh) == 6:
        return pc_or_mesh[4], pc_or_mesh[5]
    return None, None


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

_PLY_TYPE_TO_NUMPY = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def export_ply(path: str | Path, pc_or_mesh: PointCloud | MeshLike, binary: bool = True) -> None:
    """Write a PLY point cloud or triangle mesh, with ``confidence`` as a custom scalar property."""
    xyz, faces, rgb, confidence = as_geometry(pc_or_mesh)
    semantic_class, semantic_conf = semantic_of(pc_or_mesh)
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    has_rgb = rgb is not None
    has_conf = confidence is not None
    has_sem = semantic_class is not None
    has_sem_conf = semantic_conf is not None
    has_faces = faces is not None and len(faces) > 0

    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        "comment DRISHTI-3D export -- confidence is a first-class exported channel",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_rgb:
        header_lines += ["property uchar red", "property uchar green", "property uchar blue"]
    if has_conf:
        header_lines.append("property uchar confidence")
    if has_sem:
        # `semantic_class` holds semantics.classes.SemanticClass values.
        # The comment is written into the file on purpose: a PLY that
        # outlives this repo should still say what its integers mean,
        # rather than leaving a consumer to guess that 2 means "building".
        header_lines.append(
            "comment semantic_class: 0=unlabelled 1=terrain 2=building 3=road 4=vegetation "
            "5=infrastructure 6=water 7=vehicle 8=person 9=sky 10=obstacle"
        )
        header_lines.append("property uchar semantic_class")
    if has_sem_conf:
        header_lines.append("comment semantic_confidence: winning class's share of total vote weight, 0-255")
        header_lines.append("property uchar semantic_confidence")
    if has_faces:
        header_lines.append(f"element face {len(faces)}")
        header_lines.append("property list uchar int vertex_indices")
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(header)
        if binary:
            fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_rgb:
                fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
            if has_conf:
                fields.append(("confidence", "u1"))
            if has_sem:
                fields.append(("semantic_class", "u1"))
            if has_sem_conf:
                fields.append(("semantic_confidence", "u1"))
            vertex_arr = np.zeros(n, dtype=np.dtype(fields))
            vertex_arr["x"] = xyz[:, 0]
            vertex_arr["y"] = xyz[:, 1]
            vertex_arr["z"] = xyz[:, 2]
            if has_rgb:
                rgb_u8 = np.asarray(rgb, dtype=np.uint8)
                vertex_arr["red"] = rgb_u8[:, 0]
                vertex_arr["green"] = rgb_u8[:, 1]
                vertex_arr["blue"] = rgb_u8[:, 2]
            if has_conf:
                vertex_arr["confidence"] = np.asarray(confidence, dtype=np.uint8)
            if has_sem:
                vertex_arr["semantic_class"] = np.asarray(semantic_class, dtype=np.uint8)
            if has_sem_conf:
                vertex_arr["semantic_confidence"] = _quantize_unit(semantic_conf)
            f.write(vertex_arr.tobytes())

            if has_faces:
                faces_arr = np.asarray(faces, dtype=np.int64)
                face_dt = np.dtype([("n", "u1"), ("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4")])
                face_out = np.zeros(len(faces_arr), dtype=face_dt)
                face_out["n"] = 3
                face_out["v0"] = faces_arr[:, 0]
                face_out["v1"] = faces_arr[:, 1]
                face_out["v2"] = faces_arr[:, 2]
                f.write(face_out.tobytes())
        else:
            for i in range(n):
                parts = [f"{xyz[i, 0]:.7g}", f"{xyz[i, 1]:.7g}", f"{xyz[i, 2]:.7g}"]
                if has_rgb:
                    parts += [str(int(v)) for v in rgb[i]]
                if has_conf:
                    parts.append(str(int(confidence[i])))
                if has_sem:
                    parts.append(str(int(semantic_class[i])))
                if has_sem_conf:
                    parts.append(str(round(float(np.clip(semantic_conf[i], 0.0, 1.0)) * 255.0)))
                f.write((" ".join(parts) + "\n").encode("ascii"))
            if has_faces:
                for face in faces:
                    f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n".encode("ascii"))


def read_ply(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Read back a PLY written by ``export_ply``. Returns ``(xyz, faces, rgb, confidence)``.

    Supports exactly the subset of the PLY spec ``export_ply`` produces
    (a ``vertex`` element with scalar properties, an optional ``face``
    element with one ``list`` property) -- not a general-purpose PLY
    reader, but enough to verify round-trips without a new dependency.
    """
    path = Path(path)
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF while reading PLY header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        rest = f.read()

    fmt_line = next(line_ for line_ in header_lines if line_.startswith("format"))
    is_binary = "binary" in fmt_line

    elements: list[dict] = []
    for line in header_lines:
        if line.startswith("element"):
            _, name, count = line.split()
            elements.append({"name": name, "count": int(count), "properties": []})
        elif line.startswith("property"):
            parts = line.split()
            if parts[1] == "list":
                elements[-1]["properties"].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1]["properties"].append(("scalar", parts[1], parts[2]))

    result: dict[str, dict] = {}
    offset = 0
    ascii_lines = rest.decode("ascii").split("\n") if not is_binary else []
    line_ptr = 0

    for el in elements:
        props = el["properties"]
        count = el["count"]
        is_list = len(props) == 1 and props[0][0] == "list"

        if not is_list:
            names = [p[2] for p in props]
            types = [p[1] for p in props]
            if is_binary:
                dt = np.dtype([(name, _PLY_TYPE_TO_NUMPY[t]) for name, t in zip(names, types, strict=True)])
                arr = np.frombuffer(rest, dtype=dt, count=count, offset=offset)
                offset += dt.itemsize * count
                result[el["name"]] = {name: arr[name] for name in names}
            else:
                cols = {name: np.zeros(count, dtype=np.float64) for name in names}
                for i in range(count):
                    vals = ascii_lines[line_ptr].split()
                    line_ptr += 1
                    for name, v in zip(names, vals, strict=True):
                        cols[name][i] = float(v)
                result[el["name"]] = cols
        else:
            _, count_t, val_t, _name = props[0]
            faces = np.zeros((count, 3), dtype=np.int64)
            if is_binary:
                count_dt = np.dtype(_PLY_TYPE_TO_NUMPY[count_t])
                val_dt = np.dtype(_PLY_TYPE_TO_NUMPY[val_t])
                for i in range(count):
                    n_verts = int(np.frombuffer(rest, dtype=count_dt, count=1, offset=offset)[0])
                    offset += count_dt.itemsize
                    verts = np.frombuffer(rest, dtype=val_dt, count=n_verts, offset=offset)
                    offset += val_dt.itemsize * n_verts
                    faces[i] = verts[:3]
            else:
                for i in range(count):
                    vals = ascii_lines[line_ptr].split()
                    line_ptr += 1
                    faces[i] = [int(x) for x in vals[1:4]]
            result[el["name"]] = {"faces": faces}

    vertex = result.get("vertex", {})
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64) if vertex else np.zeros((0, 3))
    rgb = None
    if "red" in vertex:
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    confidence = None
    if "confidence" in vertex:
        confidence = vertex["confidence"].astype(np.uint8)
    faces_out = result["face"]["faces"] if "face" in result else None

    return xyz, faces_out, rgb, confidence


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def export_obj_textured(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture: np.ndarray,
) -> dict[str, Path]:
    """Write a genuinely texture-mapped OBJ: ``.obj`` + ``.mtl`` + ``.png``.

    This is the "textured 3D mesh" deliverable, as distinct from
    ``export_obj``'s per-vertex colour: the ``.mtl`` here references a real
    image via ``map_Kd``, and every vertex carries a ``vt`` UV coordinate
    into it, so detail is limited by the source video's resolution rather
    than by mesh density.

    ``vertices``/``faces``/``uv`` must be the *unwrapped* topology from
    ``fusion.texture.bake_texture`` -- unwrapping splits vertices along
    chart seams, and pairing the original faces with the unwrapped UVs
    produces a mesh whose texture is scrambled in a way that looks almost
    right, which is worse than looking obviously wrong.

    Returns ``{name: path}`` for all three files written, so the caller can
    report every artifact rather than just the ``.obj``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    png_path = path.with_suffix(".png")

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) != len(vertices):
        raise ValueError(f"uv has {len(uv)} entries but there are {len(vertices)} vertices")

    # cv2 writes BGR; `texture` is RGB by contract (see bake_texture).
    cv2.imwrite(str(png_path), np.asarray(texture)[:, :, ::-1])

    mtl_path.write_text(
        "# DRISHTI-3D textured material\n"
        "newmtl drishti3d_texture\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "illum 1\n"
        f"map_Kd {png_path.name}\n"
    )

    lines = [f"mtllib {mtl_path.name}", "usemtl drishti3d_texture"]
    lines += [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices]
    lines += [f"vt {t[0]:.6f} {t[1]:.6f}" for t in uv]
    # OBJ is 1-indexed. Vertex and texture indices are identical because
    # unwrapping already split every vertex that needed more than one UV.
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in faces]

    path.write_text("\n".join(lines) + "\n")
    return {"obj_textured": path, "mtl": mtl_path, "texture_png": png_path}


def export_fbx(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Write an ASCII FBX 7.4 mesh, with optional per-vertex colour.

    FBX is on this project's required output-format list, and nothing in
    the Python ecosystem writes it without a heavyweight dependency:
    trimesh does not export FBX at all, and Autodesk's own SDK is a
    closed-source native package with no wheel. The ASCII flavour of the
    format is, however, plain text with a documented object graph, and a
    triangle mesh needs only a small subset of it -- so it is hand-rolled
    here for the same reason PLY and GLB are (see this module's docstring).

    Deliberately ASCII, not binary: the binary flavour carries a 32-bit
    file offset in every nested node header, which means writing it
    requires two passes and makes a malformed file trivially easy to
    produce and very hard to diagnose. Every importer that reads FBX reads
    ASCII, and these meshes are handed to viewers rather than streamed.

    Colour is written as a ``LayerElementColor`` with ``ByVertice``/
    ``Direct`` mapping. Confidence and semantic class are NOT exported:
    FBX has no convention for arbitrary per-vertex scalars that any
    importer would surface, and inventing one would produce a channel
    nothing can read -- use PLY/LAS/GLB when the trust layer must travel.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # FBX encodes polygon ends by negating-and-decrementing the last index
    # of each face, which is how a reader finds polygon boundaries in an
    # otherwise flat index list.
    poly_index = faces.copy()
    poly_index[:, -1] = -poly_index[:, -1] - 1

    verts_flat = ",".join(f"{v:.6f}" for v in vertices.reshape(-1))
    index_flat = ",".join(str(int(i)) for i in poly_index.reshape(-1))

    colour_layer = ""
    # The Layer block must only declare a LayerElement that actually
    # exists. Emitting the declaration unconditionally leaves a dangling
    # reference to a colour layer that was never written, which a strict
    # importer is entitled to reject.
    layer_block = """
        Layer: 0 {
            Version: 100
        }"""
    if colors is not None and len(colors) == len(vertices):
        rgba = np.concatenate(
            [np.asarray(colors, dtype=np.float64) / 255.0, np.ones((len(colors), 1))], axis=1
        )
        colour_flat = ",".join(f"{c:.4f}" for c in rgba.reshape(-1))
        colour_index = ",".join(str(i) for i in range(len(vertices)))
        colour_layer = f"""
        LayerElementColor: 0 {{
            Version: 101
            Name: "VertexColors"
            MappingInformationType: "ByVertice"
            ReferenceInformationType: "IndexToDirect"
            Colors: *{len(rgba) * 4} {{
                a: {colour_flat}
            }}
            ColorIndex: *{len(vertices)} {{
                a: {colour_index}
            }}
        }}"""
        layer_block = """
        Layer: 0 {
            Version: 100
            LayerElement:  {
                Type: "LayerElementColor"
                TypedIndex: 0
            }
        }"""

    content = f"""; FBX 7.4.0 project file
; Written by DRISHTI-3D

FBXHeaderExtension:  {{
    FBXHeaderVersion: 1003
    FBXVersion: 7400
    Creator: "DRISHTI-3D"
}}
GlobalSettings:  {{
    Version: 1000
    Properties70:  {{
        P: "UpAxis", "int", "Integer", "",2
        P: "UpAxisSign", "int", "Integer", "",1
        P: "FrontAxis", "int", "Integer", "",1
        P: "FrontAxisSign", "int", "Integer", "",-1
        P: "CoordAxis", "int", "Integer", "",0
        P: "CoordAxisSign", "int", "Integer", "",1
        P: "UnitScaleFactor", "double", "Number", "",1
    }}
}}

Definitions:  {{
    Version: 100
    Count: 2
    ObjectType: "Model" {{
        Count: 1
    }}
    ObjectType: "Geometry" {{
        Count: 1
    }}
}}

Objects:  {{
    Geometry: 1000000, "Geometry::drishti3d", "Mesh" {{
        Vertices: *{vertices.size} {{
            a: {verts_flat}
        }}
        PolygonVertexIndex: *{poly_index.size} {{
            a: {index_flat}
        }}
        GeometryVersion: 124{colour_layer}{layer_block}
    }}
    Model: 2000000, "Model::drishti3d", "Mesh" {{
        Version: 232
        Properties70:  {{
            P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
        }}
        Shading: T
        Culling: "CullingOff"
    }}
}}

Connections:  {{
    C: "OO",2000000,0
    C: "OO",1000000,2000000
}}
"""
    path.write_text(content)


def export_obj(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    mtl: bool = True,
) -> None:
    """Write a Wavefront OBJ mesh.

    OBJ has no standard way to carry arbitrary per-vertex scalars, so
    ``confidence`` is not (and cannot meaningfully be) exported here --
    use PLY, GLB, or LAS when the trust layer needs to travel with the
    file. Vertex colour is written via the widely-supported (if
    non-standard) ``v x y z r g b`` extension when ``colors`` is given,
    in addition to a flat placeholder ``.mtl`` (referenced via ``mtllib``/
    ``usemtl``) when ``mtl=True``, since some importers only honour
    material-based colour.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    lines: list[str] = []
    if mtl:
        mtl_path = path.with_suffix(".mtl")
        lines.append(f"mtllib {mtl_path.name}")
        lines.append("usemtl drishti3d_default")

    if colors is not None:
        colors_f = np.asarray(colors, dtype=np.float64) / 255.0
        for v, c in zip(vertices, colors_f, strict=True):
            lines.append(f"v {v[0]:.7g} {v[1]:.7g} {v[2]:.7g} {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}")
    else:
        for v in vertices:
            lines.append(f"v {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}")

    for face in faces:
        # OBJ vertex indices are 1-based.
        lines.append(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}")

    path.write_text("\n".join(lines) + "\n")

    if mtl:
        mtl_path = path.with_suffix(".mtl")
        mtl_path.write_text(
            "newmtl drishti3d_default\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\nd 1.0\nillum 1\n"
        )


# ---------------------------------------------------------------------------
# glTF 2.0 binary (GLB)
# ---------------------------------------------------------------------------

_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_GLB_CHUNK_JSON = 0x4E4F534A
_GLB_CHUNK_BIN = 0x004E4942

_GLTF_COMPONENT_FLOAT = 5126
_GLTF_COMPONENT_UBYTE = 5121
_GLTF_COMPONENT_UINT = 5125


def _pad_bytes(data: bytes, align: int, pad_char: bytes) -> bytes:
    remainder = len(data) % align
    if remainder == 0:
        return data
    return data + pad_char * (align - remainder)


def export_glb(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> None:
    """Write a glTF 2.0 binary (.glb) file, by hand (no gltf library dependency).

    Vertex colour is written as the standard ``COLOR_0`` attribute
    (normalized ``UNSIGNED_BYTE`` VEC4, alpha forced to 255); confidence
    is written as ``_CONFIDENCE`` -- an application-specific attribute,
    per the glTF 2.0 spec's convention that any non-standard vertex
    attribute name must start with an underscore -- as a raw (non
    -normalized) ``UNSIGNED_BYTE`` SCALAR holding the ``types.Confidence``
    enum value directly (0/1/2), so a confidence-aware viewer can recover
    exact tiers rather than a lossy float.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    n = vertices.shape[0]

    buffer_chunks: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    attributes: dict[str, int] = {}

    def add_buffer_view(data: bytes, target: int | None) -> int:
        offset = sum(len(c) for c in buffer_chunks)
        # glTF requires bufferView byteOffset to be aligned to the
        # accessor's component size; 4-byte alignment covers every
        # component type used here.
        pad = (-offset) % 4
        if pad:
            buffer_chunks.append(b"\x00" * pad)
            offset += pad
        buffer_chunks.append(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    # POSITION
    pos_bytes = vertices.tobytes()
    pos_view = add_buffer_view(pos_bytes, 34962)  # ARRAY_BUFFER
    accessors.append(
        {
            "bufferView": pos_view,
            "componentType": _GLTF_COMPONENT_FLOAT,
            "count": n,
            "type": "VEC3",
            "min": vertices.min(axis=0).tolist() if n else [0.0, 0.0, 0.0],
            "max": vertices.max(axis=0).tolist() if n else [0.0, 0.0, 0.0],
        }
    )
    attributes["POSITION"] = len(accessors) - 1

    if colors is not None:
        colors_u8 = np.asarray(colors, dtype=np.uint8)
        rgba = np.zeros((n, 4), dtype=np.uint8)
        rgba[:, :3] = colors_u8[:, :3]
        rgba[:, 3] = 255
        color_view = add_buffer_view(rgba.tobytes(), 34962)
        accessors.append(
            {
                "bufferView": color_view,
                "componentType": _GLTF_COMPONENT_UBYTE,
                "normalized": True,
                "count": n,
                "type": "VEC4",
            }
        )
        attributes["COLOR_0"] = len(accessors) - 1

    if confidence is not None:
        conf_u8 = np.asarray(confidence, dtype=np.uint8)
        conf_view = add_buffer_view(conf_u8.tobytes(), 34962)
        accessors.append(
            {
                "bufferView": conf_view,
                "componentType": _GLTF_COMPONENT_UBYTE,
                "normalized": False,
                "count": n,
                "type": "SCALAR",
            }
        )
        attributes["_CONFIDENCE"] = len(accessors) - 1

    primitive: dict = {"attributes": attributes, "mode": 4}  # TRIANGLES

    if faces is not None and len(faces) > 0:
        faces_u32 = np.asarray(faces, dtype=np.uint32)
        idx_view = add_buffer_view(faces_u32.tobytes(), 34963)  # ELEMENT_ARRAY_BUFFER
        accessors.append(
            {
                "bufferView": idx_view,
                "componentType": _GLTF_COMPONENT_UINT,
                "count": int(faces_u32.size),
                "type": "SCALAR",
            }
        )
        primitive["indices"] = len(accessors) - 1
    else:
        primitive["mode"] = 0  # POINTS, when there's no index buffer

    bin_data = b"".join(buffer_chunks)

    gltf_json = {
        "asset": {"version": "2.0", "generator": "drishti3d.export.formats"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_bytes = json.dumps(gltf_json).encode("utf-8")
    json_bytes = _pad_bytes(json_bytes, 4, b" ")
    bin_bytes = _pad_bytes(bin_data, 4, b"\x00")

    json_chunk = struct.pack("<II", len(json_bytes), _GLB_CHUNK_JSON) + json_bytes
    bin_chunk = struct.pack("<II", len(bin_bytes), _GLB_CHUNK_BIN) + bin_bytes if bin_bytes else b""

    total_length = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(json_chunk)
        f.write(bin_chunk)


def read_glb(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Parse a GLB written by ``export_glb`` back into ``(vertices, faces, colors, confidence)``.

    Validates the container header/chunk structure (magic, version,
    chunk types/lengths) while it's at it -- a malformed GLB should fail
    here loudly, not downstream in some other tool.
    """
    path = Path(path)
    data = path.read_bytes()

    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"not a GLB file (bad magic {magic:#x})")
    if version != _GLB_VERSION:
        raise ValueError(f"unsupported glTF binary version {version}")
    if length != len(data):
        raise ValueError(f"GLB header length {length} does not match file size {len(data)}")

    offset = 12
    json_bytes = None
    bin_bytes = b""
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == _GLB_CHUNK_JSON:
            json_bytes = chunk_data
        elif chunk_type == _GLB_CHUNK_BIN:
            bin_bytes = chunk_data
        else:
            raise ValueError(f"unexpected GLB chunk type {chunk_type:#x}")

    if json_bytes is None:
        raise ValueError("GLB file has no JSON chunk")
    gltf = json.loads(json_bytes.decode("utf-8"))

    accessors = gltf["accessors"]
    buffer_views = gltf["bufferViews"]

    def read_accessor(idx: int) -> np.ndarray:
        acc = accessors[idx]
        view = buffer_views[acc["bufferView"]]
        comp_dtype = {
            _GLTF_COMPONENT_FLOAT: np.float32,
            _GLTF_COMPONENT_UBYTE: np.uint8,
            _GLTF_COMPONENT_UINT: np.uint32,
        }[acc["componentType"]]
        n_components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
        start = view["byteOffset"]
        count = acc["count"]
        raw = np.frombuffer(bin_bytes, dtype=comp_dtype, count=count * n_components, offset=start)
        return raw.reshape(count, n_components) if n_components > 1 else raw.reshape(count)

    primitive = gltf["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]

    vertices = read_accessor(attributes["POSITION"]).astype(np.float64)

    colors = None
    if "COLOR_0" in attributes:
        colors = read_accessor(attributes["COLOR_0"])[:, :3].astype(np.uint8)

    confidence = None
    if "_CONFIDENCE" in attributes:
        confidence = read_accessor(attributes["_CONFIDENCE"]).astype(np.uint8)

    faces = None
    if "indices" in primitive:
        faces = read_accessor(primitive["indices"]).astype(np.int64).reshape(-1, 3)

    return vertices, faces, colors, confidence


# ---------------------------------------------------------------------------
# LAS
# ---------------------------------------------------------------------------


def export_las(path: str | Path, pc: PointCloud, crs: str | None = None) -> None:
    """Write a LAS point cloud, with per-point confidence (and covariance trace) as extra bytes.

    LAS's "extra bytes" VLR mechanism is the standard, tool-interoperable
    way to attach arbitrary custom per-point channels in the point-cloud/
    LIDAR world -- exactly the right place for "uncertainty as a
    first-class exportable channel" to live, rather than inventing a
    bespoke sidecar file a downstream GIS tool would have no idea how to
    associate with the points.
    """
    xyz = np.asarray(pc.xyz, dtype=np.float64)
    n = xyz.shape[0]

    header = laspy.LasHeader(point_format=3, version="1.2")
    if n > 0:
        mins = xyz.min(axis=0)
    else:
        mins = np.zeros(3)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])

    has_conf = pc.confidence is not None
    has_cov = pc.covariance is not None
    has_sem = pc.semantic_class is not None
    has_sem_conf = pc.semantic_confidence is not None
    if has_conf:
        header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.uint8, description="types.Confidence tier"))
    if has_cov:
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="cov_trace", type=np.float32, description="trace(covariance), m^2")
        )
    if has_sem:
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="semantic_class",
                type=np.uint8,
                description="semantics.classes.SemanticClass",
            )
        )
    if has_sem_conf:
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="semantic_conf",
                type=np.uint8,
                description="class vote agreement, 0-255",
            )
        )

    if crs is not None:
        import pyproj

        header.add_crs(pyproj.CRS.from_user_input(crs))

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if pc.rgb is not None:
        rgb16 = (np.asarray(pc.rgb, dtype=np.float64) / 255.0 * 65535.0).astype(np.uint16)
        las.red = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue = rgb16[:, 2]

    if has_conf:
        las.confidence = np.asarray(pc.confidence, dtype=np.uint8)
    if has_cov:
        trace = np.trace(pc.covariance, axis1=1, axis2=2).astype(np.float32)
        las.cov_trace = trace

    if has_sem:
        from drishti3d.semantics.classes import ASPRS_FROM_CANONICAL

        sem = np.asarray(pc.semantic_class, dtype=np.uint8)
        las.semantic_class = sem

        # Also populate LAS's *standard* classification field, so the file
        # is immediately meaningful to CloudCompare/QGIS/LAStools without
        # any knowledge of this project. Lossy by design -- see
        # ASPRS_FROM_CANONICAL's comment on why unmappable classes become
        # "unclassified" instead of a plausible near-miss.
        in_range = sem < len(ASPRS_FROM_CANONICAL)
        classification = np.ones(n, dtype=np.uint8)
        classification[in_range] = ASPRS_FROM_CANONICAL[sem[in_range]]
        las.classification = classification

    if has_sem_conf:
        las.semantic_conf = _quantize_unit(pc.semantic_confidence)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------


def export_xyz(path: str | Path, pc: PointCloud) -> None:
    """Write a plain-text XYZ point cloud (``x y z [r g b]`` per line).

    The simplest, most universally-readable format offered here -- and,
    like OBJ, has no room for a confidence channel; use it for
    interchange with tools that only understand bare XYZ, not as the
    format of record.
    """
    xyz = np.asarray(pc.xyz, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if pc.rgb is not None:
        rgb = np.asarray(pc.rgb, dtype=np.uint8)
        stacked = np.concatenate([xyz, rgb.astype(np.float64)], axis=1)
        fmt = ["%.7g", "%.7g", "%.7g", "%d", "%d", "%d"]
    else:
        stacked = xyz
        fmt = ["%.7g", "%.7g", "%.7g"]

    np.savetxt(path, stacked, fmt=fmt)
