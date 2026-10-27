"""Car models for the chase view (static/car_models.json + car_models.js) and
their STL previews (tools/export_car_models.mjs).

The STL files are what the user opens in a slicer to approve the shapes, so
these tests check what a slicer relies on: a valid binary STL (size matches
the triangle count, finite normals and vertices), the stated scale (the 1:1
bounding box is the JSON's length x width x height in mm, 1:43 is that / 43),
and the wheels standing on the bed (min z = 0).

The builder runs in Node (the same ES module the UI imports), so every test
that needs it skips when node is not installed.
"""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "backend" / "app" / "static"
JSON_PATH = STATIC / "car_models.json"
EXPORTER = REPO / "backend" / "tools" / "export_car_models.mjs"
CLASSES = ("LMGT3", "Hypercar", "LMP2", "LMP3")

DIM_KEYS = ("length", "width", "height", "wheelbase", "track_front", "track_rear",
            "overhang_front", "tyre_diameter", "tyre_width_front", "tyre_width_rear",
            "ride_height", "wing", "splitter", "fin")
WING_KEYS = ("span", "chord", "height_top", "setback", "endplate_height", "mount")

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _load_json():
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def _read_stl(path: Path):
    """Binary STL -> (normals (n,3), vertices (n,3,3)); asserts the size."""
    data = path.read_bytes()
    assert len(data) >= 84, f"{path.name}: shorter than an STL header"
    (n,) = struct.unpack_from("<I", data, 80)
    assert len(data) == 84 + 50 * n, f"{path.name}: size {len(data)} != 84 + 50*{n}"
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    arr = np.frombuffer(data, dtype=rec, count=n, offset=84)
    return arr["n"].astype(float), arr["v"].astype(float)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if NODE is None:
        pytest.skip("node not installed")
    out = tmp_path_factory.mktemp("car_stl")
    r = subprocess.run([NODE, str(EXPORTER), str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"exporter failed:\n{r.stdout}\n{r.stderr}"
    return out, r.stdout


def test_json_has_every_class_and_every_key():
    j = _load_json()
    assert j["version"] == 1
    # LMP2 and LMP3 have their own entries since round 2, so no alias may
    # shadow a class, and every alias points at a real class.
    assert isinstance(j["aliases"], dict)
    for name, target in j["aliases"].items():
        assert name not in j["classes"] and target in j["classes"]
    assert isinstance(j.get("notes"), (str, list)) and j["notes"]
    for cls in CLASSES:
        d = j["classes"][cls]
        for k in DIM_KEYS:
            assert k in d, f"{cls}: missing {k}"
        for k in WING_KEYS:
            assert k in d["wing"], f"{cls}: wing missing {k}"
        assert set(d["splitter"]) >= {"depth", "width"}
        assert d["wing"]["mount"] in ("swan_neck", "pylon")
        # The tyres must sit inside the body width and the axles inside the length.
        assert max(d["track_front"] + d["tyre_width_front"], d["track_rear"] + d["tyre_width_rear"]) <= d["width"]
        assert d["wheelbase"] + d["overhang_front"] < d["length"]
        assert d["wing"]["height_top"] <= d["height"] + 1e-9
    c = j["classes"]
    assert c["LMGT3"]["wing"]["mount"] == "swan_neck"
    assert c["Hypercar"]["wing"]["mount"] == "pylon"
    assert c["LMP2"]["wing"]["mount"] == "pylon"
    assert c["LMP3"]["wing"]["mount"] == "swan_neck"
    for cls in ("Hypercar", "LMP2", "LMP3"):
        assert c[cls]["fin"] is True
    assert c["LMGT3"]["fin"] is False
    # LMP3 is the smaller prototype; LMP2 matches the Oreca 07's published size.
    assert c["LMP3"]["length"] < c["LMP2"]["length"] < c["Hypercar"]["length"]
    assert c["LMP3"]["wheelbase"] < c["LMP2"]["wheelbase"] < c["Hypercar"]["wheelbase"]
    assert (c["LMP2"]["length"], c["LMP2"]["width"], c["LMP2"]["height"]) == (4.745, 1.895, 1.045)
    # Round 2: the GT3 wing sits at (or above) the roof line, like the 911 GT3 R.
    assert c["LMGT3"]["wing"]["height_top"] >= c["LMGT3"]["roof_height"]


@needs_node
@pytest.mark.parametrize("cls", CLASSES)
def test_stl_valid_and_to_scale(exported, cls):
    out, stdout = exported
    d = _load_json()["classes"][cls]
    one = out / f"{cls}_1to1_mm.stl"
    small = out / f"{cls}_1to43_mm.stl"
    n1, v1 = _read_stl(one)
    n43, v43 = _read_stl(small)
    for n, v in ((n1, v1), (n43, v43)):
        assert len(v) > 100
        assert np.isfinite(n).all() and np.isfinite(v).all()
    assert len(v1) < 5000, f"{cls}: {len(v1)} triangles, budget is 5000"
    assert len(v1) == len(v43)
    # Outward winding: the parts are closed solids, so the summed signed
    # volume is positive (a flipped part would subtract its volume twice).
    vol = np.einsum("ij,ij->i", v1[:, 0], np.cross(v1[:, 1], v1[:, 2])).sum() / 6
    box_vol = d["length"] * d["width"] * d["height"] * 1e9
    assert 0.3 * box_vol < vol < box_vol, f"{cls}: signed volume {vol / 1e9:.2f} m3"
    # The stored normals agree with the winding (slicers may use either).
    # Slivers under 0.5 mm2 are skipped: float32 rounding of their vertices
    # can flip the recomputed cross product.
    calc = np.cross(v1[:, 1] - v1[:, 0], v1[:, 2] - v1[:, 0])
    ok = np.linalg.norm(calc, axis=1) > 1.0
    assert (np.einsum("ij,ij->i", calc[ok], n1[ok]) > 0).all()

    pts = v1.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi - lo
    # Z-up for slicers: X = length, Y = width, Z = height (wing included).
    for got, want, name in ((size[0], d["length"], "length"), (size[1], d["width"], "width"),
                            (size[2], d["height"], "height")):
        assert abs(got - want * 1000) <= 0.02 * want * 1000, f"{cls} {name}: {got:.1f} mm vs {want * 1000:.0f} mm"
    assert abs(lo[2]) < 0.5, f"{cls}: lowest point {lo[2]:.2f} mm, wheels must stand on z = 0"

    p43 = v43.reshape(-1, 3)
    np.testing.assert_allclose(p43.min(axis=0), lo / 43, atol=1e-3)
    np.testing.assert_allclose(p43.max(axis=0), hi / 43, atol=1e-3)
    assert f"{cls}_1to1_mm.stl" in stdout and "triangles" in stdout


@needs_node
def test_builder_frame_and_class_lookup():
    """buildCar's frame (nose +X, up +Y, right +Z, origin at mid-wheelbase on
    the ground; wheels at their hubs) and classDims' fallback, run in Node."""
    script = """
const THREE = await import(process.argv[1]);
const { buildCar, classDims } = await import(process.argv[2]);
const fs = await import('node:fs');
const j = JSON.parse(fs.readFileSync(new URL(process.argv[3]), 'utf8'));
const out = {};
for (const c of ['LMGT3', 'Hypercar', 'LMP2', 'LMP3', 'Mystery', undefined]) {
  const r = classDims(j, c); out[String(c)] = [r.cls, r.source];
}
const al = classDims({ classes: j.classes, aliases: { GTE: 'LMGT3' } }, 'GTE');
out.alias = [al.cls, al.source];
out.builds = {};
for (const c of Object.keys(j.classes)) {
  const k = buildCar(THREE, c, j.classes[c]);
  out.builds[c] = Object.keys(k.wheels).length;
}
const r = classDims(j, 'Hypercar');
const car = buildCar(THREE, r.cls, r.dims);
const w = {};
for (const k of ['fl', 'fr', 'rl', 'rr']) w[k] = car.wheels[k].position.toArray();
out.wheels = w;
out.outline = car.outline.isLineSegments === true;
out.bodyIsObject = car.body.isObject3D === true && car.group.children.includes(car.body);
console.log(JSON.stringify(out));
"""
    urls = [(STATIC / p).as_uri() for p in ("vendor/three/three.module.js", "car_models.js", "car_models.json")]
    r = subprocess.run([NODE, "--input-type=module", "-e", script, *urls],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["LMGT3"] == ["LMGT3", "exact"]
    assert out["Hypercar"] == ["Hypercar", "exact"]
    assert out["LMP2"] == ["LMP2", "exact"]
    assert out["LMP3"] == ["LMP3", "exact"]
    assert out["alias"] == ["LMGT3", "alias"]
    assert out["builds"] == {c: 4 for c in CLASSES}
    assert out["Mystery"] == ["LMGT3", "fallback"]
    assert out["undefined"] == ["LMGT3", "fallback"]
    d = _load_json()["classes"]["Hypercar"]
    hx, r_ = d["wheelbase"] / 2, d["tyre_diameter"] / 2
    want = {"fl": [hx, r_, -d["track_front"] / 2], "fr": [hx, r_, d["track_front"] / 2],
            "rl": [-hx, r_, -d["track_rear"] / 2], "rr": [-hx, r_, d["track_rear"] / 2]}
    for k, v in want.items():
        np.testing.assert_allclose(out["wheels"][k], v, atol=1e-6)
    assert out["outline"] is True
    assert out["bodyIsObject"] is True
