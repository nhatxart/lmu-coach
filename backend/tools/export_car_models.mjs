// Write STL previews of the chase-view car models (static/car_models.js),
// one binary STL per class at 1:1 and at 1:43, in millimetres, so the shapes
// can be checked (and printed) in a 3D-printer slicer before the UI uses them.
//
//   node backend/tools/export_car_models.mjs [outDir]     (default docs/car_models)
//
// Uses the same builder and the same vendored three.js as the UI, in Node:
// only core geometry is touched (no renderer, no loaders, no DOM). The STL
// writer is our own because three's STLExporter is an addon we do not vendor.

import * as THREE from '../app/static/vendor/three/three.module.js';
import { buildCar } from '../app/static/car_models.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const json = JSON.parse(fs.readFileSync(path.join(repo, 'backend', 'app', 'static', 'car_models.json'), 'utf8'));
const outDir = path.resolve(process.argv[2] || path.join(repo, 'docs', 'car_models'));
fs.mkdirSync(outDir, { recursive: true });

// Model frame is Y-up (nose +X, right +Z); slicers are Z-up. A +90 degree
// turn about X maps (x, y, z) -> (x, -z, y): a rotation, not a mirror, so the
// car's left stays its left and the triangle winding stays outward.
const ZUP = new THREE.Matrix4().makeRotationX(Math.PI / 2);

// All triangles of a car's meshes in world (car) space, Z-up, metres.
function triangles(group) {
  group.updateMatrixWorld(true);
  const tris = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  group.traverse((o) => {
    if (!o.isMesh) return;                                // skip the outline lines
    const m = new THREE.Matrix4().multiplyMatrices(ZUP, o.matrixWorld);
    const pos = o.geometry.attributes.position;
    const idx = o.geometry.index;
    const n = idx ? idx.count : pos.count;
    for (let i = 0; i < n; i += 3) {
      const i0 = idx ? idx.getX(i) : i, i1 = idx ? idx.getX(i + 1) : i + 1, i2 = idx ? idx.getX(i + 2) : i + 2;
      a.fromBufferAttribute(pos, i0).applyMatrix4(m);
      b.fromBufferAttribute(pos, i1).applyMatrix4(m);
      c.fromBufferAttribute(pos, i2).applyMatrix4(m);
      tris.push([a.toArray(), b.toArray(), c.toArray()]);
    }
  });
  return tris;
}

// Binary STL: 80-byte header, uint32 count, then per triangle the normal and
// three vertices (12 float32, little endian) and a uint16 attribute word.
function writeStl(file, tris, scale, header) {
  const buf = Buffer.alloc(84 + 50 * tris.length);
  buf.write(header.slice(0, 80).padEnd(80, ' '), 0, 'ascii');
  buf.writeUInt32LE(tris.length, 80);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let o = 84;
  for (const t of tris) {
    const v = t.map((p) => p.map((q) => q * scale));
    const u = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
    const w = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
    let nx = u[1] * w[2] - u[2] * w[1], ny = u[2] * w[0] - u[0] * w[2], nz = u[0] * w[1] - u[1] * w[0];
    const l = Math.hypot(nx, ny, nz);
    // A zero-area triangle has no normal; write 0,0,0 (finite), slicers
    // recompute normals from the winding anyway.
    if (l > 0) { nx /= l; ny /= l; nz /= l; } else { nx = ny = nz = 0; }
    for (const x of [nx, ny, nz]) { buf.writeFloatLE(x, o); o += 4; }
    for (const p of v) {
      for (let k = 0; k < 3; k++) {
        buf.writeFloatLE(p[k], o); o += 4;
        if (p[k] < lo[k]) lo[k] = p[k];
        if (p[k] > hi[k]) hi[k] = p[k];
      }
    }
    buf.writeUInt16LE(0, o); o += 2;
  }
  fs.writeFileSync(file, buf);
  return { lo, hi };
}

const f1 = (x) => x.toFixed(1);
for (const [cls, dims] of Object.entries(json.classes)) {
  const car = buildCar(THREE, cls, dims);
  const tris = triangles(car.group);
  for (const [tag, scale] of [['1to1', 1000], ['1to43', 1000 / 43]]) {
    const name = `${cls}_${tag}_mm.stl`;
    const { lo, hi } = writeStl(path.join(outDir, name), tris, scale,
      `LMU Coach ${cls} car model, ${tag.replace('to', ':')}, millimetres, Z up, nose +X`);
    console.log(`${name}: ${tris.length} triangles, ${f1(hi[0] - lo[0])} x ${f1(hi[1] - lo[1])} x ${f1(hi[2] - lo[2])} mm `
      + `(L x W x H; min z ${lo[2].toFixed(2)} mm)`);
  }
}
console.log(`written to ${outDir}`);
