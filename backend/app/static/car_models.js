// Procedural car models for the chase view: a representative LMGT3 car and a
// Le Mans Hypercar, built from three.js core primitives and extrusions.
//
// Shared by two callers, so it must stay DOM-free and import-free (THREE is
// passed in):
//   - index.html's chase view (lazy import('/static/car_models.js')),
//   - tools/export_car_models.mjs (Node), which writes the STL previews the
//     user approved the shapes from. One builder = the STL is what the UI draws.
//
// Frame (the chase view's car frame): nose +X, up +Y, right +Z, origin on the
// ground at mid-wheelbase. Dimensions come from static/car_models.json; the
// tyre positions (wheelbase, tracks, tyre size) are exact because the track
// limits are judged from them, the body is only a recognisable shell around
// them. The shell's own proportions (hood, beltline, deck heights) are fixed
// per class below, relative to the axles and the nose/tail from the JSON:
// representative shapes, not data.

// ---- class lookup ---------------------------------------------------------

// Unknown or missing class -> LMGT3, and say so ('fallback'), so the UI can
// label the car as a guess (Review focus 2).
export function classDims(json, carClass) {
  const classes = (json && json.classes) || {};
  if (carClass && Object.prototype.hasOwnProperty.call(classes, carClass)) {
    return { cls: carClass, dims: classes[carClass], source: 'exact' };
  }
  const alias = json && json.aliases && carClass ? json.aliases[carClass] : undefined;
  if (alias && classes[alias]) return { cls: alias, dims: classes[alias], source: 'alias' };
  return { cls: 'LMGT3', dims: classes.LMGT3, source: 'fallback' };
}

// ---- small geometry helpers -----------------------------------------------

const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);
const lerp = (a, b, t) => a + (b - a) * t;

// Piecewise-linear lookup on [[x, v], ...] (any x order), clamped at the ends.
function table(pairs) {
  const p = pairs.slice().sort((a, b) => a[0] - b[0]);
  return (x) => {
    if (x <= p[0][0]) return p[0][1];
    for (let i = 1; i < p.length; i++) {
      if (x <= p[i][0]) {
        const t = (x - p[i - 1][0]) / (p[i][0] - p[i - 1][0] || 1);
        return lerp(p[i - 1][1], p[i][1], t);
      }
    }
    return p[p.length - 1][1];
  };
}

// Extra points along long edges of a closed profile (profileSolid applies
// its half width per vertex, so a varying width needs vertices to act on;
// the thin plates pass a large step and keep their outline as is).
function densify(pts, step) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
    for (let k = 0; k < n; k++) out.push([lerp(a[0], b[0], k / n), lerp(a[1], b[1], k / n)]);
  }
  // Drop exact repeats: they make zero-area side triangles in the STL.
  return out.filter((p, i) => {
    const q = out[(i + out.length - 1) % out.length];
    return Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-6;
  });
}

// Arch over a wheel for a body's bottom line, walked front -> rear: up the
// front side, over the top, down the rear side (vertical steps at the feet).
function arch(xc, hubY, radius, footY, n = 10) {
  const pts = [[xc + radius, footY]];
  for (let i = 0; i <= n; i++) {
    const th = (Math.PI * i) / n;
    pts.push([xc + radius * Math.cos(th), hubY + radius * Math.sin(th)]);
  }
  pts.push([xc - radius, footY]);
  return pts;
}

// Lofted body: cross-sections along x, each from the bottom line up to the
// top line (both given front -> rear as [x, y] polylines), half width
// halfWidth(x, y) per side. Unlike a side-profile extrusion whose cap
// triangles are bent by a per-vertex taper (long slivers that shade as
// streaks), the side here is a grid of near-planar quads, so the shell
// shades smoothly. halfWidth must not grow with y (true for every shoulder /
// tumblehome below), which keeps each section convex for the end caps.
//
// Sections sit at every densified bottom point (so the arches, including
// their vertical feet, are exact) plus every top key point (windscreen and
// roof kinks stay sharp).
function loftSolid(T, top, bottom, halfWidth, levels = 4, step = 0.15) {
  const topY = table(top);
  const bot = [];
  for (let i = 0; i < bottom.length - 1; i++) {
    const a = bottom[i], b = bottom[i + 1];
    const n = Math.max(1, Math.ceil(Math.abs(b[0] - a[0]) / step));
    for (let k = 0; k < n; k++) bot.push([lerp(a[0], b[0], k / n), lerp(a[1], b[1], k / n)]);
  }
  bot.push(bottom[bottom.length - 1]);
  const botY = table(bottom.filter((p, i) => i === 0 || p[0] !== bottom[i - 1][0]));
  const secs = bot.map(([x, y]) => [x, y]);
  for (const [x] of top) {
    if (!secs.some((s) => Math.abs(s[0] - x) < 1e-6)) secs.push([x, botY(x)]);
  }
  // front -> rear; points with equal x keep their walk order (arch feet).
  const order = secs.map((s, i) => [s, i]);
  order.sort((p, q) => (q[0][0] - p[0][0]) || (p[1] - q[1]));
  const S = order.map((p) => p[0]).filter((s, i, a) =>
    i === 0 || Math.abs(s[0] - a[i - 1][0]) > 1e-6 || Math.abs(s[1] - a[i - 1][1]) > 1e-6);

  const ring = 2 * levels;
  const pos = [];
  for (const [x, yb] of S) {
    const yt = Math.max(topY(x), yb + 0.005);
    const ys = [];
    for (let j = 0; j < levels; j++) ys.push(lerp(yb, yt, j / (levels - 1)));
    for (let j = 0; j < levels; j++) pos.push(x, ys[j], halfWidth(x, ys[j]));
    for (let j = levels - 1; j >= 0; j--) pos.push(x, ys[j], -halfWidth(x, ys[j]));
  }
  const idx = [];
  for (let s = 0; s < S.length - 1; s++) {
    for (let k = 0; k < ring; k++) {
      const a = s * ring + k, b = s * ring + ((k + 1) % ring), c = a + ring, e = b + ring;
      // The ring runs up the right (+Z) side and down the left; c is one
      // section further back. (a, b, c) then faces outwards.
      idx.push(a, b, c, b, e, c);
    }
  }
  // End caps get their own vertices (a crisp nose/tail face, not smoothed
  // into the sides); fan triangulation, the sections are convex. Seen from
  // the front the ring runs clockwise, so the nose fan is reversed.
  for (const s of [0, S.length - 1]) {
    const base = pos.length / 3;
    for (let k = 0; k < ring; k++) pos.push(pos[(s * ring + k) * 3], pos[(s * ring + k) * 3 + 1], pos[(s * ring + k) * 3 + 2]);
    for (let k = 1; k < ring - 1; k++) {
      if (s === 0) idx.push(base, base + k + 1, base + k);
      else idx.push(base, base + k, base + k + 1);
    }
  }
  orientOutward(pos, idx);
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// Safety net for the winding above: a closed mesh wound outwards
// (counter-clockwise seen from outside) has a positive signed volume; if it
// came out negative, flip every triangle. Slicers and back-face culling both
// rely on outward winding.
function orientOutward(pos, idx) {
  const vol = (from, to) => {
    let v = 0;
    for (let i = from; i < to; i += 3) {
      const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
      v += pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1])
         - pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c])
         + pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c]);
    }
    return v;
  };
  if (vol(0, idx.length) < 0) {
    for (let i = 0; i < idx.length; i += 3) { const t = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = t; }
  }
}

// Side profile (x, y) extruded across, each vertex's z set to
// +-halfWidth(x, y). Used for the thin constant-thickness parts (wing,
// endplates, mounts, fin), where the caps stay planar. ExtrudeGeometry is
// DOM-free (Earcut triangulation), so this runs in Node for the exporter too.
function profileSolid(T, pts, halfWidth, step = 0.12) {
  const shape = new T.Shape(densify(pts, step).map(([x, y]) => new T.Vector2(x, y)));
  const g = new T.ExtrudeGeometry(shape, { depth: 1, steps: 1, bevelEnabled: false, curveSegments: 1 });
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    pos.setZ(i, (pos.getZ(i) < 0.5 ? -1 : 1) * halfWidth(x, y));
  }
  g.computeVertexNormals();
  return g;
}

// Thin plate (endplate, swan neck, pylon, fin): a profile with constant
// thickness, centred on z = 0 (position the mesh to place it).
const plate = (T, pts, thick) => profileSolid(T, pts, () => thick / 2, 10);

function box(T, sx, sy, sz) { return new T.BoxGeometry(sx, sy, sz); }

// Wing section in the side view: a thick-nosed, inverted-camber section at
// an angle of attack (trailing edge up, as on the cars), scaled to `chord`,
// leading edge at x = xLe, top of the section at y = top. Returns the closed
// outline and the upper surface (for attaching swan necks).
function wingSection(xLe, chord, top, aoaDeg = 12) {
  const S = [0, 0.02, 0.06, 0.12, 0.22, 0.36, 0.52, 0.7, 0.86, 1];
  const t = 0.11;                                     // max thickness / chord
  const thick = (s) => 5 * t * (0.2969 * Math.sqrt(s) - 0.126 * s - 0.3516 * s * s + 0.2843 * s ** 3 - 0.1036 * s ** 4);
  const camb = (s) => -0.05 * 4 * s * (1 - s);         // bowed down: a downforce wing
  const a = (aoaDeg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
  // chord coords: dx = -s*chord (rearwards), dy; rotate so the TE goes up.
  const rot = (s, dy) => {
    const dx = -s * chord, yy = dy * chord;
    return [dx * ca + yy * sa, -dx * sa + yy * ca];
  };
  const upper = S.map((s) => rot(s, camb(s) + thick(s) / 2));
  const lower = S.map((s) => rot(s, camb(s) - thick(s) / 2));
  const maxY = Math.max(...upper.map((p) => p[1]), ...lower.map((p) => p[1]));
  const minY = Math.min(...upper.map((p) => p[1]), ...lower.map((p) => p[1]));
  const place = (p) => [xLe + p[0], top - maxY + p[1]];
  const up = upper.map(place), lo = lower.map(place);
  // upper LE -> TE, then lower TE -> LE (skip the shared end points)
  const outline = up.concat(lo.slice(1, -1).reverse());
  return { outline, upper: up, bottom: top - maxY + minY };
}

// y of a polyline (sorted or not in x) at x.
function polyY(poly, x) {
  return table(poly.map((p) => [p[0], p[1]]))(x);
}

// Thick centre line -> closed strip (for the swan neck), via core SplineCurve.
function strip(T, ctrl, halfT, n = 18) {
  const c = new T.SplineCurve(ctrl.map(([x, y]) => new T.Vector2(x, y)));
  const pts = c.getSpacedPoints(n);
  const L = [], R = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    const dx = b.x - a.x, dy = b.y - a.y, l = Math.hypot(dx, dy) || 1;
    const nx = -dy / l, ny = dx / l;
    L.push([pts[i].x + nx * halfT, pts[i].y + ny * halfT]);
    R.push([pts[i].x - nx * halfT, pts[i].y - ny * halfT]);
  }
  return L.concat(R.reverse());
}

// ---- class shapes ---------------------------------------------------------
// Each returns a list of { geo, x?, y?, z? } body parts in the car frame.
// Absolute heights (hood, beltline, deck) are representative for the class.

function gt3Parts(T, d) {
  const xf = d.wheelbase / 2, xra = -d.wheelbase / 2;
  const xn = xf + d.overhang_front, xr = xn - d.length;
  const r = d.tyre_diameter / 2, rh = d.ride_height, W2 = d.width / 2;
  const roof = d.roof_height ?? d.height - 0.04;
  const Ra = r + 0.035;                                   // arch clearance
  const parts = [];

  // Lower body (top and bottom lines, front -> rear): nose, hood, front
  // fender peak, beltline, rear haunch, deck and tail; floor with arches over
  // both axles and the diffuser kick.
  const top = [
    [xn - 0.02, 0.30], [xn - 0.07, 0.48], [xn - 0.25, 0.64], [xf + 0.35, 0.77], [xf, 0.83],
    [xf - 0.55, 0.85], [-0.30, 0.87], [xra + 0.55, 0.93], [xra, 0.95], [xr + 0.30, 0.93],
    [xr + 0.03, 0.91], [xr, 0.86],
  ];
  const bottom = [
    [xn - 0.02, rh + 0.04], [xn - 0.25, rh + 0.01],
    ...arch(xf, r, Ra, rh + 0.02),
    ...arch(xra, r, Ra, rh + 0.02),
    [xr + 0.38, rh + 0.06], [xr + 0.08, 0.28], [xr, 0.42],
  ];
  // Full width (the class width) over each arch: the arch cut-out removes the
  // vertices right above the hub, so the flat part must span the arch sides.
  const planX = table([
    [xn, 0.80], [xn - 0.25, 0.93], [xf + Ra + 0.05, 1.0], [xf - Ra - 0.05, 1.0],
    [xf - 0.85, 0.955], [xra + 0.85, 0.96], [xra + Ra + 0.05, 1.0], [xra - Ra - 0.05, 1.0],
    [xr + 0.15, 0.95], [xr, 0.92],
  ]);
  // Shoulder: the side leans in above 0.6 m, so the top edge is not a box.
  const lowerHW = (x, y) => W2 * planX(x) * (1 - 0.07 * clamp01((y - 0.6) / 0.4));
  parts.push({ geo: loftSolid(T, top, bottom, lowerHW, 4) });

  // Greenhouse: windscreen, roof, long fastback rear window down to the
  // deck, narrower than the body with tumblehome (sits on the shoulder, so
  // its bottom is hidden inside the lower body).
  const xw = xf - 0.70;                                   // windscreen base
  const cabTop = [
    [xw, 0.84], [xw - 0.06, 0.88], [-0.05, roof - 0.012], [-0.25, roof], [-0.85, roof - 0.005],
    [-1.10, roof - 0.05], [-1.85, 0.95], [-1.95, 0.92],
  ];
  const cabX = table([[xw, 0.85], [xw - 0.35, 1.0], [-1.0, 1.0], [-1.95, 0.93]]);
  const cabHW = (x, y) => cabX(x) * lerp(0.84, 0.63, clamp01((y - 0.86) / (roof - 0.86)));
  parts.push({ geo: loftSolid(T, cabTop, [[xw, 0.78], [-1.95, 0.84]], cabHW, 2) });

  // Door mirrors at the A-pillar base, inside the body width.
  for (const s of [-1, 1]) parts.push({ geo: box(T, 0.10, 0.07, 0.13), x: xw - 0.06, y: 0.93, z: s * 0.92 });

  return { parts, xn, xr, deckY: (x) => polyY(top, x) };  // deck line: for the wing mounts
}

function hypercarParts(T, d) {
  const xf = d.wheelbase / 2, xra = -d.wheelbase / 2;
  const xn = xf + d.overhang_front, xr = xn - d.length;
  const r = d.tyre_diameter / 2, rh = d.ride_height, W2 = d.width / 2;
  const roof = d.roof_height ?? d.height - 0.05;
  const Ra = r + 0.035;
  const parts = [];

  // Central tub: low nose between the front fenders, cockpit, sidepods
  // widening into full-width rear fenders, short tail. The front wheels sit
  // outside it (in the fender pods below), so it has only the rear arch.
  const top = [
    [xn - 0.03, 0.20], [xn - 0.22, 0.35], [xn - 0.60, 0.48], [xf, 0.58], [xf - 0.60, 0.64],
    [0.20, 0.68], [-0.60, 0.73], [xra + 0.50, 0.83], [xra, 0.90], [xra - 0.35, 0.88],
    [xr + 0.05, 0.86], [xr, 0.80],
  ];
  const bottom = [
    [xn - 0.03, rh + 0.03], [xn - 0.30, rh + 0.02],
    ...arch(xra, r, Ra, rh + 0.02),
    [xr + 0.45, rh + 0.05], [xr + 0.10, 0.30], [xr, 0.42],
  ];
  const planX = table([
    [xn, 0.30], [xn - 0.30, 0.45], [xf + 0.30, 0.58], [xf, 0.62], [xf - 0.50, 0.66],
    [0.0, 0.80], [-0.80, 0.95], [xra + 0.40, W2], [xra, W2], [xra - 0.40, W2 * 0.99], [xr, W2 * 0.95],
  ]);
  const tubHW = (x, y) => planX(x) * (1 - 0.06 * clamp01((y - 0.5) / 0.4));
  parts.push({ geo: loftSolid(T, top, bottom, tubHW, 4) });

  // Front fender pods: the tall, separate front wings of a modern LMH/LMDh,
  // each over one front wheel, outer face on the body width.
  const podHW = 0.22;
  const podTop = [
    [xn - 0.10, 0.30], [xn - 0.35, 0.55], [xf + 0.40, 0.78], [xf, 0.86],
    [xf - 0.45, 0.82], [xf - 0.85, 0.62], [xf - 1.15, 0.45],
  ];
  const podBot = [[xn - 0.10, rh + 0.06], [xn - 0.30, rh + 0.04], ...arch(xf, r, Ra, rh + 0.02), [xf - 1.15, rh + 0.04]];
  const podX = table([[xn - 0.12, 0.6], [xn - 0.40, 0.95], [xf, 1.0], [xf - 1.15, 0.9]]);
  for (const s of [-1, 1]) {
    parts.push({ geo: loftSolid(T, podTop, podBot, (x) => podHW * podX(x), 2), z: s * (W2 - podHW) });
  }

  // Canopy bubble: narrow, centred, running back into the engine cover.
  const canTop = [
    [1.10, 0.62], [0.95, 0.70], [0.15, roof - 0.02], [-0.15, roof], [-0.55, roof - 0.01],
    [-0.95, 0.93], [-1.50, 0.85],
  ];
  const canX = table([[1.10, 0.7], [0.80, 1.0], [-0.60, 1.0], [-1.50, 0.6]]);
  const canHW = (x, y) => canX(x) * lerp(0.58, 0.36, clamp01((y - 0.65) / (roof - 0.65)));
  parts.push({ geo: loftSolid(T, canTop, [[1.10, 0.55], [-1.50, 0.70]], canHW, 2) });

  // Mirrors on the inner tops of the fender pods.
  for (const s of [-1, 1]) parts.push({ geo: box(T, 0.10, 0.07, 0.14), x: xf - 0.75, y: 0.78, z: s * 0.74 });

  return { parts, xn, xr, deckY: (x) => polyY(top, x) };
}

// Rear wing (section + endplates + mounts), splitter, shark fin and the inner
// floor between the wheels (so the arches do not show daylight through the
// whole car). Shared by both classes; the JSON says which mount and fin.
function aeroParts(T, d, xn, xr, deckY) {
  const parts = [];
  const w = d.wing, top = w.height_top;
  const xTe = xr + w.setback, xLe = xTe + w.chord;
  const swan = w.mount === 'swan_neck';
  // A swan neck hooks OVER the wing, so its top is the overall height and the
  // wing sits a little lower; pylons hold the wing from below.
  const wingTop = swan ? top - 0.035 : top - 0.02;
  const sec = wingSection(xLe, w.chord, wingTop);
  const epT = 0.012 + (swan ? 0 : 0.004);                 // endplate thickness
  const spanIn = w.span / 2 - epT;                        // wing ends at the plates' inner faces
  parts.push({ geo: profileSolid(T, sec.outline, () => spanIn, 1) });

  // Endplates: GT3 small and square; Hypercar long and tall, reaching down
  // into the rear bodywork.
  // Pylon wings: the endplate tops are the overall height. The plates may
  // overhang the trailing edge a little, but never past the car's tail.
  const epTop = swan ? top - 0.01 : top, epBot = epTop - w.endplate_height;
  const epFront = xLe + (swan ? 0.05 : 0.25), epBack = Math.max(xr, xTe - 0.02);
  const ep = [[epFront, epBot], [epFront - 0.04, epTop - 0.02], [epFront - 0.10, epTop], [epBack, epTop], [epBack, epBot]];
  for (const s of [-1, 1]) parts.push({ geo: plate(T, ep, epT), z: s * (w.span / 2 - epT / 2) });

  if (swan) {
    // Swan neck: rises from the deck ahead of the wing, arches over the
    // leading edge and lands on the upper surface (the GT3 signature).
    const xLand = xLe - 0.30 * w.chord;
    const land = [xLand, polyY(sec.upper, xLand) - 0.012];
    const halfT = 0.02;
    let ctrl = [
      [xLe + 0.12, deckY(xLe + 0.12) - 0.05], [xLe + 0.10, wingTop - 0.12],
      [xLe + 0.02, top - halfT], land,
    ];
    let pts = strip(T, ctrl, halfT);
    // The spline over- or undershoots its control points; move the hook so
    // the overall height is exactly height_top.
    for (let k = 0; k < 4; k++) {
      const over = Math.max(...pts.map((p) => p[1])) - top;
      if (Math.abs(over) <= 1e-4) break;
      ctrl = ctrl.map((p, i) => (i === 1 || i === 2 ? [p[0], p[1] - over] : p));
      pts = strip(T, ctrl, halfT);
    }
    for (const s of [-1, 1]) parts.push({ geo: plate(T, pts, 0.024), z: s * 0.32 });
  } else {
    // Two pylons from the deck up to the wing's underside.
    const yTop = sec.bottom + 0.03;
    const py = [[xLe + 0.10, deckY(xLe + 0.10) - 0.06], [xLe - 0.02, yTop], [xLe - 0.26, yTop], [xLe - 0.22, deckY(xLe - 0.22) - 0.06]];
    for (const s of [-1, 1]) parts.push({ geo: plate(T, py, 0.02), z: s * 0.22 });
  }

  if (d.fin) {
    // Shark fin on the centre line, from the roof back to the wing.
    const roof = d.roof_height ?? d.height - 0.05;
    const fin = [[-0.45, 0.90], [-0.52, roof - 0.01], [xLe + 0.04, wingTop - 0.05], [xLe + 0.04, 0.82], [-0.45, 0.82]];
    parts.push({ geo: plate(T, fin, 0.014) });
  }

  // Splitter: a thin plate under the nose, its lip is the car's front end.
  const sp = d.splitter;
  parts.push({ geo: box(T, sp.depth, 0.018, sp.width), x: xn - sp.depth / 2, y: d.ride_height - 0.011 });

  // Inner floor between the inner tyre faces, so the arches look closed.
  const inner = Math.min(d.track_front - d.tyre_width_front, d.track_rear - d.tyre_width_rear) / 2 - 0.02;
  const fy = d.ride_height, ty = 0.62;
  parts.push({ geo: box(T, d.wheelbase + 0.4, ty - fy, inner * 2), y: (fy + ty) / 2 });
  return parts;
}

function makeWheel(T, d, front) {
  const r = d.tyre_diameter / 2;
  const w = front ? d.tyre_width_front : d.tyre_width_rear;
  // One material per wheel, so each can be tinted on its own (track limits).
  const mat = new T.MeshStandardMaterial({ color: '#1b1f27', roughness: 0.9, metalness: 0 });
  const wheel = new T.Group();
  const tyre = new T.CylinderGeometry(r, r, w, 24, 1);
  tyre.rotateX(Math.PI / 2);                              // axle along Z (left-right)
  // Rim face: a slightly wider disc so the wheel reads as a wheel, not a drum.
  const rim = new T.CylinderGeometry(r * 0.64, r * 0.64, w + 0.004, 16, 1);
  rim.rotateX(Math.PI / 2);
  wheel.add(new T.Mesh(tyre, mat));
  wheel.add(new T.Mesh(rim, mat));
  wheel.userData.material = mat;
  return wheel;
}

// ---- public builder ---------------------------------------------------------

// Build one car. Returns
//   group   - add this to the scene; place/rotate it like the chevron rig
//   body    - the shell (body, wing, splitter, fin), one shared material
//   wheels  - {fl, fr, rl, rr}: Object3D at each hub; rotate about own Y to steer
//   outline - LineSegments of the body's creases (EdgesGeometry), one material,
//             for the track-limit tint
export function buildCar(THREE, cls, dims) {
  const T = THREE, d = dims;
  const shell = cls === 'Hypercar' ? hypercarParts(T, d) : gt3Parts(T, d);
  const parts = shell.parts.concat(aeroParts(T, d, shell.xn, shell.xr, shell.deckY));

  const group = new T.Group();
  group.name = `car:${cls}`;
  const body = new T.Group();
  body.name = 'body';
  const bodyMat = new T.MeshStandardMaterial({ color: '#d8dde6', roughness: 0.6, metalness: 0.1 });
  for (const p of parts) {
    const m = new T.Mesh(p.geo, bodyMat);
    m.position.set(p.x || 0, p.y || 0, p.z || 0);
    body.add(m);
  }
  body.userData.material = bodyMat;
  group.add(body);

  const r = d.tyre_diameter / 2, hx = d.wheelbase / 2;
  const wheels = {
    fl: makeWheel(T, d, true), fr: makeWheel(T, d, true),
    rl: makeWheel(T, d, false), rr: makeWheel(T, d, false),
  };
  // Right is +Z, so the right wheels are at +track/2.
  wheels.fl.position.set(hx, r, -d.track_front / 2);
  wheels.fr.position.set(hx, r, d.track_front / 2);
  wheels.rl.position.set(-hx, r, -d.track_rear / 2);
  wheels.rr.position.set(-hx, r, d.track_rear / 2);
  for (const k of ['fl', 'fr', 'rl', 'rr']) {
    wheels[k].name = `wheel:${k}`;
    group.add(wheels[k]);
  }

  // Outline: every body part's crease edges (30 degrees) merged into one
  // LineSegments, so one material colour tints the whole car.
  body.updateMatrixWorld(true);
  const segs = [];
  let n = 0;
  for (const m of body.children) {
    const e = new T.EdgesGeometry(m.geometry, 30);
    e.applyMatrix4(m.matrix);
    segs.push(e.attributes.position.array);
    n += e.attributes.position.array.length;
    e.dispose();
  }
  const arr = new Float32Array(n);
  let o = 0;
  for (const a of segs) { arr.set(a, o); o += a.length; }
  const og = new T.BufferGeometry();
  og.setAttribute('position', new T.BufferAttribute(arr, 3));
  const outline = new T.LineSegments(og, new T.LineBasicMaterial({ color: '#ffffff' }));
  outline.name = 'outline';
  group.add(outline);

  return { group, body, wheels, outline };
}
