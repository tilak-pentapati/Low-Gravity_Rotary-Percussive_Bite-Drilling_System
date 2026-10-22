/**
 * Interactive Three.js digital twin (docs_src/assets/js/drill_kinematics.js)
 * Autonomous Low-Gravity Rotary-Percussive Bite-Drilling System
 *
 * Replays kinematic_keyframes.json (src/state_machine.py) on full_drill_rig.glb (src/cad_generator.py).
 * Colour encodes physics only:
 *   - ground layers: unconfined compressive strength (UCS), log10, cividis colormap
 *   - bit: temperature as a share of the way from its start temperature to the limit, grey -> red
 *   - controller state: Okabe-Ito colour-blind-safe categorical palette
 * Strip charts under the view are drawn on a 2D canvas from the same keyframes.
 *
 * Uses direct ESM URLs (https://esm.sh/three@0.164.1) so no <script type="importmap">
 * ordering conflict can occur with Quarto's head module scripts.
 */

import * as THREE from "https://esm.sh/three@0.164.1";
import { OrbitControls } from "https://esm.sh/three@0.164.1/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://esm.sh/three@0.164.1/examples/jsm/loaders/GLTFLoader.js";

(function initDrillDigitalTwin() {
  const container = document.getElementById("drill-canvas-container");
  if (!container) return;

  // ---------------------------------------------------------------------------
  // 1. STATE, COLOUR SCALES & ENVIRONMENT METADATA
  // ---------------------------------------------------------------------------
  let keyframesData = window.DRILL_KEYFRAMES_DATA || null;
  let currentEnv = "moon";
  let frameIndex = 0;
  let frameAccumulator = 0;
  let isPlaying = true;
  let playbackSpeed = 1; // 1x, 0.5x, 0.25x
  let tipDisplay = null; // smoothed bit-tip height (m, world Y)
  let stationDisplay = null; // smoothed carousel station (0 = blank under the port, s = cup s)
  let lastTravel = 0.0;
  let viewMode = "rig"; // 'rig' | 'bit'

  const ENV = {
    moon: { caption: "Moon, lunar south pole, g = 1.62 m/s²", ground: 0x9e9c95 },
    mars: { caption: "Mars, 68° N, g = 3.71 m/s²", ground: 0xb07e5e },
  };

  // matplotlib cividis sampled at 0, 0.1 ... 1.0
  const CIVIDIS = ["#00224e", "#083370", "#35456c", "#4f576c", "#666970", "#7d7c78",
                   "#948e77", "#aea371", "#c8b866", "#e5cf52", "#fee838"].map((h) => new THREE.Color(h));
  const UCS_LOG_MIN = -3; // 0.001 MPa
  const UCS_LOG_MAX = 2; // 100 MPa

  function ucsFraction(ucsMpa) {
    const l = Math.log10(Math.max(ucsMpa, 1e-6));
    return Math.min(1, Math.max(0, (l - UCS_LOG_MIN) / (UCS_LOG_MAX - UCS_LOG_MIN)));
  }
  function rampColor(stops, x) {
    const s = Math.min(1, Math.max(0, x)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(s));
    return stops[i].clone().lerp(stops[i + 1], s - i);
  }
  const ucsColor = (ucsMpa) => rampColor(CIVIDIS, ucsFraction(ucsMpa));
  const fmtUcs = (u) => (u >= 1 ? u.toFixed(0) : u.toPrecision(2));

  // Bit temperature: start temperature (steel grey) -> 50 % (bronze) -> limit (red); matches .twin-tbar
  const THERMAL = ["#8b939c", "#c99a5b", "#b42318"].map((h) => new THREE.Color(h));
  const thermalColor = (frac) => rampColor(THERMAL, frac);

  // Okabe & Ito (2008) colour-blind-safe palette
  const STATE_COLORS = {
    IDLE: "#999999",
    ACTIVE: "#E69F00",
    SEARCH: "#56B4E9",
    DRILL: "#009E73",
    RETRACTION: "#CC79A7",
    TRANSFER: "#0072B2",
  };
  const STATE_LABEL = {
    IDLE: "Idle", ACTIVE: "Warm-up", SEARCH: "Search", DRILL: "Drill", RETRACTION: "Retract", TRANSFER: "Transfer",
  };

  const envData = () => (keyframesData && keyframesData.environments[currentEnv]) || null;
  const framesOf = () => (envData() ? envData().frames : []);

  // ---------------------------------------------------------------------------
  // 2. SCENE, CAMERA, RENDERER & CONTROLS (drawing-sheet light theme)
  // ---------------------------------------------------------------------------
  const PAPER = 0xf3f1ec;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(PAPER);

  const width = Math.max(container.clientWidth, 300);
  const height = Math.max(container.clientHeight, 360);

  const camera = new THREE.PerspectiveCamera(36, width / height, 0.01, 40.0);
  const RIG_VIEW = { pos: new THREE.Vector3(2.45, 1.1, 3.95), target: new THREE.Vector3(-0.1, 0.3, -0.5) };
  const BIT_OFFSET = new THREE.Vector3(0.2, 0.08, 0.3); // follow-bit camera offset from the tip (m)
  const CUP_OFFSET = new THREE.Vector3(0.3, 0.12, 0.5); // sample carousel close-up, level with the cups
  let userMoved = false; // keep the reader's own orbit when the viewport resizes
  // Pull the default camera back on portrait viewports so the whole rig stays in frame
  function applyRigView() {
    const k = Math.pow(Math.max(1, 1.3 / (camera.aspect || 1)), 0.8);
    camera.position.copy(RIG_VIEW.target).addScaledVector(RIG_VIEW.pos.clone().sub(RIG_VIEW.target), k);
    controls.target.copy(RIG_VIEW.target);
    controls.update();
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxDistance = 9.0;
  controls.minDistance = 0.08;
  controls.addEventListener("start", () => (userMoved = true));
  applyRigView();

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b2a6, 1.9));
  const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
  sunLight.position.set(3.2, 4.4, 2.4);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(2048, 2048);
  Object.assign(sunLight.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 14 });
  sunLight.shadow.bias = -0.0004;
  scene.add(sunLight);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.7);
  fillLight.position.set(-3.0, 1.5, -1.0);
  scene.add(fillLight);

  // ---------------------------------------------------------------------------
  // 3. GROUND AS A BLOCK DIAGRAM: strata coloured by UCS, depth scale, live borehole
  // ---------------------------------------------------------------------------
  // A 1 m deep slab under the lander. A wedge between two vertical section planes through the
  // borehole axis is removed in front of the drill, clear of the footpads (at about 17 and 163 deg),
  // so the auger and bit stay visible all the way down. The 146 deg plane faces the default camera
  // almost square on, so it carries the depth scale and the layer labels.
  const SLAB = { x0: -1.4, x1: 1.4, z0: -2.4, z1: 1.0 };
  const BLOCK_DEPTH = 1.0;
  const HOLE_R = 0.0105; // 21 mm gauge borehole
  const CUT = [(20 * Math.PI) / 180, (146 * Math.PI) / 180]; // section planes, angle from +x towards +z
  const groundGroup = new THREE.Group();
  scene.add(groundGroup);
  const holeStrips = [];

  function disposeTree(obj) {
    obj.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      }
    });
  }

  // Distance from the axis to the slab edge along the ray at angle phi (x = r cos phi, z = r sin phi)
  function rayToEdge(phi) {
    const c = Math.cos(phi);
    const s = Math.sin(phi);
    const hits = [];
    if (c > 1e-9) hits.push(SLAB.x1 / c);
    if (c < -1e-9) hits.push(SLAB.x0 / c);
    if (s > 1e-9) hits.push(SLAB.z1 / s);
    if (s < -1e-9) hits.push(SLAB.z0 / s);
    return Math.min(...hits);
  }

  // Section face painted on a 2D canvas: layer bands, boundaries, 10 cm depth scale and layer labels
  function sectionTexture(strata, faceLen, labelled, axisLeft) {
    const pxPerM = 1200;
    const cw = Math.round(faceLen * pxPerM);
    const ch = Math.round(BLOCK_DEPTH * pxPerM);
    const cv = document.createElement("canvas");
    cv.width = cw;
    cv.height = ch;
    const g = cv.getContext("2d");
    const font = (w, px) => `${w} ${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    strata.forEach((s) => {
      g.fillStyle = "#" + ucsColor(s.ucs_mpa).getHexString();
      g.fillRect(0, s.z0 * pxPerM, cw, (s.z1 - s.z0) * pxPerM);
    });
    g.fillStyle = "rgba(15,15,15,0.9)";
    strata.forEach((s) => g.fillRect(0, s.z1 * pxPerM - 2, cw, 4));
    if (labelled) {
      // Depth scale on the outer edge, away from the axis
      const edge = axisLeft ? cw : 0;
      const dir = axisLeft ? -1 : 1;
      g.textBaseline = "middle";
      g.textAlign = axisLeft ? "right" : "left";
      g.font = font(600, 60);
      for (let cm = 0; cm <= 100; cm += 5) {
        const y = Math.min(ch - 3, Math.max(3, (cm / 100) * pxPerM));
        const len = cm % 10 === 0 ? 70 : 35;
        g.fillStyle = "#111";
        g.fillRect(Math.min(edge, edge + dir * len), y - 2.5, len, 5);
        if (cm % 20 === 0 && cm > 0 && cm < 100) {
          const tx = edge + dir * 88;
          const label = `${cm} cm`;
          const w = g.measureText(label).width;
          g.fillStyle = "rgba(255,255,255,0.9)";
          g.fillRect(axisLeft ? tx - w - 10 : tx - 10, y - 38, w + 20, 76);
          g.fillStyle = "#111";
          g.fillText(label, tx, y);
        }
      }
      // Layer name and UCS, set in from the axis
      g.textAlign = axisLeft ? "left" : "right";
      const lx = axisLeft ? 150 : cw - 150;
      strata.forEach((s) => {
        const yMid = 0.5 * (s.z0 + s.z1) * pxPerM;
        if ((s.z1 - s.z0) * pxPerM < 180) return;
        const lines = [s.name.replace(/\s*\(.*\)\s*$/, ""), `UCS ${fmtUcs(s.ucs_mpa)} MPa`];
        g.font = font(600, 66);
        const w = Math.max(...lines.map((t) => g.measureText(t).width));
        g.fillStyle = "rgba(255,255,255,0.88)";
        g.fillRect(axisLeft ? lx - 16 : lx - w - 16, yMid - 80, w + 32, 160);
        g.fillStyle = "#111";
        g.fillText(lines[0], lx, yMid - 34);
        g.font = font(400, 66);
        g.fillText(lines[1], lx, yMid + 38);
      });
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  function buildGround(envKey) {
    disposeTree(groundGroup);
    groundGroup.clear();
    holeStrips.length = 0;
    const strata = (keyframesData && keyframesData.environments[envKey].strata) || [];
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x1b1b1b, transparent: true, opacity: 0.75 });
    const [a1, a2] = CUT;
    const L1 = rayToEdge(a1);
    const L2 = rayToEdge(a2);

    // Plan outline of the slab with the wedge removed (shape x, y = world x, -z after rotateX(-pi/2) below)
    const p1 = [L1 * Math.cos(a1), L1 * Math.sin(a1)];
    const p2 = [L2 * Math.cos(a2), L2 * Math.sin(a2)];
    // Solid region runs counter-clockwise in angle from a2 round to a1 + 2 pi
    const unwrap = (ang) => {
      let u = ang;
      while (u <= a2) u += 2 * Math.PI;
      return u;
    };
    const corners = [[SLAB.x0, SLAB.z0], [SLAB.x1, SLAB.z0], [SLAB.x1, SLAB.z1], [SLAB.x0, SLAB.z1]]
      .filter(([x, z]) => unwrap(Math.atan2(z, x)) < a1 + 2 * Math.PI)
      .sort((u, v) => unwrap(Math.atan2(u[1], u[0])) - unwrap(Math.atan2(v[1], v[0])));
    const outline = [[0, 0], p2, ...corners, p1];
    const shape = new THREE.Shape(outline.map(([x, z]) => new THREE.Vector2(x, -z)));

    const groundMat = new THREE.MeshStandardMaterial({ color: ENV[envKey].ground, roughness: 1.0, metalness: 0.0 });
    strata.forEach((s, li) => {
      const h = s.z1 - s.z0;
      const geo = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false, curveSegments: 1 });
      geo.rotateX(-Math.PI / 2); // extrusion now runs along +y, from 0 to h
      const wall = new THREE.MeshLambertMaterial({ color: ucsColor(s.ucs_mpa) });
      // ExtrudeGeometry groups: 0 = caps, 1 = side walls; the top cap of the first layer is the ground surface
      const mesh = new THREE.Mesh(geo, [li === 0 ? groundMat : wall, wall]);
      mesh.position.y = -s.z1;
      mesh.receiveShadow = li === 0;
      groundGroup.add(mesh);
      const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), edgeMat);
      lines.position.y = mesh.position.y;
      groundGroup.add(lines);
    });

    // Section faces facing into the wedge: a1 plane (axis at its left) and a2 plane (axis at its right, labelled)
    const faces = [
      { phi: a1, len: L1, rotY: -a1, axisLeft: true, normal: new THREE.Vector3(-Math.sin(a1), 0, Math.cos(a1)) },
      { phi: a2, len: L2, rotY: Math.PI - a2, axisLeft: false, normal: new THREE.Vector3(Math.sin(a2), 0, -Math.cos(a2)) },
    ];
    const holeMat = new THREE.MeshBasicMaterial({ color: 0x1e1e1e, polygonOffset: true, polygonOffsetFactor: -2 });
    faces.forEach((fc, k) => {
      const dir = new THREE.Vector3(Math.cos(fc.phi), 0, Math.sin(fc.phi));
      const face = new THREE.Mesh(
        new THREE.PlaneGeometry(fc.len, BLOCK_DEPTH),
        // not tone-mapped, so the section shows the legend colours exactly
        new THREE.MeshBasicMaterial({ map: sectionTexture(strata, fc.len, k === 1, fc.axisLeft), toneMapped: false })
      );
      face.rotation.y = fc.rotY;
      face.position.copy(dir).multiplyScalar(fc.len / 2).addScaledVector(fc.normal, 0.0004);
      face.position.y = -BLOCK_DEPTH / 2;
      groundGroup.add(face);
      // Drilled borehole drawn as a dark slot along the axis edge of the face (scaled to hole_cm each frame)
      const strip = new THREE.Mesh(new THREE.PlaneGeometry(HOLE_R, 1), holeMat);
      strip.rotation.y = fc.rotY;
      strip.position.copy(dir).multiplyScalar(HOLE_R / 2).addScaledVector(fc.normal, 0.0009);
      strip.scale.y = 0.0001;
      groundGroup.add(strip);
      holeStrips.push(strip);
    });
  }

  // ---------------------------------------------------------------------------
  // 4. BITE SAMPLE: cuttings in the sampling flutes, brush stripping, chute flow, cup fill
  // ---------------------------------------------------------------------------
  // Kinematic constants exported by src/cad_generator.py (exports/data/cad_manifest.json)
  const KIN = {
    home_tip_m: 0.3,
    brush_axle_m: [0.0, 0.19, 0.0355],
    brush_radius_m: 0.029,
    transport_pitch_m: 0.039,
    sampling_pitch_m: 0.013,
    sampling_zone_from_tip_m: [0.026, 0.251],
  };
  // Sample carousel and chute (glTF metres) come from the manifest: KIN.carousel, KIN.chute
  const CHUTE = { a: new THREE.Vector3(), b: new THREE.Vector3() };
  function setChute() {
    const c = KIN.chute;
    if (!c) return;
    CHUTE.a.set(c.inlet_m[0], c.inlet_m[1] - 0.004, c.inlet_m[2] + 0.004); // on the trough floor at the inlet lip
    CHUTE.b.set(c.outlet_m[0], c.outlet_m[1], c.outlet_m[2]); // outlet over the port
  }

  // Bulked cuttings are drawn as solid sleeves filling the flutes and as level slabs in the cups,
  // each slice coloured by the layer it was cut from (same UCS colours as the section)
  function slicePool(n, radius) {
    return Array.from({ length: n }, () => {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 28), new THREE.MeshLambertMaterial());
      m.visible = false;
      scene.add(m);
      return m;
    });
  }
  const fluteSlices = slicePool(5, 0.0099); // just inside the 20 mm flight diameter
  const cupSlices = []; // per cup (index = bite - 1): slices parented to Carousel_Node, built once the GLB is in

  function buildCupSlices(node) {
    const C = KIN.carousel;
    if (!C || !node) return;
    const th = (2 * Math.PI) / C.stations;
    const r = C.cup_id_m / 2 - 0.0004;
    for (let s = 1; s <= C.cups; s++) {
      const a = Math.PI / 2 - s * th; // CAD angle about +Z; glTF (x, z) = (R cos a, -R sin a)
      const pool = Array.from({ length: 4 }, () => {
        const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 32), new THREE.MeshLambertMaterial());
        m.position.set(C.pitch_radius_m * Math.cos(a), 0, -C.pitch_radius_m * Math.sin(a));
        m.visible = false;
        node.add(m);
        return m;
      });
      cupSlices.push(pool);
    }
  }

  const PARTICLE_COUNT = 80;
  const particlePositions = new Float32Array(PARTICLE_COUNT * 3);
  const particleState = Array.from({ length: PARTICLE_COUNT }, (_, i) => ({
    u: i / PARTICLE_COUNT, // progress: 0-0.6 sliding down the chute, 0.6-1 falling into the cup
    speed: 0.75 + Math.random() * 0.5,
    ox: (Math.random() - 0.5) * 0.012,
    oz: (Math.random() - 0.5) * 0.012,
  }));
  const particleGeo = new THREE.BufferGeometry();
  particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
  const particleMat = new THREE.PointsMaterial({ color: 0x888888, size: 0.0045 });
  const cuttingsParticles = new THREE.Points(particleGeo, particleMat);
  cuttingsParticles.visible = false;
  scene.add(cuttingsParticles);

  const sampleData = () => (keyframesData && keyframesData.sample) || null;
  let bites = []; // per bite: { start, end } hole depth in cm

  function buildBiteTable() {
    bites = [];
    framesOf().forEach((f) => {
      if (f.bite < 1) return;
      const b = bites[f.bite - 1] || (bites[f.bite - 1] = { start: f.hole_cm, end: f.hole_cm });
      b.end = Math.max(b.end, f.hole_cm);
    });
  }

  // Split a depth interval (cm) at the layer boundaries
  function layerSegments(d0, d1) {
    const out = [];
    ((envData() && envData().strata) || []).forEach((s) => {
      const a = Math.max(d0, s.z0 * 100);
      const b = Math.min(d1, s.z1 * 100);
      if (b - a > 1e-6) out.push({ a, b, ucs: s.ucs_mpa, name: s.name });
    });
    return out;
  }

  // Where this bite's cuttings are at frame f. The flutes fill from the bottom of the sampling
  // zone upward (earliest, shallowest cuttings highest); on the way out the brush strips
  // everything that has risen past it, shallowest first.
  function biteSample(f, tipY) {
    const S = sampleData();
    const b = bites[f.bite - 1];
    if (!S || !b || !["DRILL", "RETRACTION", "TRANSFER"].includes(f.state)) return null;
    const k = (S.hole_area_m2 * S.bulking_factor) / S.flute_area_m2; // m of flute filled per m drilled
    const top = b.start;
    const bottom = f.hole_cm;
    const fillLen = Math.min(S.sampling_length_m, ((bottom - top) / 100) * k);
    if (fillLen <= 0) return null;
    const s0 = KIN.sampling_zone_from_tip_m[0];
    const sCut = f.state === "DRILL" ? Infinity : KIN.brush_axle_m[1] - tipY - s0;
    const kept = Math.max(0, Math.min(fillLen, sCut));
    return { top, bottom, k, s0, fillLen, kept, stripped: 1 - kept / fillLen, cutDepth: bottom - (kept / k) * 100 };
  }

  function showSlices(pool, segs, place) {
    pool.forEach((m, i) => {
      const seg = segs[i];
      m.visible = !!seg;
      if (!seg) return;
      const [y0, y1] = place(seg);
      m.scale.y = Math.max(1e-5, y1 - y0);
      m.position.y = 0.5 * (y0 + y1);
      m.material.color.copy(ucsColor(seg.ucs));
    });
  }

  // Share of bite n's sample that has reached its own cup (cup n) at frame f
  function cupFraction(n, f, smp) {
    if (n < f.bite) return 1;
    if (n > f.bite || f.bite < 1) return 0;
    if (smp) return smp.stripped;
    return f.state === "IDLE" ? 1 : 0; // final stow after the last transfer
  }
  const cupHeightPerCm = (S) => (S.hole_area_m2 * S.bulking_factor * 0.01) / (Math.PI * (KIN.carousel.cup_id_m / 2) ** 2);

  function updateSample(f, tipY, dtSec) {
    const S = sampleData();
    const smp = biteSample(f, tipY);
    // Cuttings still in the flutes (deep part of the bite, below the brush)
    const kept = smp ? layerSegments(smp.cutDepth, smp.bottom) : [];
    showSlices(fluteSlices, kept, (seg) => [
      tipY + smp.s0 + ((smp.bottom - seg.b) / 100) * smp.k,
      tipY + smp.s0 + ((smp.bottom - seg.a) / 100) * smp.k,
    ]);
    fluteSlices.forEach((m) => {
      m.position.x = 0;
      m.position.z = 0;
    });

    // Each bite in its own cup: level fill V / (pi r^2), shallowest material at the bottom
    const C = KIN.carousel;
    const fracs = bites.map((_, i) => cupFraction(i + 1, f, smp));
    let level = C ? C.cup_floor_m : 0;
    if (S && C) {
      const hPerCm = cupHeightPerCm(S);
      cupSlices.forEach((pool, i) => {
        const b = bites[i];
        const dEnd = b ? b.start + fracs[i] * (b.end - b.start) : 0;
        showSlices(pool, b ? layerSegments(b.start, dEnd) : [], (seg) => [
          C.cup_floor_m + (seg.a - b.start) * hPerCm,
          C.cup_floor_m + (seg.b - b.start) * hPerCm,
        ]);
        if (b && i + 1 === f.bite) level = C.cup_floor_m + (dEnd - b.start) * hPerCm;
      });
    }

    // Stream down the chute while the brush is actually removing cuttings
    const flowing = !!smp && smp.stripped > 0 && smp.kept > 0;
    cuttingsParticles.visible = flowing;
    if (flowing) {
      const src = layerSegments(Math.max(smp.top, smp.cutDepth - 0.5), smp.cutDepth)[0];
      if (src) particleMat.color.copy(ucsColor(src.ucs));
      const dropTop = CHUTE.b.y;
      particleState.forEach((p, i) => {
        p.u = (p.u + dtSec * p.speed * Math.min(playbackSpeed, 1) * 0.9) % 1;
        let x;
        let y;
        let z;
        if (p.u < 0.6) {
          const v = p.u / 0.6;
          x = CHUTE.a.x + (CHUTE.b.x - CHUTE.a.x) * v + p.ox * 0.7;
          y = CHUTE.a.y + (CHUTE.b.y - CHUTE.a.y) * v + 0.002;
          z = CHUTE.a.z + (CHUTE.b.z - CHUTE.a.z) * v;
        } else {
          const v = (p.u - 0.6) / 0.4;
          x = CHUTE.b.x + p.ox;
          y = dropTop - (dropTop - level) * v * v;
          z = CHUTE.b.z + p.oz;
        }
        particlePositions[3 * i] = x;
        particlePositions[3 * i + 1] = y;
        particlePositions[3 * i + 2] = z;
      });
      particleGeo.getAttribute("position").needsUpdate = true;
    }
    updateTray(fracs, f.bite);
  }

  // Sample log in the side panel: one cell per bite, shallowest material on top
  let trayKey = "";
  function buildTray() {
    const tray = document.getElementById("twin-tray");
    if (!tray) return;
    tray.innerHTML = "";
    bites.forEach((b, i) => {
      const cell = document.createElement("div");
      cell.className = "twin-core";
      cell.title = `Bite ${i + 1}: ${b.start.toFixed(0)} to ${b.end.toFixed(0)} cm`;
      cell.innerHTML = `<div class="twin-core-box"></div><span>${i + 1}</span>`;
      tray.appendChild(cell);
    });
    trayKey = "";
  }

  function updateTray(fracs, current) {
    const tray = document.getElementById("twin-tray");
    const S = sampleData();
    const C = KIN.carousel;
    if (!tray || !S || !C) return;
    const key = `${current}:${fracs.map((x) => x.toFixed(3)).join(",")}`;
    if (key === trayKey) return;
    trayKey = key;
    let volume = 0;
    let sealed = 0;
    tray.querySelectorAll(".twin-core").forEach((cell, i) => {
      const b = bites[i];
      const dEnd = b.start + fracs[i] * (b.end - b.start);
      volume += ((dEnd - b.start) / 100) * S.hole_area_m2 * S.bulking_factor;
      if (fracs[i] >= 1 && i + 1 !== current) sealed += 1;
      const span = b.end - b.start;
      cell.classList.toggle("is-current", i + 1 === current);
      cell.querySelector(".twin-core-box").innerHTML = layerSegments(b.start, dEnd)
        .map((s) => `<i style="height:${(((s.b - s.a) / span) * 100).toFixed(2)}%;background:#${ucsColor(s.ucs).getHexString()}"></i>`)
        .join("");
    });
    const note = document.getElementById("twin-tray-note");
    if (note) {
      note.textContent =
        `${(volume * 1e6).toFixed(0)} cm³ of bulked cuttings collected. Each bite has its own cup ` +
        `(Ø${(C.cup_id_m * 1e3).toFixed(0)} mm, ${(S.cup_volume_m3 * 1e6).toFixed(0)} cm³) on a ${C.stations}-station carousel ` +
        "under the mast foot plate; only the cup under the chute port is open, the rest seal against the plate.";
    }
  }


  // ---------------------------------------------------------------------------
  // 5. LOAD full_drill_rig.glb: named kinematic nodes, outline edges, bit materials
  // ---------------------------------------------------------------------------
  const rigRoot = new THREE.Group();
  scene.add(rigRoot);

  const rigNodes = {
    Carriage_Node: null,
    Auger_Node: null,
    Hammer_Cam_Node: null,
    Cam_Ring_Node: null,
    Brush_Gear_Node: null,
    Carousel_Node: null,
    Solar_Wing_Port_Az_Node: null,
    Solar_Wing_Port_Elev_Node: null,
    Solar_Wing_Starboard_Az_Node: null,
    Solar_Wing_Starboard_Elev_Node: null,
  };
  const baseY = { Carriage_Node: 0, Auger_Node: 0, Hammer_Cam_Node: 0, Cam_Ring_Node: 0 };
  const BIT_PARTS = ["Bit_Crown_Body", "WC_Insert", "Bit_Sub"];
  const bitMaterials = []; // { mat, base: Color }

  function showLoadError() {
    const msg = document.createElement("div");
    msg.className = "hud-load-error";
    msg.innerHTML =
      "<div><strong>The 3D model could not be loaded.</strong><br/>Browsers block <code>.glb</code> files opened via " +
      "<code>file://</code>. Serve the site instead: <code>make serve</code>, then open " +
      "<code>http://localhost:8000</code>.</div>";
    container.appendChild(msg);
  }

  function isBitPart(obj) {
    for (let o = obj; o && o !== rigRoot; o = o.parent) {
      if (BIT_PARTS.some((p) => o.name && o.name.startsWith(p))) return true;
    }
    return false;
  }

  function bindNamedNodes(rootObj) {
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x1b1b1b, transparent: true, opacity: 0.32 });
    const meshes = [];
    rootObj.traverse((child) => {
      if (child.isMesh) meshes.push(child);
      if (Object.prototype.hasOwnProperty.call(rigNodes, child.name)) rigNodes[child.name] = child;
    });
    meshes.forEach((mesh) => {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      // Feature edges give the shaded model the look of an engineering illustration
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 40), edgeMat));
      if (isBitPart(mesh)) {
        mesh.material = mesh.material.clone();
        bitMaterials.push({ mat: mesh.material, base: mesh.material.color.clone() });
      }
    });
    Object.keys(baseY).forEach((k) => {
      if (rigNodes[k]) baseY[k] = rigNodes[k].position.y;
    });
    // Cups are drawn translucent so the samples inside the carousel show (the foot plate covers them from above)
    meshes.forEach((mesh) => {
      if (!mesh.name.includes("Sample_Cup_")) return;
      mesh.material = mesh.material.clone();
      Object.assign(mesh.material, { transparent: true, opacity: 0.35, depthWrite: false });
      mesh.children.forEach((c) => (c.visible = false)); // no outline edges through the phantom walls
    });
    buildCupSlices(rigNodes.Carousel_Node);
  }

  new GLTFLoader().load(
    "assets/cad/full_drill_rig.glb",
    (gltf) => {
      rigRoot.add(gltf.scene);
      bindNamedNodes(gltf.scene);
      updateFrame(frameIndex, 0.016, true);
    },
    undefined,
    showLoadError
  );

  // ---------------------------------------------------------------------------
  // 6. DATA LOADING
  // ---------------------------------------------------------------------------
  async function ensureManifestLoaded() {
    try {
      const resp = await fetch("assets/data/cad_manifest.json");
      if (resp.ok) Object.assign(KIN, (await resp.json()).kinematics || {});
      setChute();
    } catch (err) {
      console.warn("Using built-in kinematic constants:", err);
    }
  }

  async function ensureKeyframesLoaded() {
    if (keyframesData && keyframesData.environments) return;
    try {
      const resp = await fetch("assets/data/kinematic_keyframes.json");
      if (resp.ok) keyframesData = await resp.json();
    } catch (err) {
      console.warn("Keyframes could not be loaded:", err);
    }
  }

  // ---------------------------------------------------------------------------
  // 7. STRIP CHARTS (2D canvas; static layer cached, cursor redrawn per frame)
  // ---------------------------------------------------------------------------
  const chartCanvas = document.getElementById("twin-strip");
  const chart = { base: null, w: 0, h: 0, dpr: 1, x0: 0, x1: 0, rows: [], tMax: 1 };

  function chartColors() {
    const cs = getComputedStyle(document.getElementById("twin") || document.body);
    const v = (n, fb) => (cs.getPropertyValue(n) || "").trim() || fb;
    return {
      ink: v("--tw-ink", "#1f2328"),
      muted: v("--tw-muted", "#5b6170"),
      rule: v("--tw-rule", "#d9d6cf"),
      soft: v("--tw-rule-soft", "#ebe8e2"),
      limit: v("--tw-limit", "#b42318"),
      font: getComputedStyle(document.body).fontFamily,
    };
  }

  function buildChartBase() {
    const env = envData();
    if (!chartCanvas || !env) return;
    const frames = env.frames;
    const rect = chartCanvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.max(280, Math.round(rect.width));
    const H = Math.max(240, Math.round(rect.height));
    chartCanvas.width = W * dpr;
    chartCanvas.height = H * dpr;
    const base = document.createElement("canvas");
    base.width = W * dpr;
    base.height = H * dpr;
    const g = base.getContext("2d");
    g.scale(dpr, dpr);
    const C = chartColors();
    const narrow = W < 560;
    const x0 = narrow ? 64 : 104;
    const x1 = W - 10;
    const tMax = frames[frames.length - 1].t;
    const X = (t) => x0 + (t / tMax) * (x1 - x0);
    const fsz = narrow ? 10.5 : 11.5;
    g.font = `${fsz}px ${C.font}`;
    g.textBaseline = "middle";

    // Layout: state ribbon, four rows, time axis
    const ribbonTop = 4;
    const ribbonH = 12;
    const axisH = 20;
    const gap = 9;
    const rowTop = ribbonTop + ribbonH + gap;
    const rowH = (H - rowTop - axisH - gap * 3) / 4;
    const rowY = (k) => rowTop + k * (rowH + gap);

    const crit = frames[0].crit_k;
    const t0k = frames[0].temp_k;
    const wobLim = frames[0].wob_max;
    const wobSet = env.wob_setpoint_n || null;
    const pMax = Math.max(...frames.map((f) => f.power_w));
    const rows = [
      { key: "depth", label: "Depth", short: "Depth", unit: "cm", lo: -30, hi: 100, invert: true },
      { key: "wob", label: "Weight on bit", short: "WOB", unit: "N", lo: 0, hi: Math.ceil((wobLim * 1.12) / 10) * 10 },
      { key: "power_w", label: "Electrical load", short: "Load", unit: "W", lo: 0, hi: Math.ceil((pMax * 1.12) / 20) * 20 },
      { key: "temp_k", label: "Bit temperature", short: "Bit T", unit: "K", lo: Math.floor((t0k - 8) / 10) * 10, hi: Math.ceil((crit + 8) / 10) * 10 },
    ].map((r, k) => {
      const top = rowY(k);
      const Y = (v) => {
        const a = (v - r.lo) / (r.hi - r.lo);
        return r.invert ? top + a * rowH : top + rowH - a * rowH;
      };
      return { ...r, top, Y };
    });

    // State ribbon
    for (let i = 0; i < frames.length - 1; i++) {
      g.fillStyle = STATE_COLORS[frames[i].state] || "#999";
      g.fillRect(X(frames[i].t), ribbonTop, Math.max(1, X(frames[i + 1].t) - X(frames[i].t) + 0.5), ribbonH);
    }
    g.fillStyle = C.muted;
    g.textAlign = "right";
    g.fillText("State", x0 - 8, ribbonTop + ribbonH / 2);

    // Row frames, labels and y ticks
    rows.forEach((r) => {
      g.fillStyle = "#fff";
      g.fillRect(x0, r.top, x1 - x0, rowH);
      g.textAlign = "right";
      g.fillStyle = C.ink;
      if (narrow) {
        g.fillText(r.short, x0 - 8, r.top + rowH / 2 - 7);
        g.fillStyle = C.muted;
        g.fillText(r.unit, x0 - 8, r.top + rowH / 2 + 7);
      } else {
        g.fillText(r.label, x0 - 30, r.top + rowH / 2 - 7);
        g.fillStyle = C.muted;
        g.fillText(r.unit, x0 - 30, r.top + rowH / 2 + 7);
      }
    });

    // Depth row: ground layers as background bands (same cividis scale as the 3D section)
    const rd = rows[0];
    env.strata.forEach((s) => {
      g.fillStyle = "#" + ucsColor(s.ucs_mpa).getHexString();
      g.globalAlpha = 0.55;
      g.fillRect(x0, rd.Y(s.z0 * 100), x1 - x0, rd.Y(s.z1 * 100) - rd.Y(s.z0 * 100));
      g.globalAlpha = 1;
    });

    // WOB row: percussion intervals hatched
    const rw = rows[1];
    const hatch = document.createElement("canvas");
    hatch.width = hatch.height = 6;
    const hg = hatch.getContext("2d");
    hg.strokeStyle = "rgba(31,35,40,0.35)";
    hg.lineWidth = 1;
    hg.beginPath();
    hg.moveTo(0, 6);
    hg.lineTo(6, 0);
    hg.stroke();
    g.fillStyle = g.createPattern(hatch, "repeat");
    for (let i = 0; i < frames.length - 1; i++) {
      if (frames[i].state === "DRILL" && frames[i].perc > 0.05) {
        g.fillRect(X(frames[i].t), rw.top, Math.max(1, X(frames[i + 1].t) - X(frames[i].t) + 0.5), rowH);
      }
    }

    // Y ticks and reference lines
    const tickRow = (r, vals) => {
      g.textAlign = "right";
      g.fillStyle = C.muted;
      vals.forEach((v) => {
        const y = r.Y(v);
        g.fillRect(x0 - 4, Math.round(y) - 0.5, 4, 1);
        // keep labels inside their own row so neighbouring rows never collide
        g.textBaseline = y - r.top < 4 ? "top" : r.top + rowH - y < 4 ? "bottom" : "middle";
        g.fillText(String(v), x0 - 6, y);
      });
      g.textBaseline = "middle";
    };
    if (!narrow) {
      tickRow(rd, [0, 50, 100]);
      tickRow(rw, [0, Math.round(rw.hi / 2), rw.hi]);
      tickRow(rows[2], [0, rows[2].hi / 2, rows[2].hi]);
      tickRow(rows[3], [rows[3].lo, crit]);
    }
    const refLine = (r, v, color, label, dash) => {
      const y = Math.round(r.Y(v)) + 0.5;
      g.save();
      g.strokeStyle = color;
      g.setLineDash(dash);
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x0, y);
      g.lineTo(x1, y);
      g.stroke();
      g.restore();
      if (label) {
        g.textAlign = "right";
        const w = g.measureText(label).width;
        g.fillStyle = "rgba(255,255,255,0.88)";
        g.fillRect(x1 - w - 8, y - 13, w + 6, 12);
        g.fillStyle = color;
        g.fillText(label, x1 - 4, y - 7);
      }
    };
    refLine(rd, 0, C.ink, null, []);
    refLine(rw, wobLim, C.limit, `lift-off limit ${wobLim.toFixed(0)} N`, [5, 3]);
    if (wobSet) refLine(rw, wobSet, C.muted, null, [2, 3]);
    refLine(rows[3], crit, C.limit, `limit ${crit.toFixed(1)} K`, [5, 3]);

    // Traces
    const trace = (r, get, color, lw, step) => {
      g.strokeStyle = color;
      g.lineWidth = lw;
      g.lineJoin = "round";
      g.beginPath();
      frames.forEach((f, i) => {
        const x = X(f.t);
        const y = r.Y(Math.min(r.hi, Math.max(r.lo, get(f))));
        if (i === 0) g.moveTo(x, y);
        else if (step) {
          g.lineTo(x, r.Y(Math.min(r.hi, Math.max(r.lo, get(frames[i - 1])))));
          g.lineTo(x, y);
        } else g.lineTo(x, y);
      });
      g.stroke();
    };
    g.save();
    g.setLineDash([3, 2]);
    trace(rd, (f) => f.hole_cm, "rgba(255,255,255,0.95)", 1.2, false);
    g.restore();
    trace(rd, (f) => f.z_cm, C.ink, 1.4, false);
    trace(rw, (f) => f.wob, C.ink, 1.3, true);
    trace(rows[2], (f) => f.power_w, C.ink, 1.1, true);
    // Bit temperature coloured with the same ramp as the 3D bit
    g.lineWidth = 1.8;
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1];
      const b = frames[i];
      g.strokeStyle = "#" + thermalColor((b.temp_k - t0k) / (crit - t0k)).getHexString();
      g.beginPath();
      g.moveTo(X(a.t), rows[3].Y(a.temp_k));
      g.lineTo(X(b.t), rows[3].Y(b.temp_k));
      g.stroke();
    }

    // In-row keys
    g.textAlign = "left";
    g.fillStyle = C.muted;
    const keyText = (r, s) => {
      const w = g.measureText(s).width;
      g.fillStyle = "rgba(255,255,255,0.85)";
      g.fillRect(x0 + 3, r.top + 2, w + 6, 13);
      g.fillStyle = C.muted;
      g.fillText(s, x0 + 6, r.top + 9);
    };
    if (!narrow) {
      keyText(rd, "solid: bit tip   dashed: hole bottom   bands: layer UCS");
      keyText(rw, wobSet ? `hatched: percussion on   dotted: set-point ${wobSet.toFixed(0)} N` : "hatched: percussion on");
    }

    // Row borders
    g.strokeStyle = C.rule;
    g.lineWidth = 1;
    rows.forEach((r) => g.strokeRect(x0 + 0.5, Math.round(r.top) + 0.5, x1 - x0 - 1, Math.round(rowH)));

    // Time axis (minutes)
    const tMin = tMax / 60;
    const minPx = narrow ? 44 : 60;
    const step = [1, 2, 5, 10, 15, 20, 30].find((s) => ((x1 - x0) / tMin) * s >= minPx) || 30;
    const yAx = rowY(3) + rowH;
    g.textAlign = "center";
    g.fillStyle = C.muted;
    for (let m = 0; m <= tMin + 1e-9; m += step) {
      const x = X(m * 60);
      g.fillRect(Math.round(x) - 0.5, yAx, 1, 4);
      g.fillText(String(m), x, yAx + 12);
    }
    g.textAlign = "right";
    g.fillText(narrow ? "min" : "time, min", x0 - 8, yAx + 12);

    Object.assign(chart, { base, w: W, h: H, dpr, x0, x1, rows, tMax, rowH, ribbonTop, crit, t0k });
  }

  function drawChartCursor(f) {
    if (!chartCanvas || !chart.base) return;
    const g = chartCanvas.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(chart.base, 0, 0);
    g.scale(chart.dpr, chart.dpr);
    const x = Math.round(chart.x0 + (f.t / chart.tMax) * (chart.x1 - chart.x0)) + 0.5;
    const C = chartColors();
    g.strokeStyle = C.ink;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(x, chart.ribbonTop);
    g.lineTo(x, chart.rows[3].top + chart.rowH);
    g.stroke();
    const vals = [f.z_cm, f.wob, f.power_w, f.temp_k];
    chart.rows.forEach((r, k) => {
      const v = Math.min(r.hi, Math.max(r.lo, vals[k]));
      g.beginPath();
      g.arc(x, r.Y(v), 3.2, 0, 2 * Math.PI);
      g.fillStyle = "#fff";
      g.fill();
      g.stroke();
    });
  }

  function frameAtTime(t) {
    const frames = framesOf();
    let lo = 0;
    let hi = frames.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].t <= t) lo = mid;
      else hi = mid;
    }
    return Math.abs(frames[hi].t - t) < Math.abs(frames[lo].t - t) ? hi : lo;
  }

  if (chartCanvas) {
    let dragging = false;
    const seek = (e) => {
      if (!chart.base) return;
      const rect = chartCanvas.getBoundingClientRect();
      const a = (e.clientX - rect.left - chart.x0) / (chart.x1 - chart.x0);
      seekTo(frameAtTime(Math.min(1, Math.max(0, a)) * chart.tMax));
    };
    chartCanvas.addEventListener("pointerdown", (e) => {
      dragging = true;
      chartCanvas.setPointerCapture(e.pointerId);
      seek(e);
    });
    chartCanvas.addEventListener("pointermove", (e) => dragging && seek(e));
    chartCanvas.addEventListener("pointerup", () => (dragging = false));
    chartCanvas.addEventListener("pointercancel", () => (dragging = false));
  }

  // ---------------------------------------------------------------------------
  // 8. CONTROLS
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const el = {
    state: $("hud-val-state"),
    bite: $("hud-val-bite"),
    depth: $("hud-val-depth"),
    wob: $("hud-val-wob"),
    rpm: $("hud-val-rpm"),
    torque: $("hud-val-torque"),
    rop: $("hud-val-rop"),
    temp: $("hud-val-temp"),
    power: $("hud-val-power"),
    solar: $("hud-val-solar"),
    layer: $("hud-val-layer"),
    swatch: $("twin-layer-swatch"),
    layerProps: $("twin-layer-props"),
    cbarMark: $("twin-cbar-mark"),
    tbarMark: $("twin-tbar-mark"),
    mode: $("twin-mode"),
    why: $("twin-mode-why"),
    fault: $("hud-fault-banner"),
    scrub: $("hud-timeline-scrubber"),
    play: $("hud-btn-play"),
    time: $("hud-time-label"),
    caption: $("twin-env-caption"),
  };

  function seekTo(i) {
    frameIndex = i;
    if (el.scrub) el.scrub.value = String(i);
    updateFrame(frameIndex, 0.016, true);
  }

  function setupScrubberRange() {
    if (!el.scrub || !envData()) return;
    el.scrub.max = String(Math.max(0, framesOf().length - 1));
    el.scrub.value = String(frameIndex);
  }

  if (el.play) {
    el.play.addEventListener("click", () => {
      isPlaying = !isPlaying;
      el.play.textContent = isPlaying ? "Pause" : "Play";
    });
  }
  if (el.scrub) el.scrub.addEventListener("input", (e) => seekTo(parseInt(e.target.value, 10) || 0));

  document.querySelectorAll(".hud-speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".hud-speed-btn").forEach((b) => b.classList.remove("is-on"));
      btn.classList.add("is-on");
      playbackSpeed = parseFloat(btn.getAttribute("data-speed") || "1");
    });
  });

  document.querySelectorAll("[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("is-on", b === btn));
      viewMode = btn.getAttribute("data-view");
      const tipY = tipDisplay === null ? KIN.home_tip_m : tipDisplay;
      if (viewMode === "bit") {
        controls.target.set(0, tipY, 0);
        camera.position.set(0, tipY, 0).add(BIT_OFFSET);
        controls.update();
      } else if (viewMode === "cup" && KIN.carousel) {
        const C = KIN.carousel;
        controls.target.set(0, 0.5 * (C.cup_floor_m + C.cup_rim_m), C.axis_m[2]);
        camera.position.copy(controls.target).add(CUP_OFFSET);
        controls.update();
      } else {
        applyRigView();
      }
      userMoved = false;
    });
  });

  const btnMoon = $("hud-env-moon");
  const btnMars = $("hud-env-mars");

  function switchEnvironment(targetEnv) {
    currentEnv = targetEnv;
    frameIndex = 0;
    buildGround(currentEnv);
    buildBiteTable();
    buildTray();
    [btnMoon, btnMars].forEach((b) => {
      if (!b) return;
      const on = b === (currentEnv === "moon" ? btnMoon : btnMars);
      b.classList.toggle("is-on", on);
      b.setAttribute("aria-pressed", String(on));
    });
    if (el.caption) el.caption.textContent = ENV[currentEnv].caption;
    setupScrubberRange();
    refreshBookmarkButtons();
    buildChartBase();
    updateFrame(frameIndex, 0.016, true);
  }
  if (btnMoon) btnMoon.addEventListener("click", () => switchEnvironment("moon"));
  if (btnMars) btnMars.addEventListener("click", () => switchEnvironment("mars"));

  function refreshBookmarkButtons() {
    const bm = (envData() && envData().bookmarks) || {};
    document.querySelectorAll(".hud-override-btn").forEach((btn) => {
      const ok = typeof bm[btn.getAttribute("data-bookmark")] === "number";
      btn.disabled = !ok;
      btn.title = ok ? "" : "No fault occurs in the nominal simulation; the off-nominal case is on the Autonomous Control page";
    });
  }
  document.querySelectorAll(".hud-override-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const bm = envData() && envData().bookmarks;
      const i = bm && bm[btn.getAttribute("data-bookmark")];
      if (typeof i === "number") seekTo(i);
    });
  });

  // ---------------------------------------------------------------------------
  // 9. PER-FRAME KINEMATICS AND READOUTS
  // ---------------------------------------------------------------------------
  const fmtMse = (m) => (m >= 10 ? m.toFixed(0) : m.toFixed(1));

  function modeText(f) {
    const ucs = fmtUcs(f.ucs);
    const fault = f.fault && f.fault !== "NONE" ? f.fault.replace(/_/g, " ").toLowerCase() : null;
    switch (f.state) {
      case "DRILL":
        if (f.perc > 0.05) {
          return [
            "Rotary-percussive",
            f.mse_alt === null || f.mse_alt === undefined
              ? `At ${f.wob.toFixed(0)} N the cutters alone cannot penetrate ${ucs} MPa ground, so the hammer supplies the breaking energy (MSE ${fmtMse(f.mse)} MJ/m³).`
              : `The hammer lowers the specific energy from ${fmtMse(f.mse_alt)} to ${fmtMse(f.mse)} MJ/m³ in ${ucs} MPa ground, so it stays on.`,
          ];
        }
        return [
          "Rotary only",
          f.mse_alt === null || f.mse_alt === undefined
            ? "Rotary cutting penetrates and the hammer offers no gain here."
            : `In ${ucs} MPa ground the hammer would raise the specific energy from ${fmtMse(f.mse)} to ${fmtMse(f.mse_alt)} MJ/m³, so it stays off.`,
        ];
      case "SEARCH":
        return ["Not cutting", "Feeding down fast to just above the hole bottom, then slowly until the load cell senses contact."];
      case "RETRACTION":
        return fault
          ? ["Fault recovery", `Tripped on ${fault}: lifting off the bottom without rotation.`]
          : ["Not cutting", "Pulling out without rotation, so the cuttings stay in the auger flutes."];
      case "TRANSFER":
        return ["Not cutting", "The auger turns slowly as it rises past the brush, which sweeps the sample into the cup."];
      case "ACTIVE":
        return ["Not cutting", "Heaters warm the drive train before the first bite."];
      default:
        return ["Not cutting", f.t < 60 ? "Health check before drilling." : "Drill stowed at the end of the mission."];
    }
  }

  function updateFrame(idx, dtSec, snap = false) {
    const env = envData();
    if (!env || !env.frames.length) return;
    const frames = env.frames;
    const f = frames[Math.min(idx, frames.length - 1)];
    const holeM = Math.max(0.0, f.hole_cm / 100.0);

    // Bit-tip height (m, world Y) from telemetry: z = depth below surface (negative = above)
    const tipTarget = -f.z_cm / 100.0;
    if (snap || tipDisplay === null) tipDisplay = tipTarget;
    else tipDisplay += (tipTarget - tipDisplay) * (1.0 - Math.exp(-dtSec * 5.0 * Math.min(playbackSpeed, 4)));
    const prevTravel = lastTravel;
    const verticalTravel = tipDisplay - KIN.home_tip_m;
    lastTravel = verticalTravel;
    const vAxial = dtSec > 0 ? (verticalTravel - prevTravel) / dtSec : 0.0;

    if (rigNodes.Carriage_Node) rigNodes.Carriage_Node.position.y = baseY.Carriage_Node + verticalTravel;
    const omegaRadS = (f.rpm * 2.0 * Math.PI) / 60.0;
    const spinScale = Math.min(playbackSpeed, 6);
    if (rigNodes.Auger_Node) {
      rigNodes.Auger_Node.position.y = baseY.Auger_Node + verticalTravel;
      // Right-hand auger turning clockwise seen from above (-Y rotation in the Y-up frame)
      rigNodes.Auger_Node.rotation.y -= omegaRadS * dtSec * spinScale;
    }

    // Hammer: turns with the spindle (splined), lifted by the cam and dropped at the blow rate
    const percussing = f.state === "DRILL" && f.perc > 0.05;
    if (rigNodes.Hammer_Cam_Node) {
      let hammerOsc = 0.0;
      if (percussing) {
        const phase = (performance.now() * 0.001 * (KIN.blow_hz || 26.0)) % 1.0;
        hammerOsc = (phase < 0.83 ? phase / 0.83 : 0.0) * (KIN.cam_lift_m || 0.006) * f.perc;
      }
      if (rigNodes.Auger_Node) rigNodes.Hammer_Cam_Node.rotation.y = rigNodes.Auger_Node.rotation.y;
      rigNodes.Hammer_Cam_Node.position.y = baseY.Hammer_Cam_Node + verticalTravel + hammerOsc;
    }
    if (rigNodes.Cam_Ring_Node) {
      if (percussing) rigNodes.Cam_Ring_Node.rotation.y -= 14.0 * dtSec; // shown slowed
      rigNodes.Cam_Ring_Node.position.y = baseY.Cam_Ring_Node + verticalTravel;
    }

    // Brush worm-wheel: pitch radius x brush speed = flight lead x auger speed (+ axial travel)
    const brushY = KIN.brush_axle_m[1];
    const flightsAtBrush = brushY > tipDisplay + 0.024 && brushY < tipDisplay + 1.2;
    const zone = KIN.sampling_zone_from_tip_m;
    const inSampling = brushY > tipDisplay + zone[0] && brushY < tipDisplay + zone[1];
    const lead = inSampling ? KIN.sampling_pitch_m : KIN.transport_pitch_m;
    const vSurface = (omegaRadS * lead) / (2.0 * Math.PI) + vAxial;
    if (rigNodes.Brush_Gear_Node && flightsAtBrush) {
      rigNodes.Brush_Gear_Node.rotation.x += (vSurface / KIN.brush_radius_m) * dtSec * spinScale;
    }

    // Sample carousel: index moves from the simulation (one station per bite, cycloidal)
    if (rigNodes.Carousel_Node && KIN.carousel) {
      const target = f.station || 0;
      if (snap || stationDisplay === null) stationDisplay = target;
      else stationDisplay += (target - stationDisplay) * (1.0 - Math.exp(-dtSec * 4.0 * Math.min(playbackSpeed, 4)));
      rigNodes.Carousel_Node.rotation.y = (stationDisplay * 2 * Math.PI) / KIN.carousel.stations;
    }

    // Two-axis sun-tracking wings (cells face -Y_cad = +Z_gltf at azimuth 180 deg)
    const azRad = ((180.0 - (f.sun_az || 180.0)) * Math.PI) / 180.0;
    const elRad = (Math.max(0.0, f.sun_el || 0.0) * Math.PI) / 180.0;
    ["Port", "Starboard"].forEach((side) => {
      const az = rigNodes[`Solar_Wing_${side}_Az_Node`];
      const elv = rigNodes[`Solar_Wing_${side}_Elev_Node`];
      if (az) az.rotation.y = azRad;
      if (elv) elv.rotation.x = -elRad;
    });

    // This bite's cuttings: in the flutes, down the chute, in the cup
    updateSample(f, tipDisplay, dtSec);

    // Borehole slot on the section faces
    const holeLen = Math.max(0.0001, holeM);
    holeStrips.forEach((s) => {
      s.scale.y = holeLen;
      s.position.y = -0.5 * holeLen;
    });

    // Bit tint: share of the way from the start temperature to the limit
    const t0k = frames[0].temp_k;
    const tFrac = Math.min(1, Math.max(0, (f.temp_k - t0k) / (f.crit_k - t0k)));
    const tint = thermalColor(tFrac);
    if (el.tbarMark) el.tbarMark.style.left = `${(tFrac * 100).toFixed(1)}%`;
    bitMaterials.forEach(({ mat, base }) => {
      mat.color.copy(base).lerp(tint, 0.85);
      if (mat.emissive) mat.emissive.copy(tint).multiplyScalar(0.35 * tFrac);
    });

    // Follow-bit camera: carry the orbit along with the tip
    if (viewMode === "bit") {
      const dy = tipDisplay - controls.target.y;
      controls.target.y += dy;
      camera.position.y += dy;
    }

    // Readouts
    if (el.state) {
      el.state.textContent = STATE_LABEL[f.state] || f.state;
      el.state.style.borderLeftColor = STATE_COLORS[f.state] || "#999";
    }
    if (el.bite) el.bite.textContent = f.bite > 0 ? `Bite ${f.bite} of 10` : "Before first bite";
    if (el.depth) {
      el.depth.innerHTML =
`${f.z_cm.toFixed(1)} cm <small>/ ${f.hole_cm.toFixed(1)}</small>`;
    }
    if (el.wob) el.wob.innerHTML = `${f.wob.toFixed(1)} N <small>/ ${f.wob_max.toFixed(0)}</small>`;
    if (el.rpm) el.rpm.textContent = `${f.rpm.toFixed(0)} rpm`;
    if (el.torque) el.torque.textContent = `${f.torque.toFixed(2)} N·m`;
    if (el.rop) el.rop.textContent = f.state === "DRILL" ? `${(f.rop || 0).toFixed(2)} mm/s` : "-";
    if (el.power) el.power.textContent = `${f.power_w.toFixed(0)} W`;
    if (el.temp) {
      el.temp.innerHTML = `${f.temp_k.toFixed(1)} K <small>/ ${f.crit_k.toFixed(1)}</small>`;
      el.temp.classList.toggle("twin-hot", f.temp_k >= f.crit_k - 5.0);
    }
    if (el.solar) el.solar.innerHTML = `${(f.solar_w || 0).toFixed(0)} W <small>/ ${(f.bat_wh || 0).toFixed(0)} Wh</small>`;
    if (el.time) {
      const mins = Math.floor(f.t / 60);
      const secs = Math.floor(f.t % 60);
      el.time.textContent = `T+${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }

    // Ground at the bit
    const layer = (env.strata || []).find((s) => s.name === f.layer);
    const ucs = typeof f.ucs === "number" ? f.ucs : layer ? layer.ucs_mpa : 0;
    if (el.layer) el.layer.textContent = f.layer;
    if (el.swatch) el.swatch.style.background = "#" + ucsColor(ucs).getHexString();
    if (el.cbarMark) el.cbarMark.style.left = `${(ucsFraction(ucs) * 100).toFixed(1)}%`;
    if (el.layerProps && layer) {
      el.layerProps.textContent = `UCS ${fmtUcs(ucs)} MPa, ice ${layer.ice_wt_pct} wt%, k ${layer.k_w_mk} W/(m·K)`;
    }

    // Drilling mode and the reason for it
    const [mode, why] = modeText(f);
    if (el.mode) el.mode.textContent = mode;
    if (el.why) el.why.textContent = why;

    if (el.fault) {
      const on = f.fault && f.fault !== "NONE";
      el.fault.hidden = !on;
      if (on) el.fault.textContent = `Fault recovery: ${f.fault.replace(/_/g, " ").toLowerCase()}, tripping out`;
    }

    drawChartCursor(f);
  }

  // ---------------------------------------------------------------------------
  // 10. ANIMATION LOOP & RESIZE
  // ---------------------------------------------------------------------------
  let lastTime = performance.now();

  function animate(now) {
    requestAnimationFrame(animate);
    const dtSec = Math.min(0.1, (now - lastTime) * 0.001);
    lastTime = now;
    const frames = framesOf();
    if (isPlaying && frames.length) {
      frameAccumulator += dtSec * 8.0 * playbackSpeed;
      if (frameAccumulator >= 1.0) {
        const steps = Math.floor(frameAccumulator);
        frameAccumulator -= steps;
        frameIndex = (frameIndex + steps) % frames.length;
        if (el.scrub) el.scrub.value = String(frameIndex);
      }
    }
    updateFrame(frameIndex, dtSec);
    controls.update();
    renderer.render(scene, camera);
  }

  // Follow the container, which stretches to the readout column's height
  const fitViewport = () => {
    const w = Math.max(container.clientWidth, 280);
    const h = Math.max(container.clientHeight, 300);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    if (viewMode === "rig" && !userMoved) applyRigView();
  };
  let chartW = 0;
  const fitChart = () => {
    const w = chartCanvas ? Math.round(chartCanvas.getBoundingClientRect().width) : 0;
    if (w && w !== chartW) {
      chartW = w;
      buildChartBase();
      const frames = framesOf();
      if (frames.length) drawChartCursor(frames[Math.min(frameIndex, frames.length - 1)]);
    }
  };
  window.addEventListener("resize", () => {
    fitViewport();
    fitChart();
  });
  if ("ResizeObserver" in window) {
    new ResizeObserver(fitViewport).observe(container);
    if (chartCanvas) new ResizeObserver(fitChart).observe(chartCanvas);
  }

  Promise.all([ensureManifestLoaded(), ensureKeyframesLoaded()]).then(() => {
    if (!envData()) return;
    switchEnvironment(currentEnv);
    requestAnimationFrame(animate);
  });
})();
