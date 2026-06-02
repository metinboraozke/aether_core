/**
 * Aether Core Dashboard — main entry (C Paketi 2026-06-03).
 *
 * Eklemeler:
 *   - Sol navbar + view switching (live / products)
 *   - Çift doğrulama slot UI (model rec vs DB lookup)
 *   - Voxel renkler DB color_hex'ten (model rec output bypass)
 *   - Son 50 okuma history (memory-only, page reload sıfır)
 *   - /api/products fetch ile katalog render
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  createVoxelMesh, updateVoxels,
  createShelfWireframe, createRoomWireframe,
} from './voxel_renderer.js';
import { initProductsView, pushReading } from './products_view.js';


// ── View switching ─────────────────────────────────────────

const VIEWS = ['live', 'products'];
function showView(name) {
  if (!VIEWS.includes(name)) return;
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.getElementById(`view-${name}`)?.classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === name);
  });
  if (name === 'live') {
    // Three.js renderer canvas boyut güncelle (CSS layout değişti)
    onResize();
  }
}
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => showView(btn.dataset.view));
});


// ── Three.js scene + camera + renderer ─────────────────────

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);
scene.fog = new THREE.Fog(0x0a0a0a, 20, 50);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
camera.position.set(8, 5, 8);
camera.lookAt(5, 1.5, -2.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('app').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(5, 1.5, -2.5);
controls.update();
controls.enableDamping = true;
controls.dampingFactor = 0.05;

function onResize() {
  const container = document.getElementById('app');
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w === 0 || h === 0) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);
setTimeout(onResize, 100);   // ilk render layout settle olduktan sonra


// ── Aydınlatma ─────────────────────────────────────────────

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(10, 10, 5);
scene.add(dir);

scene.add(createRoomWireframe(10, 5, 3));
scene.add(createShelfWireframe());

const voxelMesh = createVoxelMesh();
scene.add(voxelMesh);


// ── WebSocket client ───────────────────────────────────────

const WS_URL = `ws://${location.hostname || 'localhost'}:8000/ws/predict`;
let ws = null;
let reconnectTimer = null;
let lastPacketTime = performance.now();
let packetCount = 0;
let fps = 0;

function connectWS() {
  console.log(`[WS] connecting to ${WS_URL}`);
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('[WS] connected');
    setHUD('ws-status', 'connected', 'value');
    setSidebarStatus(true, 'bağlı');
    if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (e) => {
    try {
      const packet = JSON.parse(e.data);
      handlePacket(packet);
    } catch (err) {
      console.warn('[WS] invalid JSON:', err);
    }
  };

  ws.onclose = () => {
    console.log('[WS] closed, reconnecting in 2s...');
    setHUD('ws-status', 'closed (reconnecting)', 'warn');
    setSidebarStatus(false, 'kopuk');
    if (!reconnectTimer) reconnectTimer = setInterval(connectWS, 2000);
  };

  ws.onerror = (err) => {
    console.error('[WS] error:', err);
    setHUD('ws-status', 'error', 'err');
    setSidebarStatus(false, 'hata');
  };
}

function setHUD(elemId, text, cls = '') {
  const el = document.getElementById(elemId);
  if (!el) return;
  el.textContent = text;
  el.className = cls;
}

function setSidebarStatus(ok, label) {
  const dot = document.getElementById('sidebar-ws-dot');
  const txt = document.getElementById('sidebar-ws-text');
  if (dot) dot.className = 'dot ' + (ok ? 'ok' : 'warn');
  if (txt) txt.textContent = label;
}


// ── Paket işleme (Live + Products view'lere aktarım) ──────

const MATERIAL_NAMES_TR = {
  metal: 'METAL', plastik: 'PLASTİK', ahşap: 'AHŞAP', karton: 'KARTON',
  unknown: '?',
};

function handlePacket(packet) {
  const now = performance.now();
  const dt = now - lastPacketTime;
  lastPacketTime = now;
  packetCount++;

  if (packetCount % 20 === 0) fps = Math.round(1000 / dt);
  setHUD('fps', fps, 'value');

  const tel = packet.telemetry || {};
  setHUD('inference-ms', (tel.inference_ms || 0).toFixed(1), 'value');
  setHUD('tick-count', tel.tick_count || 0, 'value');

  // Sidebar live status update
  const sbTick = document.getElementById('sidebar-tick');
  const sbInf = document.getElementById('sidebar-inf');
  if (sbTick) sbTick.textContent = tel.tick_count || 0;
  if (sbInf) sbInf.textContent = (tel.inference_ms || 0).toFixed(1);

  // C Paketi: products field (DB lookup) varsa onu kullan, yoksa eski materials fallback
  const products = packet.products || [];
  const detection = packet.detection_mask || [];

  renderSlotStatus(detection, products);

  // Voxel mesh güncelle (color = DB'den products[].color_hex)
  updateVoxels(voxelMesh, packet);

  // Products view'ı için reading history (sadece dolu slot'lar, tek tick'te
  // birden fazla okuma olabilir)
  products.forEach(p => {
    if (!p.empty) pushReading({
      timestamp: tel.tick_count || 0,
      slot: p.slot,
      id: p.id,
      name: p.name,
      material_db: p.material_db,
      material_model: p.material_model,
      match: p.match,
      hamming_corrected: p.hamming_corrected,
    });
  });
}

function renderSlotStatus(detection, products) {
  let html = '';
  const productBySlot = {};
  products.forEach(p => { productBySlot[p.slot] = p; });

  for (let s = 0; s < 6; s++) {
    const p = productBySlot[s];
    const isEmpty = !detection[s] || (p && p.empty);
    if (isEmpty) {
      html += `<div class="slot-row"><span class="label">S${s + 1}:</span> <span style="color:#555;">○ boş</span></div>`;
      continue;
    }
    if (!p) {
      // Detection dolu ama product yok (DB devre dışı?)
      html += `<div class="slot-row"><span class="label">S${s + 1}:</span> ● dolu (DB yok)</div>`;
      continue;
    }
    const modelMat = MATERIAL_NAMES_TR[p.material_model] || '?';
    const dbMat = MATERIAL_NAMES_TR[p.material_db] || '?';
    const matchIcon = p.match
      ? '<span class="ok-tick">✓</span>'
      : '<span class="warn-tick">⚠</span>';
    const hammingFlag = p.hamming_corrected
      ? ' <span class="warn" title="Hamming 1-bit düzeltti">±1</span>'
      : '';
    html += `
      <div class="slot-row">
        <div><span class="label">S${s + 1}:</span> ● <span class="slot-name">${p.name}</span> <span class="slot-id">(ID#${p.id})</span>${hammingFlag}</div>
        <div class="dual">Model: ${modelMat} &nbsp;·&nbsp; DB: ${dbMat} ${matchIcon}</div>
      </div>`;
  }
  const el = document.getElementById('slot-status');
  if (el) el.innerHTML = html || '<span class="label">--</span>';
}


// ── Products view init + render loop ──────────────────────

initProductsView();   // /api/products fetch + katalog grid render


function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

connectWS();
animate();
console.log('[Aether Core Dashboard] başlatıldı (C paketi: DB lookup + navbar)');
