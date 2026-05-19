/**
 * Aether Core Dashboard — main entry.
 *
 * - Three.js Scene + camera + OrbitControls
 * - Oda wireframe + raf wireframe + voxel grid
 * - WebSocket /ws/predict → 50 ms tick'te InstancedMesh güncellemesi
 * - HUD: ws-status, FPS, inference_ms, tick_count, slot durumu
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  createVoxelMesh, updateVoxels,
  createShelfWireframe, createRoomWireframe,
} from './voxel_renderer.js';


// ── Scene + camera + renderer ──────────────────────────────

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);
scene.fog = new THREE.Fog(0x0a0a0a, 20, 50);

const camera = new THREE.PerspectiveCamera(
  60, window.innerWidth / window.innerHeight, 0.1, 100,
);
camera.position.set(8, 5, 8);
camera.lookAt(5, 1.5, -2.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('app').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(5, 1.5, -2.5);
controls.update();
controls.enableDamping = true;
controls.dampingFactor = 0.05;


// ── Aydınlatma ─────────────────────────────────────────────

const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);

const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(10, 10, 5);
scene.add(dir);


// ── Sahne objesi: oda + raf + voxel ────────────────────────

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
    if (reconnectTimer) {
      clearInterval(reconnectTimer);
      reconnectTimer = null;
    }
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
    if (!reconnectTimer) {
      reconnectTimer = setInterval(connectWS, 2000);
    }
  };

  ws.onerror = (err) => {
    console.error('[WS] error:', err);
    setHUD('ws-status', 'error', 'err');
  };
}

function setHUD(elemId, text, cls = '') {
  const el = document.getElementById(elemId);
  if (!el) return;
  el.textContent = text;
  el.className = cls;
}


// ── Paket işleme ───────────────────────────────────────────

function handlePacket(packet) {
  const now = performance.now();
  const dt = now - lastPacketTime;
  lastPacketTime = now;
  packetCount++;

  // FPS hesabı (her saniye)
  if (packetCount % 20 === 0) {
    fps = Math.round(1000 / dt);
  }
  setHUD('fps', fps, 'value');

  // Telemetry
  const tel = packet.telemetry || {};
  setHUD('inference-ms', (tel.inference_ms || 0).toFixed(1), 'value');
  setHUD('tick-count', tel.tick_count || 0, 'value');

  // Slot durumu
  const detection = packet.detection_mask || [];
  const materials = packet.materials || [];
  const MATERIAL_NAMES = ['Metal', 'Plastik', 'Ahşap', 'Karton'];
  let html = '';
  for (let s = 0; s < 6; s++) {
    const occ = detection[s] ? '●' : '○';
    const mat = detection[s] ? (MATERIAL_NAMES[materials[s]] || '?') : 'boş';
    const cls = detection[s] ? 'value' : 'label';
    html += `<div>S${s + 1}: <span class="${cls}">${occ} ${mat}</span></div>`;
  }
  const slotStatus = document.getElementById('slot-status');
  if (slotStatus) slotStatus.innerHTML = html;

  // Voxel mesh güncelle
  updateVoxels(voxelMesh, packet);
}


// ── Render loop ───────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// Window resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Başlat
connectWS();
animate();

console.log('[Aether Core Dashboard] başlatıldı');
