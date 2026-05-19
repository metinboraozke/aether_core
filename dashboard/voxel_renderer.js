/**
 * Voxel renderer — Bayesian TSDF görselleştirme.
 * PDF Modül 5 Adım 4 (README.md).
 *
 * Render kuralları:
 *   - InstancedMesh: 6 slot × 8×8×8 voxel = 3072 instance, tek draw call.
 *   - Opacity   = μ (Bayesian doluluk, sigmoid → [0, 1])
 *   - Color     = material (Metal: gümüş, Plastik: mavi, Ahşap: kahve, Karton: bej)
 *   - Glow      = σ² yüksekse (bulanık/belirsiz görünüm)
 */

import * as THREE from 'three';


// Material renkleri (model index 0..3 → hex)
const MATERIAL_COLORS = {
  0: 0xc0c0c0,  // Metal
  1: 0x4488ff,  // Plastik
  2: 0x884422,  // Ahşap
  3: 0xddccaa,  // Karton
};
const EMPTY_COLOR = 0x666666;

// scene.yaml slot_centers_m (classroom_default için, render referans)
const SLOT_CENTERS = [
  [4.75, 2.5, 1.83],
  [5.25, 2.5, 1.83],
  [4.75, 2.5, 1.50],
  [5.25, 2.5, 1.50],
  [4.75, 2.5, 1.17],
  [5.25, 2.5, 1.17],
];

const SLOT_SIZE = [0.5, 0.4, 0.33];      // m
const GRID_SIZE = [8, 8, 8];               // voxel sayısı/slot
const VOXEL_DIM = [
  SLOT_SIZE[0] / GRID_SIZE[0],
  SLOT_SIZE[1] / GRID_SIZE[1],
  SLOT_SIZE[2] / GRID_SIZE[2],
];


export function createVoxelMesh() {
  const N_SLOTS = SLOT_CENTERS.length;
  const [gx, gy, gz] = GRID_SIZE;
  const totalVoxels = N_SLOTS * gx * gy * gz;

  const geometry = new THREE.BoxGeometry(VOXEL_DIM[0], VOXEL_DIM[1], VOXEL_DIM[2]);
  const material = new THREE.MeshStandardMaterial({
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
    emissive: new THREE.Color(0x000000),
    emissiveIntensity: 0.3,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, totalVoxels);
  mesh.userData = { nSlots: N_SLOTS, gx, gy, gz };
  mesh.frustumCulled = false;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  // Renk attribute (InstancedMesh için)
  const colors = new Float32Array(totalVoxels * 3);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);

  return mesh;
}


export function updateVoxels(mesh, packet) {
  if (!packet || !mesh) return;
  const { nSlots, gx, gy, gz } = mesh.userData;
  const materials = packet.materials || [];
  const detection = packet.detection_mask || [];
  const uncertainty = packet.uncertainty || [];     // [6][8][8][8]

  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  const baseColor = new THREE.Color();

  let idx = 0;
  for (let s = 0; s < nSlots; s++) {
    const occupied = !!detection[s];
    const matIdx = (materials[s] !== undefined) ? materials[s] : 0;
    const baseHex = occupied ? (MATERIAL_COLORS[matIdx] ?? EMPTY_COLOR) : EMPTY_COLOR;
    baseColor.setHex(baseHex);

    const center = SLOT_CENTERS[s];

    for (let i = 0; i < gx; i++) {
      for (let j = 0; j < gy; j++) {
        for (let k = 0; k < gz; k++) {
          // μ değeri: detection ile binary proxy (eğer voxel-level mu varsa onu kullan)
          // Şu an mock'ta voxel-level mu yok, sadece uncertainty (σ²) var
          // Detection 1 ise tüm voksellere uniform doluluk göster
          const mu = occupied ? 1.0 : 0.0;
          const sigma2 = (uncertainty[s] && uncertainty[s][i] && uncertainty[s][i][j])
                          ? uncertainty[s][i][j][k]
                          : 0.0;

          // Voxel pozisyonu — slot merkezi + offset
          const x = center[0] + (i - gx / 2 + 0.5) * VOXEL_DIM[0];
          const y = center[1] + (j - gy / 2 + 0.5) * VOXEL_DIM[1];
          const z = center[2] + (k - gz / 2 + 0.5) * VOXEL_DIM[2];
          // mu < 0.05 → görünmez (scale 0)
          const visScale = (mu > 0.05) ? mu : 0.0;
          matrix.compose(
            new THREE.Vector3(x, z, -y),                      // Three.js y-up için swap
            new THREE.Quaternion(),
            new THREE.Vector3(visScale, visScale, visScale),
          );
          mesh.setMatrixAt(idx, matrix);

          // Renk: base + uncertainty glow
          color.copy(baseColor);
          if (sigma2 > 0.5) {
            // Glow: parlaklığı artır
            color.multiplyScalar(1.0 + Math.min(sigma2, 1.5));
          }
          mesh.setColorAt(idx, color);
          idx++;
        }
      }
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}


// Raf wireframe (referans için)
export function createShelfWireframe() {
  const group = new THREE.Group();
  const SHELF_CENTER = [5.0, 2.5, 1.5];
  const SHELF_SIZE = [1.0, 0.4, 1.0];

  // Raf dış kenarı
  const geo = new THREE.BoxGeometry(SHELF_SIZE[0], SHELF_SIZE[2], SHELF_SIZE[1]);
  const edges = new THREE.EdgesGeometry(geo);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x888888 });
  const shelf = new THREE.LineSegments(edges, lineMat);
  shelf.position.set(SHELF_CENTER[0], SHELF_CENTER[2], -SHELF_CENTER[1]);
  group.add(shelf);

  // 6 slot kenar
  for (const c of SLOT_CENTERS) {
    const sg = new THREE.BoxGeometry(SLOT_SIZE[0], SLOT_SIZE[2], SLOT_SIZE[1]);
    const se = new THREE.EdgesGeometry(sg);
    const sl = new THREE.LineSegments(se, new THREE.LineBasicMaterial({ color: 0x4caf50, transparent: true, opacity: 0.3 }));
    sl.position.set(c[0], c[2], -c[1]);
    group.add(sl);
  }
  return group;
}


// Oda kenarı
export function createRoomWireframe(width = 10, depth = 5, height = 3) {
  const geo = new THREE.BoxGeometry(width, height, depth);
  const edges = new THREE.EdgesGeometry(geo);
  const mat = new THREE.LineBasicMaterial({ color: 0x444444 });
  const box = new THREE.LineSegments(edges, mat);
  box.position.set(width / 2, height / 2, -depth / 2);
  return box;
}
