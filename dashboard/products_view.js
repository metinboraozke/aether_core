/**
 * Products View — C Paketi 2026-06-03.
 *
 * /api/products fetch → 16 ürün kartı grid
 * Son 50 okuma (memory-only rolling buffer) tablo
 * Her okumada: timestamp, slot, id, name, db_mat, model_mat, match, hamming
 */

const HISTORY_MAX = 50;
let history = [];               // rolling array
let catalog = [];               // /api/products sonucu
let readCountById = new Map();  // ID → toplam okuma sayısı (session)


// ── Init: catalog fetch + grid render ─────────────────────

export async function initProductsView() {
  try {
    const res = await fetch('/api/products');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    catalog = data.products || [];
    const metaEl = document.getElementById('catalog-meta');
    if (metaEl) {
      metaEl.textContent = `${data.count || 0} ürün yüklendi · ${
        Object.keys(data.material_palette || {}).length
      } materyal sınıfı`;
    }
    renderCatalog();
  } catch (err) {
    console.warn('[products] catalog fetch fail:', err);
    const metaEl = document.getElementById('catalog-meta');
    if (metaEl) metaEl.innerHTML =
      `<span style="color:#f44336;">Katalog yüklenemedi: ${err.message}</span>`;
  }
}

function renderCatalog() {
  const grid = document.getElementById('products-grid');
  if (!grid) return;
  grid.innerHTML = '';
  catalog.forEach(p => {
    const count = readCountById.get(p.id) || 0;
    const card = document.createElement('div');
    card.className = 'product-card';
    card.dataset.id = p.id;
    card.innerHTML = `
      <div>
        <span class="id-badge">ID#${p.id}</span>
        <span class="color-swatch" style="background:${p.color_hex || '#404040'}"></span>
        <span style="color:#888; font-size:11px;">${p.material || '?'}</span>
      </div>
      <div class="name">${p.name || '?'}</div>
      <div class="meta">${p.description || ''}</div>
      <div class="read-count" data-count-id="${p.id}">
        ${count > 0 ? `📊 ${count} okuma (session)` : '⏳ henüz okunmadı'}
      </div>
    `;
    grid.appendChild(card);
  });
}

function updateCardCount(id) {
  const el = document.querySelector(`[data-count-id="${id}"]`);
  if (!el) return;
  const count = readCountById.get(id) || 0;
  el.textContent = count > 0 ? `📊 ${count} okuma (session)` : '⏳ henüz okunmadı';
}


// ── Reading history (main.js'ten çağrılır) ────────────────

export function pushReading(reading) {
  history.unshift(reading);   // en yeni başa
  if (history.length > HISTORY_MAX) history.pop();
  // Counter güncelle
  if (reading.id !== undefined) {
    readCountById.set(reading.id, (readCountById.get(reading.id) || 0) + 1);
    updateCardCount(reading.id);
  }
  renderHistory();
}

function renderHistory() {
  // D Paketi (2026-06-03): Çift doğrulama kolonları (Model Mat. + Eşleşme) kaldırıldı.
  // Tek kaynak DB lookup → 6 kolon (Zaman / Slot / ID / Ürün / Materyal / Hamming).
  const tbody = document.getElementById('readings-tbody');
  if (!tbody) return;
  if (history.length === 0) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="6">Henüz okuma yok.</td></tr>';
    return;
  }
  const rows = history.map(r => {
    const dbMat = r.material_db || '?';
    const hamming = r.hamming_corrected
      ? '<span class="match-warn" title="Hamming 1-bit düzeltti">±1</span>'
      : '<span style="color:#666;">—</span>';
    return `<tr>
      <td>#${r.timestamp}</td>
      <td>S${(r.slot ?? -1) + 1}</td>
      <td><span class="id-badge">${r.id}</span></td>
      <td>${r.name || '?'}</td>
      <td>${dbMat}</td>
      <td>${hamming}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = rows;
}


// ── Debug helper ──────────────────────────────────────────

export function getStats() {
  return {
    history_count: history.length,
    catalog_count: catalog.length,
    unique_ids_read: readCountById.size,
    total_reads: [...readCountById.values()].reduce((a, b) => a + b, 0),
  };
}
