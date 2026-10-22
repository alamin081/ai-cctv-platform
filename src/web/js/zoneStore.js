/**
 * AI CCTV Platform - Frontend Zone Store
 * Single source of truth for zone definitions on the client side.
 * Fetches from GET /api/zones and notifies subscribers on change.
 */

const zoneStore = (() => {
  let _zones = [];        // Array of zone definition objects
  let _raw = {};           // Raw API response (with version, updated_at, etc.)
  let _listeners = [];     // Change callbacks
  let _loading = false;
  let _error = null;

  /**
   * Parse a hex color string "#RRGGBB" to an rgba fill string with low opacity.
   */
  function hexToFill(hex, alpha = 0.12) {
    if (!hex || hex.length !== 7) return `rgba(128, 128, 128, ${alpha})`;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  /**
   * Normalize a zone definition from the API into the shape used by
   * player.js (canvas), analytics.js (cards), and the floating HUD.
   */
  function normalize(z) {
    const color = z.color || '#00f0ff';
    return {
      id: z.id,
      name: z.name || z.id,
      color: color,
      fill: hexToFill(color, 0.12),
      polygon: z.polygon || [],
      assignment_mode: z.assignment_mode || 'center',
      enabled: z.enabled !== false,
    };
  }

  /**
   * Fetch zones from GET /api/zones and notify listeners.
   * Safe to call multiple times — deduplicates concurrent calls.
   */
  async function refresh() {
    if (_loading) return _zones;
    _loading = true;
    _error = null;
    try {
      const res = await fetch('/api/zones');
      if (!res.ok) throw new Error(`GET /api/zones returned ${res.status}`);
      const data = await res.json();
      _raw = data;
      const defs = data.definitions || [];
      _zones = defs.map(normalize);
      _notifyAll();
    } catch (e) {
      _error = e;
      console.error('[zoneStore] Failed to load zones:', e);
      // Don't clear existing zones — keep stale data so UI doesn't break
    } finally {
      _loading = false;
    }
    return _zones;
  }

  /**
   * Apply zones directly from a zones array (e.g. from session response)
   * without making an API call.
   */
  function setFromSession(definitions) {
    if (!Array.isArray(definitions)) return;
    _zones = definitions.map(normalize);
    _notifyAll();
  }

  /**
   * Subscribe to zone changes. Callback receives the zones array.
   * Returns an unsubscribe function.
   */
  function subscribe(fn) {
    _listeners.push(fn);
    return () => {
      _listeners = _listeners.filter(f => f !== fn);
    };
  }

  function _notifyAll() {
    for (const fn of _listeners) {
      try { fn(_zones); } catch (e) { console.error('[zoneStore] Listener error:', e); }
    }
  }

  /** Get current zones (synchronous). */
  function getZones() { return _zones; }

  /** Get only enabled zones. */
  function getEnabledZones() { return _zones.filter(z => z.enabled); }

  /** Get a zone by ID. */
  function getById(id) { return _zones.find(z => z.id === id) || null; }

  /** Get last error, if any. */
  function getError() { return _error; }

  /** Get raw API response. */
  function getRaw() { return _raw; }

  return {
    refresh,
    setFromSession,
    subscribe,
    getZones,
    getEnabledZones,
    getById,
    getError,
    getRaw,
    hexToFill,
  };
})();

// Make globally accessible
window.zoneStore = zoneStore;
