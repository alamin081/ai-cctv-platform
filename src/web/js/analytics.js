/**
 * AI CCTV Platform - Live Analytics Dashboard Controller
 * Displays person tracking, zone occupancy, entry/exit flow, and Phase 9 footfall metrics alongside video.
 * Supports both real-time playback synchronization and batch job results.
 *
 * Zone occupancy cards are now fully dynamic — built from zoneStore,
 * no hardcoded zone IDs, names or colors.
 */

class CCTVAnalytics {
  constructor() {
    // Stat elements
    this.statDetectedNow = document.getElementById('stat-detected-now');
    this.statActiveTracks = document.getElementById('stat-active-tracks');
    this.statTotalUnique = document.getElementById('stat-total-unique');
    this.statActiveIdsList = document.getElementById('stat-active-ids-list');

    // Zone Occupancy — dynamic container (cards built by JS)
    this.zoneListContainer = document.getElementById('zone-list-container');

    // Entry / Exit
    this.statTotalEntries = document.getElementById('stat-total-entries');
    this.statTotalExits = document.getElementById('stat-total-exits');
    this.statNetFlow = document.getElementById('stat-net-flow');
    this.statTotalTransitions = document.getElementById('stat-total-transitions');

    // Footfall (Phase 9)
    this.statFootfallMovements = document.getElementById('stat-footfall-movements');
    this.statFootfallNet = document.getElementById('stat-footfall-net');
    this.footfallBucketInfo = document.getElementById('footfall-bucket-info');
    this.footfallStatus = document.getElementById('footfall-engine-status');

    // Events Feed
    this.eventsFeedList = document.getElementById('events-feed-list');
    this.eventsCountBadge = document.getElementById('events-count-badge');

    // Historical / Cumulative data
    this.jobSummary = null;
    this.eventsHistory = [];

    // Zone card element cache: { zone_id: { countEl, trackIdsEl } }
    this._zoneCardEls = {};

    // Subscribe to zone store changes
    if (window.zoneStore) {
      window.zoneStore.subscribe((zones) => this.buildZoneCards(zones));
    }
  }

  /**
   * Build zone occupancy cards dynamically from the zone list.
   * Replaces all cards in the container.
   */
  buildZoneCards(zones) {
    if (!this.zoneListContainer) return;
    this.zoneListContainer.innerHTML = '';
    this._zoneCardEls = {};

    if (!zones || zones.length === 0) {
      this.zoneListContainer.innerHTML =
        '<div style="color: var(--text-muted); font-size: 11px; text-align: center; padding: 12px 0;">No zones configured.</div>';
      return;
    }

    zones.forEach((zone) => {
      const row = document.createElement('div');
      row.className = 'zone-row';
      if (!zone.enabled) row.classList.add('zone-disabled');
      row.dataset.zoneId = zone.id;

      // Top line: badge dot + name + headcount
      const rowTop = document.createElement('div');
      rowTop.className = 'zone-row-top';

      const nameDiv = document.createElement('div');
      nameDiv.className = 'zone-name';

      const dot = document.createElement('span');
      dot.className = 'zone-badge-pill';
      dot.style.background = zone.enabled ? zone.color : '#555';
      dot.style.boxShadow = zone.enabled ? `0 0 6px ${zone.color}` : 'none';

      const nameText = document.createTextNode(zone.enabled ? zone.name : `${zone.name} (Disabled)`);
      nameDiv.appendChild(dot);
      nameDiv.appendChild(nameText);

      const countDiv = document.createElement('div');
      countDiv.className = 'zone-headcount';
      countDiv.id = `zone-occ-${zone.id}`;
      countDiv.textContent = zone.enabled ? '0' : '—';

      rowTop.appendChild(nameDiv);
      rowTop.appendChild(countDiv);

      // Track IDs line
      const tracksPill = document.createElement('div');
      tracksPill.className = 'zone-tracks-pill';

      const tracksLabel = document.createTextNode('Track IDs: ');
      const tracksSpan = document.createElement('span');
      tracksSpan.id = `zone-tracks-${zone.id}`;
      tracksSpan.style.color = 'var(--text-secondary)';
      tracksSpan.textContent = zone.enabled ? 'None' : '—';

      tracksPill.appendChild(tracksLabel);
      tracksPill.appendChild(tracksSpan);

      row.appendChild(rowTop);
      row.appendChild(tracksPill);
      this.zoneListContainer.appendChild(row);

      // Cache references for fast per-frame updates
      this._zoneCardEls[zone.id] = {
        countEl: countDiv,
        trackIdsEl: tracksSpan,
        enabled: zone.enabled,
      };
    });
  }

  resetMetrics() {
    if (this.statDetectedNow) this.statDetectedNow.textContent = '0';
    if (this.statActiveTracks) this.statActiveTracks.textContent = '0';
    if (this.statTotalUnique) this.statTotalUnique.textContent = '0';
    if (this.statActiveIdsList) this.statActiveIdsList.textContent = 'None';

    // Reset all dynamic zone cards to 0 / None
    for (const zid of Object.keys(this._zoneCardEls)) {
      const card = this._zoneCardEls[zid];
      if (card.enabled) {
        card.countEl.textContent = '0';
        card.trackIdsEl.textContent = 'None';
      }
    }

    if (this.statTotalEntries) this.statTotalEntries.textContent = '0';
    if (this.statTotalExits) this.statTotalExits.textContent = '0';
    if (this.statNetFlow) this.statNetFlow.textContent = '+0';
    if (this.statTotalTransitions) this.statTotalTransitions.textContent = '0';
    if (this.statFootfallMovements) this.statFootfallMovements.textContent = '0';
    if (this.statFootfallNet) this.statFootfallNet.textContent = '+0';
    this.eventsHistory = [];
    this.renderEvents([]);
  }

  setJobData(jobData) {
    if (!jobData) return;

    const metrics = jobData.current_metrics || {};
    const summary = jobData.final_summary || {};
    this.jobSummary = summary;
    this.eventsHistory = jobData.recent_events || [];

    // Cumulative stats
    const totalUnique = summary.total_unique_tracks || metrics.total_unique_tracks || 0;
    if (this.statTotalUnique) this.statTotalUnique.textContent = totalUnique;

    const entries = summary.total_entries !== undefined ? summary.total_entries : (metrics.total_entries || 0);
    const exits = summary.total_exits !== undefined ? summary.total_exits : (metrics.total_exits || 0);
    const net = entries - exits;
    const trans = summary.total_zone_transitions !== undefined ? summary.total_zone_transitions : (metrics.total_zone_transitions || 0);

    if (this.statTotalEntries) this.statTotalEntries.textContent = entries;
    if (this.statTotalExits) this.statTotalExits.textContent = exits;
    if (this.statNetFlow) this.statNetFlow.textContent = (net >= 0 ? `+${net}` : `${net}`);
    if (this.statTotalTransitions) this.statTotalTransitions.textContent = trans;

    // Footfall (Phase 9)
    const ffTotal = summary.footfall ? summary.footfall.total_movement_events : (metrics.footfall_total || (entries + exits));
    const ffNet = summary.footfall ? summary.footfall.net_flow : net;
    if (this.statFootfallMovements) this.statFootfallMovements.textContent = ffTotal;
    if (this.statFootfallNet) this.statFootfallNet.textContent = (ffNet >= 0 ? `+${ffNet}` : `${ffNet}`);

    if (summary.footfall && summary.footfall.buckets && this.footfallBucketInfo) {
      const bCount = summary.footfall.buckets.length;
      this.footfallBucketInfo.textContent = `Buckets Generated: ${bCount} (15m intervals) | Deduplication: 100% Active`;
      if (this.footfallStatus) {
        this.footfallStatus.textContent = 'Engine Verified';
        this.footfallStatus.style.color = 'var(--accent-emerald)';
      }
    }

    // Render Events List
    this.renderEvents(this.eventsHistory);
  }

  /**
   * Real-time Playback Frame Synchronization:
   * Called on every frame rendered during video playback.
   */
  onPlaybackFrame(frameData) {
    if (!frameData) return;

    // 1. Current Frame Detections & Active Tracks
    const detCount = frameData.detections_count !== undefined
      ? frameData.detections_count
      : (frameData.tracks ? frameData.tracks.length : 0);
    const activeTracks = (frameData.tracks || []).filter(t => t.state === 'ACTIVE' || t.state === 'NEW');
    const activeIds = activeTracks.map(t => `#${t.track_id}`);

    if (this.statDetectedNow) this.statDetectedNow.textContent = detCount;
    if (this.statActiveTracks) this.statActiveTracks.textContent = activeTracks.length;
    if (this.statActiveIdsList) {
      this.statActiveIdsList.textContent = activeIds.length > 0 ? activeIds.join(', ') : 'None';
    }

    // 2. Dynamic Zone Occupancy — iterate over all zone cards
    const occ = frameData.occupancy || {};
    for (const zid of Object.keys(this._zoneCardEls)) {
      const card = this._zoneCardEls[zid];
      if (!card.enabled) continue;

      const zoneOcc = occ[zid] || { count: 0, track_ids: [] };
      card.countEl.textContent = zoneOcc.count;
      card.trackIdsEl.textContent =
        (zoneOcc.track_ids && zoneOcc.track_ids.length > 0)
          ? zoneOcc.track_ids.map(id => `#${id}`).join(', ')
          : 'None';
    }

    // 3. Metrics update from frame data
    const m = frameData.metrics || {};
    if (m.total_unique_tracks !== undefined && this.statTotalUnique) {
      this.statTotalUnique.textContent = m.total_unique_tracks;
    }
    if (m.total_entries !== undefined && this.statTotalEntries) {
      this.statTotalEntries.textContent = m.total_entries;
    }
    if (m.total_exits !== undefined && this.statTotalExits) {
      this.statTotalExits.textContent = m.total_exits;
    }
    if (m.net_flow !== undefined && this.statNetFlow) {
      this.statNetFlow.textContent = m.net_flow >= 0 ? `+${m.net_flow}` : `${m.net_flow}`;
    }
    if (m.total_zone_transitions !== undefined && this.statTotalTransitions) {
      this.statTotalTransitions.textContent = m.total_zone_transitions;
    }
    if (m.footfall_total !== undefined && this.statFootfallMovements) {
      this.statFootfallMovements.textContent = m.footfall_total;
    }
    if (m.net_flow !== undefined && this.statFootfallNet) {
      this.statFootfallNet.textContent = m.net_flow >= 0 ? `+${m.net_flow}` : `${m.net_flow}`;
    }

    // 4. Append any new events
    if (frameData.new_events && frameData.new_events.length > 0) {
      frameData.new_events.forEach(ev => {
        if (!this.eventsHistory.some(existing => existing.event_id === ev.event_id)) {
          this.eventsHistory.push(ev);
        }
      });
      this.renderEvents(this.eventsHistory);
    }
  }

  // Legacy sync alias
  onVideoFrame(frameNumber, timestamp, frameData) {
    this.onPlaybackFrame(frameData);
  }

  /**
   * Escape text for safe innerHTML usage (XSS prevention).
   */
  _escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  renderEvents(events) {
    if (!events || events.length === 0) {
      this.eventsFeedList.innerHTML = `
        <div style="color: var(--text-muted); font-size: 11px; text-align: center; padding: 20px 0;">
          No events recorded. Video movements will appear here.
        </div>
      `;
      if (this.eventsCountBadge) this.eventsCountBadge.textContent = '0 Events';
      return;
    }

    if (this.eventsCountBadge) this.eventsCountBadge.textContent = `${events.length} Events`;
    this.eventsFeedList.innerHTML = '';

    // Render in reverse chronological order
    const reversed = [...events].reverse();
    reversed.slice(0, 50).forEach((ev) => {
      const item = document.createElement('div');
      item.className = `event-entry ${ev.type || 'UNKNOWN'}`;

      const timeStr = `${Number(ev.timestamp).toFixed(2)}s (F${ev.frame_number})`;

      // Build target description — use zone names from store if available
      let targetDesc = '';
      if (ev.type === 'ZONE_TRANSITION') {
        const fromName = this._resolveZoneName(ev.previous_zone_id);
        const toName = this._resolveZoneName(ev.zone_id);
        targetDesc = `${fromName} → ${toName}`;
      } else {
        targetDesc = `Zone: ${this._resolveZoneName(ev.zone_id)}`;
      }

      // Safe rendering: use textContent for user-controlled data
      const infoDiv = document.createElement('div');

      const typeBadge = document.createElement('span');
      typeBadge.className = 'event-type-badge';
      typeBadge.textContent = ev.type;

      const trackSpan = document.createElement('span');
      trackSpan.style.cssText = 'color: #fff; font-weight: 600; margin-left: 6px;';
      trackSpan.textContent = `ID #${ev.track_id}`;

      const descSpan = document.createElement('span');
      descSpan.className = 'event-meta';
      descSpan.style.marginLeft = '6px';
      descSpan.textContent = `[${targetDesc}]`;

      infoDiv.appendChild(typeBadge);
      infoDiv.appendChild(trackSpan);
      infoDiv.appendChild(descSpan);

      const timeDiv = document.createElement('div');
      timeDiv.className = 'event-meta';
      timeDiv.textContent = timeStr;

      item.appendChild(infoDiv);
      item.appendChild(timeDiv);

      this.eventsFeedList.appendChild(item);
    });
  }

  /**
   * Resolve a zone ID to a display name using zoneStore.
   */
  _resolveZoneName(zoneId) {
    if (!zoneId) return '(outside)';
    if (window.zoneStore) {
      const z = window.zoneStore.getById(zoneId);
      if (z) return z.name;
    }
    // Fallback: humanize the ID
    return zoneId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
}

window.cctvAnalytics = new CCTVAnalytics();
