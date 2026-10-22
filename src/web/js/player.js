/**
 * AI CCTV Platform - Video Player & Live AI Overlay Controller
 * Real-time video frame synchronization with AI detection boxes, track IDs,
 * spatial zone containment, and live dashboard analytics during playback.
 * 
 * Features:
 * - requestVideoFrameCallback() synchronized frame detection with rAF fallback
 * - Adaptive backpressure: strictly <= 1 in-flight inference request
 * - Documented tolerance window: continuous, non-flickering bounding boxes and track IDs
 * - Stable ByteTrack tracking with state preservation across playback
 * - Clean state reset on intentional seek
 * - Accurate aspect ratio letterbox/pillarbox coordinate transformation
 */

class CCTVPlayer {
  constructor() {
    this.video = document.getElementById('main-video');
    this.canvas = document.getElementById('overlay-canvas');
    this.ctx = this.canvas.getContext('2d');

    // UI Controls
    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.playPauseIcon = document.getElementById('play-pause-icon');
    this.btnStepBack = document.getElementById('btn-step-back');
    this.btnStepFwd = document.getElementById('btn-step-fwd');
    this.seekSlider = document.getElementById('seek-slider');
    this.timecodeDisplay = document.getElementById('timecode-display');
    this.frameCounterDisplay = document.getElementById('frame-counter-display');

    // Telemetry display elements in bottom status bar
    this.barLatency = document.getElementById('bar-latency');
    this.barFps = document.getElementById('bar-fps');
    this.barFrames = document.getElementById('bar-frames');
    this.barDropped = document.getElementById('bar-dropped');
    this.barStatus = document.getElementById('bar-status');

    // Status Overlays
    this.videoLoading = document.getElementById('video-loading');
    this.videoLoadingText = document.getElementById('video-loading-text');
    this.videoError = document.getElementById('video-error');
    this.videoErrorTitle = document.getElementById('video-error-title');
    this.videoErrorDesc = document.getElementById('video-error-desc');
    this.btnRetryVideo = document.getElementById('btn-retry-video');

    // Layer Toggles
    this.toggleBoxes = document.getElementById('toggle-boxes');
    this.toggleTracks = document.getElementById('toggle-tracks');
    this.toggleZones = document.getElementById('toggle-zones');

    // Full Video Mode & Floating Controls
    this.isFullVideoMode = false;
    this.btnFullVideo = document.getElementById('btn-full-video');
    this.btnDashboardView = document.getElementById('btn-dashboard-view');
    this.btnExitFullVideo = document.getElementById('btn-exit-full-video');
    this.btnFullscreenNative = document.getElementById('btn-fullscreen-native');
    this.btnVolumeToggle = document.getElementById('btn-volume-toggle');
    this.volumeSlider = document.getElementById('volume-slider');

    // Floating Overlay Layer Toggles (Full Video Stage)
    this.fToggleBoxes = document.getElementById('f-toggle-boxes');
    this.fToggleTracks = document.getElementById('f-toggle-tracks');
    this.fToggleZones = document.getElementById('f-toggle-zones');

    // Configuration & State
    this.showBoxes = true;
    this.showTracks = true;
    this.showZones = true;
    this.viewMode = 'overlay'; // 'overlay' or 'annotated'

    this.sessionId = null;
    this.currentVideoFilename = '';
    this.fps = 30.0;
    this.totalFrames = 0;
    this.durationSeconds = 0.0;
    this.rawVideoUrl = '';
    this.annotatedVideoUrl = '';

    // Synchronized Frame State
    this.latestInference = null;
    this.isInferenceBusy = false;
    this.inFlightRequestId = 0;
    this.isLoopRunning = false;
    this.isSeeking = false;
    this.droppedFramesCount = 0;
    this.processedFramesCount = 0;

    // Spatial Zones — loaded from zoneStore (no hardcoded defaults)
    this.zones = [];

    // Subscribe to zoneStore updates for live zone refresh
    if (window.zoneStore) {
      window.zoneStore.subscribe((zones) => {
        this.zones = zones.filter(z => z.enabled);
        // Re-render canvas overlay immediately so polygons/labels appear
        this.renderOverlay(this.latestInference);
      });
    }

    this.initEvents();
  }

  initEvents() {
    // Video streaming lifecycle
    this.video.addEventListener('loadstart', () => {
      this.showLoading(true, 'Connecting video stream...');
      this.showError(false);
    });

    this.video.addEventListener('waiting', () => {
      this.showLoading(true, 'Buffering video...');
    });

    this.video.addEventListener('playing', () => {
      this.showLoading(false);
      this.showError(false);
      this.onPlay();
    });

    this.video.addEventListener('pause', () => {
      this.onPause();
    });

    this.video.addEventListener('ended', () => {
      this.onPause();
      this.updatePlayIcon(false);
    });

    this.video.addEventListener('canplay', () => {
      this.showLoading(false);
    });

    this.video.addEventListener('loadeddata', () => {
      this.showLoading(false);
    });

    this.video.addEventListener('error', () => {
      this.handleVideoError();
    });

    if (this.btnRetryVideo) {
      this.btnRetryVideo.addEventListener('click', () => {
        this.showError(false);
        this.showLoading(true, 'Retrying video stream...');
        if (this.video.src) {
          const src = this.video.src;
          this.video.src = '';
          this.video.src = src;
          this.video.load();
        }
      });
    }

    // Play/Pause button
    this.btnPlayPause.addEventListener('click', () => this.togglePlay());

    // Seeking events
    this.seekSlider.addEventListener('input', (e) => this.onSeekInput(parseFloat(e.target.value)));
    this.seekSlider.addEventListener('change', (e) => this.onSeekCommit(parseFloat(e.target.value)));

    // Frame step buttons
    this.btnStepBack.addEventListener('click', () => this.stepFrame(-1));
    this.btnStepFwd.addEventListener('click', () => this.stepFrame(1));

    // Playback rate buttons
    document.getElementById('btn-speed-05').addEventListener('click', (e) => this.setSpeed(0.5, e.target));
    document.getElementById('btn-speed-1').addEventListener('click', (e) => this.setSpeed(1.0, e.target));
    document.getElementById('btn-speed-2').addEventListener('click', (e) => this.setSpeed(2.0, e.target));

    // Layer toggles (synchronized between toolbar and floating stage)
    this.toggleBoxes.addEventListener('click', () => this.toggleLayer('boxes'));
    this.toggleTracks.addEventListener('click', () => this.toggleLayer('tracks'));
    this.toggleZones.addEventListener('click', () => this.toggleLayer('zones'));

    if (this.fToggleBoxes) {
      this.fToggleBoxes.addEventListener('click', () => this.toggleLayer('boxes'));
    }
    if (this.fToggleTracks) {
      this.fToggleTracks.addEventListener('click', () => this.toggleLayer('tracks'));
    }
    if (this.fToggleZones) {
      this.fToggleZones.addEventListener('click', () => this.toggleLayer('zones'));
    }

    // Full Video Mode Activators & Exit Controls
    if (this.btnFullVideo) {
      this.btnFullVideo.addEventListener('click', () => this.setFullVideoMode(true));
    }
    if (this.btnDashboardView) {
      this.btnDashboardView.addEventListener('click', () => this.setFullVideoMode(false));
    }
    if (this.btnExitFullVideo) {
      this.btnExitFullVideo.addEventListener('click', () => this.setFullVideoMode(false));
    }
    if (this.btnFullscreenNative) {
      this.btnFullscreenNative.addEventListener('click', () => this.toggleNativeFullscreen());
    }

    // Volume & Mute Controls
    if (this.btnVolumeToggle) {
      this.btnVolumeToggle.addEventListener('click', () => this.toggleMute());
    }
    if (this.volumeSlider) {
      this.volumeSlider.addEventListener('input', (e) => this.setVolume(parseFloat(e.target.value)));
    }

    // Keyboard Shortcuts (Esc to exit Full Video / Native Fullscreen, F to toggle)
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        }
        if (this.isFullVideoMode) {
          this.setFullVideoMode(false);
        }
      } else if (e.key === 'f' || e.key === 'F') {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
          this.setFullVideoMode(!this.isFullVideoMode);
        }
      }
    });

    document.addEventListener('fullscreenchange', () => {
      const isFs = !!document.fullscreenElement;
      if (this.btnFullscreenNative) {
        this.btnFullscreenNative.classList.toggle('active', isFs);
        this.btnFullscreenNative.title = isFs ? 'Exit Browser Fullscreen (Esc)' : 'Toggle Browser Fullscreen';
      }
      this.resizeCanvas();
    });

    // Window and video dimension updates
    window.addEventListener('resize', () => this.resizeCanvas());
    this.video.addEventListener('loadedmetadata', () => {
      this.showLoading(false);
      this.resizeCanvas();
      this.updateTimeDisplay(this.video.currentTime);
      this.renderOverlay(this.latestInference);
    });
  }

  showLoading(show, message = 'Loading video stream...') {
    if (!this.videoLoading) return;
    this.videoLoading.style.display = show ? 'flex' : 'none';
    if (this.videoLoadingText && message) {
      this.videoLoadingText.textContent = message;
    }
  }

  showError(show, title = '', desc = '') {
    if (!this.videoError) return;
    this.videoError.style.display = show ? 'flex' : 'none';
    if (title && this.videoErrorTitle) {
      this.videoErrorTitle.textContent = title;
    }
    if (desc && this.videoErrorDesc) {
      this.videoErrorDesc.textContent = desc;
    }
  }

  handleVideoError() {
    this.showLoading(false);
    const err = this.video.error;
    let title = 'Video Playback Error';
    let desc = 'Failed to load video stream from server.';

    if (err) {
      switch (err.code) {
        case 1:
          title = 'Playback Aborted';
          desc = 'The video playback was aborted.';
          break;
        case 2:
          title = 'Network Streaming Error';
          desc = 'A network error occurred while streaming video from the server.';
          break;
        case 3:
          title = 'Video Decoding Failed';
          desc = 'The video could not be decoded. Media or browser-unsupported video codec.';
          break;
        case 4:
          title = 'Source Not Found';
          desc = 'The video source was not found (HTTP 404).';
          break;
        default:
          desc = err.message || `Media error occurred (code ${err.code}).`;
      }
    }
    this.showError(true, title, desc);
  }

  /**
   * Initialize a real-time playback session for a selected video.
   */
  async initSession(videoFilename) {
    this.currentVideoFilename = videoFilename;
    this.latestInference = null;
    this.isInferenceBusy = false;
    this.inFlightRequestId++;
    this.droppedFramesCount = 0;
    this.processedFramesCount = 0;

    try {
      this.showLoading(true, 'Initializing AI playback engine...');
      const res = await fetch('/api/playback/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_filename: videoFilename }),
      });

      if (!res.ok) {
        throw new Error(`Failed to initialize session: ${res.statusText}`);
      }

      const data = await res.json();
      this.sessionId = data.session_id;
      this.fps = data.fps || 30.0;
      this.totalFrames = data.total_frames || 90;
      this.durationSeconds = data.duration_seconds || 3.0;
      this.rawVideoUrl = data.video_url;

      // Push zone definitions into zoneStore (session response includes normalized zones)
      if (data.zones && data.zones.length > 0) {
        if (window.zoneStore) {
          window.zoneStore.setFromSession(data.zones);
        }
      } else {
        // Session has no zones — try loading from API
        if (window.zoneStore) {
          await window.zoneStore.refresh();
        }
      }

      // Load initial frame analysis if available
      if (data.initial_frame) {
        this.latestInference = data.initial_frame;
      }

      // Load video in HTML5 player
      this.showError(false);
      this.video.src = this.rawVideoUrl;
      this.video.load();
      this.resizeCanvas();

      // Update Full Video Source Video Filename Display
      const fvSourceEl = document.getElementById('full-video-source-filename');
      if (fvSourceEl) {
        fvSourceEl.textContent = videoFilename;
      }

      // Display initial frame overlay & telemetry
      this.renderOverlay(this.latestInference);
      if (window.cctvAnalytics && this.latestInference) {
        window.cctvAnalytics.onPlaybackFrame(this.latestInference);
      }
      if (data.metrics) {
        this.updateTelemetry(data.metrics);
      }

      this.showLoading(false);
    } catch (e) {
      console.error('Session initialization error:', e);
      this.showLoading(false);
      this.showError(true, 'AI Engine Initialization Error', e.message);
    }
  }

  resizeCanvas() {
    const rect = this.video.getBoundingClientRect();
    if (rect.width && rect.height) {
      this.canvas.width = rect.width;
      this.canvas.height = rect.height;
      this.renderOverlay(this.latestInference);
    }
  }

  togglePlay() {
    if (this.video.paused || this.video.ended) {
      this.video.play().catch((e) => console.warn('Video play error:', e));
    } else {
      this.video.pause();
    }
  }

  updatePlayIcon(isPlaying) {
    if (isPlaying) {
      this.playPauseIcon.innerHTML = `
        <rect x="6" y="4" width="4" height="16"></rect>
        <rect x="14" y="4" width="4" height="16"></rect>
      `;
    } else {
      this.playPauseIcon.innerHTML = `
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      `;
    }
  }

  async onPlay() {
    this.updatePlayIcon(true);
    if (!this.sessionId) return;

    if (this.barStatus) {
      this.barStatus.textContent = 'STATUS: ANALYZING LIVE';
      this.barStatus.style.color = 'var(--accent-cyan)';
    }

    try {
      await fetch(`/api/playback/${this.sessionId}/play`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: this.video.currentTime,
          speed: this.video.playbackRate,
        }),
      });
    } catch (e) {
      console.warn('Error starting playback on server:', e);
    }

    // Start presentation synchronization loop
    this.startPlaybackLoop();
  }

  async onPause() {
    this.updatePlayIcon(false);
    this.isLoopRunning = false;
    this.isInferenceBusy = false;

    if (this.barStatus) {
      this.barStatus.textContent = 'STATUS: PAUSED';
      this.barStatus.style.color = 'var(--accent-amber)';
    }

    if (!this.sessionId) return;

    try {
      await fetch(`/api/playback/${this.sessionId}/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp: this.video.currentTime }),
      });
    } catch (e) {
      console.warn('Error pausing playback on server:', e);
    }

    // Freeze overlay cleanly on current frame
    this.renderOverlay(this.latestInference);
  }

  /**
   * Continuous Video Presentation Loop:
   * Uses requestVideoFrameCallback() where supported (Chrome/Edge/Firefox),
   * falling back to requestAnimationFrame().
   */
  startPlaybackLoop() {
    if (this.isLoopRunning) return;
    this.isLoopRunning = true;

    if ('requestVideoFrameCallback' in this.video) {
      const onVideoFrame = (now, metadata) => {
        if (!this.isLoopRunning || this.video.paused || this.video.ended) {
          this.isLoopRunning = false;
          return;
        }

        this.handlePresentedFrame(metadata.mediaTime);
        this.video.requestVideoFrameCallback(onVideoFrame);
      };
      this.video.requestVideoFrameCallback(onVideoFrame);
    } else {
      const onRaf = () => {
        if (!this.isLoopRunning || this.video.paused || this.video.ended) {
          this.isLoopRunning = false;
          return;
        }

        this.handlePresentedFrame(this.video.currentTime);
        requestAnimationFrame(onRaf);
      };
      requestAnimationFrame(onRaf);
    }
  }

  /**
   * Handle each presented video frame:
   * 1. Updates timeline UI.
   * 2. Submits frame for inference if pipeline is idle (Adaptive Backpressure).
   * 3. Renders synchronized bounding boxes, track IDs, and zones on canvas.
   */
  handlePresentedFrame(mediaTime) {
    this.updateTimeDisplay(mediaTime);

    const currentFrameIdx = Math.min(this.totalFrames, Math.max(1, Math.floor(mediaTime * this.fps) + 1));

    // Schedule inference if idle (Strictly <= 1 in-flight request prevents queue buildup)
    if (!this.isInferenceBusy && this.sessionId && !this.video.paused) {
      this.submitFrameForInference(mediaTime, currentFrameIdx);
    } else if (this.isInferenceBusy) {
      this.droppedFramesCount++;
      if (this.barDropped) {
        this.barDropped.textContent = `${this.droppedFramesCount} Skipped`;
      }
    }

    // Overlay Rendering with Documented Stale Result Tolerance:
    // Renders latest valid inference smoothly across playback presentation frames.
    // Detections remain stably visible until either a fresh inference updates them,
    // or an intentional seek/large jump occurs (> 1.2s).
    if (this.latestInference) {
      const timeDelta = Math.abs(mediaTime - this.latestInference.timestamp);
      // Adaptive persistence window (min 1.2s, scaled for higher playback speeds)
      const maxTolerance = Math.max(1.2, (1.0 / this.fps) * 15 * (this.video.playbackRate || 1.0));

      if (timeDelta <= maxTolerance) {
        this.renderOverlay(this.latestInference);
        if (window.cctvAnalytics) {
          window.cctvAnalytics.onPlaybackFrame(this.latestInference);
        }
      } else {
        // Discontinuous jump / unhandled seek: clear bounding boxes
        this.renderOverlay(null);
      }
    } else {
      this.renderOverlay(null);
    }
  }

  /**
   * Submit frame for inference with chronological tracking:
   */
  async submitFrameForInference(mediaTime, currentFrameIdx) {
    this.isInferenceBusy = true;
    const reqId = ++this.inFlightRequestId;
    const tStart = performance.now();

    try {
      const res = await fetch(`/api/playback/${this.sessionId}/process_frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: mediaTime,
          frame_number: currentFrameIdx,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        // Check if this request is still relevant (not cancelled by seek)
        if (reqId === this.inFlightRequestId) {
          const frameData = data.frame_data;
          this.latestInference = frameData;
          this.processedFramesCount++;

          // Immediately render fresh bounding boxes and tracks
          this.renderOverlay(frameData);

          if (window.cctvAnalytics) {
            window.cctvAnalytics.onPlaybackFrame(frameData);
          }

          if (frameData.metrics) {
            this.updateTelemetry(frameData.metrics);
          }
        }
      }
    } catch (e) {
      console.warn('Inference request error:', e);
    } finally {
      if (reqId === this.inFlightRequestId) {
        this.isInferenceBusy = false;
      }
    }
  }

  updateTimeDisplay(curTime) {
    const dur = this.video.duration || this.durationSeconds || 0.01;
    const pct = (curTime / dur) * 100;

    if (!this.isSeeking) {
      this.seekSlider.value = pct;
    }

    this.timecodeDisplay.textContent = `${this.formatTimecode(curTime)} / ${this.formatTimecode(dur)}`;
    const curFrame = Math.min(this.totalFrames, Math.max(1, Math.floor(curTime * this.fps) + 1));
    this.frameCounterDisplay.textContent = `Frame ${curFrame} / ${this.totalFrames}`;
    if (this.barFrames) {
      this.barFrames.textContent = `Frame ${curFrame}/${this.totalFrames}`;
    }
  }

  onSeekInput(percent) {
    this.isSeeking = true;
    if (!this.video.duration) return;
    const targetTime = (percent / 100) * this.video.duration;
    this.timecodeDisplay.textContent = `${this.formatTimecode(targetTime)} / ${this.formatTimecode(this.video.duration)}`;
  }

  async onSeekCommit(percent) {
    this.isSeeking = false;
    if (!this.video.duration) return;
    const targetTime = (percent / 100) * this.video.duration;
    await this.seekToTime(targetTime);
  }

  stepFrame(delta) {
    this.video.pause();
    const frameTime = 1.0 / this.fps;
    const targetTime = Math.max(
      0,
      Math.min(this.video.duration || this.durationSeconds, this.video.currentTime + delta * frameTime)
    );
    this.seekToTime(targetTime);
  }

  /**
   * Instant Seek Re-synchronization:
   * 1. Cancels in-flight requests.
   * 2. Clears stale canvas overlay.
   * 3. Resets ByteTrack state on backend to eliminate ghost tracks.
   * 4. Analyzes and renders seeked frame immediately.
   */
  async seekToTime(targetTime) {
    // Invalidate any in-flight requests
    ++this.inFlightRequestId;
    this.isInferenceBusy = false;

    // Clear old overlay immediately
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.video.currentTime = targetTime;
    this.updateTimeDisplay(targetTime);

    if (!this.sessionId) return;

    if (this.barStatus) {
      this.barStatus.textContent = 'STATUS: SEEKING...';
      this.barStatus.style.color = 'var(--accent-amber)';
    }

    try {
      const res = await fetch(`/api/playback/${this.sessionId}/seek`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp: targetTime }),
      });

      if (res.ok) {
        const data = await res.json();
        const frameData = data.frame_data;

        if (frameData) {
          this.latestInference = frameData;
          this.renderOverlay(frameData);

          if (window.cctvAnalytics) {
            window.cctvAnalytics.onPlaybackFrame(frameData);
          }
          if (frameData.metrics) {
            this.updateTelemetry(frameData.metrics);
          }
        }

        if (this.barStatus) {
          this.barStatus.textContent = this.video.paused ? 'STATUS: PAUSED' : 'STATUS: ANALYZING LIVE';
          this.barStatus.style.color = this.video.paused ? 'var(--accent-amber)' : 'var(--accent-cyan)';
        }
      }
    } catch (e) {
      console.warn('Seek error:', e);
    }
  }

  async setSpeed(speed, btnEl) {
    this.video.playbackRate = speed;
    document.querySelectorAll('.playback-buttons-row .ctrl-btn-group button').forEach((b) => {
      b.classList.remove('primary');
    });
    if (btnEl) btnEl.classList.add('primary');

    if (this.sessionId) {
      try {
        await fetch(`/api/playback/${this.sessionId}/play`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            timestamp: this.video.currentTime,
            speed: speed,
          }),
        });
      } catch (e) {
        console.warn('Error setting speed on server:', e);
      }
    }
  }

  updateTelemetry(metrics) {
    if (!metrics) return;
    if (this.barLatency && metrics.latency_ms !== undefined) {
      this.barLatency.textContent = `${metrics.latency_ms} ms`;
    }
    if (this.barFps && metrics.pipeline_fps !== undefined) {
      this.barFps.textContent = `${metrics.pipeline_fps} FPS`;
    }
    if (this.barDropped && metrics.dropped_frames !== undefined) {
      this.barDropped.textContent = `${metrics.dropped_frames + this.droppedFramesCount} Skipped`;
    }
    if (window.cctvApp && metrics.pipeline_fps) {
      window.cctvApp.barFps.textContent = `${metrics.pipeline_fps} FPS`;
    }
  }

  formatTimecode(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(5, '0')}`;
  }

  /**
   * Precise Canvas Overlay Renderer:
   * Handles letterbox / pillarbox aspect ratio scaling, bounding boxes with
   * track IDs and confidence scores, and spatial zone containment.
   */
  renderOverlay(frameData) {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    this.ctx.clearRect(0, 0, cw, ch);

    // Live sync for compact floating AI overlay
    this.updateFloatingHud(frameData);

    if (this.viewMode === 'annotated') return;

    const vw = this.video.videoWidth || 640;
    const vh = this.video.videoHeight || 480;

    // Accurate object-fit: contain scaling calculation
    const videoRatio = vw / vh;
    const containerRatio = cw / ch;

    let renderW, renderH, offsetX, offsetY;
    if (containerRatio > videoRatio) {
      // Pillarbox (black bars left and right)
      renderH = ch;
      renderW = ch * videoRatio;
      offsetX = (cw - renderW) / 2;
      offsetY = 0;
    } else {
      // Letterbox (black bars top and bottom)
      renderW = cw;
      renderH = cw / videoRatio;
      offsetX = 0;
      offsetY = (ch - renderH) / 2;
    }

    const toCanvasX = (x) => offsetX + (x / vw) * renderW;
    const toCanvasY = (y) => offsetY + (y / vh) * renderH;
    const toScaleW = (w) => (w / vw) * renderW;
    const toScaleH = (h) => (h / vh) * renderH;

    // 1. Draw Spatial Zone fills and borders (BEFORE tracks)
    // Labels are drawn AFTER tracks (pass 2) so they are never hidden.
    let zoneCanvasData = [];
    if (this.showZones && this.zones) {
      this.zones.forEach((zone) => {
        const poly = zone.polygon;
        if (!poly || poly.length < 3) return;

        this.ctx.beginPath();
        const canvasPts = poly.map(pt => ({
          x: toCanvasX(pt[0] * vw),
          y: toCanvasY(pt[1] * vh),
        }));
        this.ctx.moveTo(canvasPts[0].x, canvasPts[0].y);
        for (let i = 1; i < canvasPts.length; i++) {
          this.ctx.lineTo(canvasPts[i].x, canvasPts[i].y);
        }
        this.ctx.closePath();

        this.ctx.fillStyle = zone.fill || 'rgba(0, 240, 255, 0.12)';
        this.ctx.fill();

        this.ctx.strokeStyle = zone.color || '#00f0ff';
        this.ctx.lineWidth = 1.8;
        this.ctx.stroke();

        zoneCanvasData.push({ zone, canvasPts });
      });
    }

    if (!frameData || !frameData.tracks) {
      this._drawZoneLabels(zoneCanvasData, renderW);
      return;
    }

    // 2. Draw Person Bounding Boxes, Confidence, and Track IDs
    frameData.tracks.forEach((track) => {
      const [x1, y1, x2, y2] = track.bbox;
      const sx1 = toCanvasX(x1);
      const sy1 = toCanvasY(y1);
      const sw = toScaleW(x2 - x1);
      const sh = toScaleH(y2 - y1);

      const isLost = track.state === 'LOST' || track.state === 'COASTED';
      const boxColor = isLost ? '#f59e0b' : '#00f0ff';

      // Draw Detection Bounding Box
      if (this.showBoxes) {
        this.ctx.strokeStyle = boxColor;
        this.ctx.lineWidth = 2;
        this.ctx.shadowColor = boxColor;
        this.ctx.shadowBlur = 8;
        this.ctx.strokeRect(sx1, sy1, sw, sh);
        this.ctx.shadowBlur = 0;

        // Tactical Corner Accents
        const cLen = Math.min(10, sw / 3);
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2.5;

        // Top-left corner
        this.ctx.beginPath();
        this.ctx.moveTo(sx1, sy1 + cLen);
        this.ctx.lineTo(sx1, sy1);
        this.ctx.lineTo(sx1 + cLen, sy1);
        this.ctx.stroke();

        // Bottom-right corner
        this.ctx.beginPath();
        this.ctx.moveTo(sx1 + sw, sy1 + sh - cLen);
        this.ctx.lineTo(sx1 + sw, sy1 + sh);
        this.ctx.lineTo(sx1 + sw - cLen, sy1 + sh);
        this.ctx.stroke();
      }

      // Draw Track ID & Confidence Label
      if (this.showTracks) {
        const confText = track.confidence ? ` ${(track.confidence * 100).toFixed(0)}%` : '';
        const labelText = `ID #${track.track_id} [${track.state}]${confText}`;
        this.ctx.font = 'bold 11px JetBrains Mono, monospace';
        const textMetrics = this.ctx.measureText(labelText);
        const tagW = textMetrics.width + 12;
        const tagH = 19;

        const tagY = Math.max(0, sy1 - tagH);

        this.ctx.fillStyle = 'rgba(7, 9, 14, 0.88)';
        this.ctx.fillRect(sx1, tagY, tagW, tagH);

        this.ctx.strokeStyle = boxColor;
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(sx1, tagY, tagW, tagH);

        this.ctx.fillStyle = boxColor;
        this.ctx.fillText(labelText, sx1 + 6, tagY + 13);
      }
    });

    // 3. Draw Zone Labels LAST (on top of tracks)
    this._drawZoneLabels(zoneCanvasData, renderW);
  }

  /**
   * Draw zone labels with anti-overlap. Separate pass drawn AFTER tracks
   * so labels are never hidden behind bounding boxes.
   */
  _drawZoneLabels(zoneCanvasData, renderW) {
    if (!zoneCanvasData || zoneCanvasData.length === 0) return;
    const placedLabels = [];
    const labelFontSize = Math.max(11, Math.min(14, Math.round(renderW / 55)));
    const labelFont = `bold ${labelFontSize}px JetBrains Mono, monospace`;
    const labelPadH = 6, labelPadV = 3, labelInset = 8;

    zoneCanvasData.forEach(({ zone, canvasPts }) => {
      let minX = Infinity, minY = Infinity, maxX = -Infinity;
      for (const pt of canvasPts) {
        if (pt.x < minX) minX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.x > maxX) maxX = pt.x;
      }
      const zoneCanvasW = maxX - minX;
      this.ctx.font = labelFont;
      let labelText = zone.name.toUpperCase();
      const maxLabelW = zoneCanvasW - labelInset * 2 - labelPadH * 2;
      if (maxLabelW > 20) {
        if (this.ctx.measureText(labelText).width > maxLabelW) {
          while (labelText.length > 1 && this.ctx.measureText(labelText + '…').width > maxLabelW) {
            labelText = labelText.slice(0, -1);
          }
          labelText += '…';
        }
      }
      const textW = this.ctx.measureText(labelText).width;
      const textH = labelFontSize;
      let labelX = minX + labelInset;
      let labelY = minY + labelInset + labelPadV + textH;

      for (const prev of placedLabels) {
        const overlapX = labelX < prev.x + prev.w + 4 && labelX + textW + labelPadH * 2 > prev.x - 4;
        const overlapY = Math.abs(labelY - prev.y) < textH + labelPadV * 2 + 2;
        if (overlapX && overlapY) labelY = prev.y + textH + labelPadV * 2 + 4;
      }

      const bgX = labelX - labelPadH, bgY = labelY - textH - labelPadV;
      const bgW = textW + labelPadH * 2, bgH = textH + labelPadV * 2;
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
      this.ctx.beginPath();
      const r = 3;
      this.ctx.moveTo(bgX + r, bgY);
      this.ctx.lineTo(bgX + bgW - r, bgY);
      this.ctx.quadraticCurveTo(bgX + bgW, bgY, bgX + bgW, bgY + r);
      this.ctx.lineTo(bgX + bgW, bgY + bgH - r);
      this.ctx.quadraticCurveTo(bgX + bgW, bgY + bgH, bgX + bgW - r, bgY + bgH);
      this.ctx.lineTo(bgX + r, bgY + bgH);
      this.ctx.quadraticCurveTo(bgX, bgY + bgH, bgX, bgY + bgH - r);
      this.ctx.lineTo(bgX, bgY + r);
      this.ctx.quadraticCurveTo(bgX, bgY, bgX + r, bgY);
      this.ctx.closePath();
      this.ctx.fill();

      this.ctx.fillStyle = zone.color || '#00f0ff';
      this.ctx.font = labelFont;
      this.ctx.fillText(labelText, labelX, labelY);
      placedLabels.push({ x: bgX, y: labelY, w: bgW });
    });
  }

  /**
   * Toggle or set Full Video Mode.
   * Expands the video stage to the viewport, hides extraneous sidebars,
   * displays the streamlined top bar and floating HUD, and preserves playback and AI state.
   */
  setFullVideoMode(enable) {
    this.isFullVideoMode = !!enable;
    const appContainer = document.querySelector('.app-container');
    if (appContainer) {
      appContainer.classList.toggle('full-video-mode', this.isFullVideoMode);
    }

    if (this.btnFullVideo) {
      this.btnFullVideo.classList.toggle('active', this.isFullVideoMode);
    }

    // Immediately sync floating HUD with latest inference state
    if (this.latestInference) {
      this.updateFloatingHud(this.latestInference);
    }

    // Force canvas resize to match new container geometry seamlessly
    this.resizeCanvas();
    requestAnimationFrame(() => this.resizeCanvas());
    setTimeout(() => this.resizeCanvas(), 50);
    setTimeout(() => this.resizeCanvas(), 160);
  }

  /**
   * Browser-native Fullscreen API handler.
   */
  async toggleNativeFullscreen() {
    try {
      const container = document.querySelector('.app-container') || document.documentElement;
      if (!document.fullscreenElement) {
        await container.requestFullscreen();
        if (!this.isFullVideoMode) {
          this.setFullVideoMode(true);
        }
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.warn('Native browser fullscreen error:', err);
    }
  }

  /**
   * Synchronized layer toggling between main controls and floating HUD pills.
   */
  toggleLayer(layer) {
    if (layer === 'boxes') {
      this.showBoxes = !this.showBoxes;
      document.querySelectorAll('#toggle-boxes, #f-toggle-boxes').forEach((el) => {
        el.classList.toggle('active', this.showBoxes);
      });
    } else if (layer === 'tracks') {
      this.showTracks = !this.showTracks;
      document.querySelectorAll('#toggle-tracks, #f-toggle-tracks').forEach((el) => {
        el.classList.toggle('active', this.showTracks);
      });
    } else if (layer === 'zones') {
      this.showZones = !this.showZones;
      document.querySelectorAll('#toggle-zones, #f-toggle-zones').forEach((el) => {
        el.classList.toggle('active', this.showZones);
      });
    }
    this.renderOverlay(this.latestInference);
  }

  /**
   * Volume Mute/Unmute toggle.
   */
  toggleMute() {
    this.video.muted = !this.video.muted;
    this.updateVolumeUi();
  }

  /**
   * Volume level adjustment.
   */
  setVolume(vol) {
    this.video.volume = Math.max(0, Math.min(1, vol));
    this.video.muted = this.video.volume === 0;
    this.updateVolumeUi();
  }

  /**
   * Update Volume icon and slider UI state.
   */
  updateVolumeUi() {
    const icon = document.getElementById('volume-icon');
    if (!icon) return;
    if (this.video.muted || this.video.volume === 0) {
      icon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>';
    } else {
      icon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>';
    }
    if (this.volumeSlider) {
      this.volumeSlider.value = this.video.muted ? 0 : this.video.volume;
    }
  }

  /**
   * Live synchronization of compact floating AI HUD metrics on the video stage.
   */
  updateFloatingHud(frameData) {
    if (!frameData) return;
    const detEl = document.getElementById('fhud-det');
    const tracksEl = document.getElementById('fhud-tracks');
    const occEl = document.getElementById('fhud-occ-pill');

    if (detEl) {
      detEl.textContent = frameData.detections_count !== undefined
        ? frameData.detections_count
        : (frameData.tracks ? frameData.tracks.length : 0);
    }
    if (tracksEl) {
      const active = (frameData.tracks || []).filter((t) => t.state === 'ACTIVE' || t.state === 'NEW').length;
      tracksEl.textContent = active;
    }
    if (occEl && frameData.occupancy) {
      const entries = Object.entries(frameData.occupancy);
      if (entries.length === 0) {
        occEl.textContent = 'Occ: 0';
      } else {
        // Show up to 3 zones, "+N more" if more
        const parts = entries.slice(0, 3).map(([k, v]) => {
          const name = v.zone_name || (window.zoneStore ? (window.zoneStore.getById(k)?.name || k) : k);
          return `${name}: ${v.count}`;
        });
        const remainder = entries.length - 3;
        occEl.textContent = parts.join(' | ') + (remainder > 0 ? ` +${remainder} more` : '');
      }
    }
  }

  /**
   * Compute the video render rectangle inside the canvas, accounting for
   * object-fit: contain scaling (letterbox / pillarbox).
   * Returns { renderW, renderH, offsetX, offsetY }.
   * Reusable by the zone editor (Phase 4).
   */
  getVideoRenderRect() {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const vw = this.video.videoWidth || 640;
    const vh = this.video.videoHeight || 480;
    const videoRatio = vw / vh;
    const containerRatio = cw / ch;

    let renderW, renderH, offsetX, offsetY;
    if (containerRatio > videoRatio) {
      renderH = ch;
      renderW = ch * videoRatio;
      offsetX = (cw - renderW) / 2;
      offsetY = 0;
    } else {
      renderW = cw;
      renderH = cw / videoRatio;
      offsetX = 0;
      offsetY = (ch - renderH) / 2;
    }
    return { renderW, renderH, offsetX, offsetY, vw, vh };
  }
}

window.cctvPlayer = new CCTVPlayer();
