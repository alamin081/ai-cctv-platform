/**
 * AI CCTV Platform - Main Application Coordinator
 * Handles API communications, video file uploads, session management,
 * and UI state synchronization for real-time AI video playback.
 */

class CCTVApp {
  constructor() {
    this.videoSelect = document.getElementById('video-select');
    this.videoSourceBadge = document.getElementById('video-source-badge');
    this.gpuDeviceBadge = document.getElementById('gpu-device-badge');
    this.gpuDeviceText = document.getElementById('gpu-device-text');
    this.processingStatusBadge = document.getElementById('processing-status-badge');
    this.processingStatusText = document.getElementById('processing-status-text');

    // Video Meta Elements
    this.metaResolution = document.getElementById('meta-resolution');
    this.metaFps = document.getElementById('meta-fps');
    this.metaFrames = document.getElementById('meta-frames');
    this.metaDuration = document.getElementById('meta-duration');

    // Processing & Progress Elements
    this.btnStartProcess = document.getElementById('btn-start-process');
    this.progressContainer = document.getElementById('progress-container');
    this.progressBar = document.getElementById('progress-bar');
    this.progressPercent = document.getElementById('progress-percent');
    this.progressStatusLabel = document.getElementById('progress-status-label');
    this.progressFramesStat = document.getElementById('progress-frames-stat');
    this.progressFpsStat = document.getElementById('progress-fps-stat');
    this.progressElapsedStat = document.getElementById('progress-elapsed-stat');

    // Upload Elements
    this.fileInput = document.getElementById('file-input');
    this.btnUploadNew = document.getElementById('btn-upload-new');
    this.dropzone = document.getElementById('dropzone');
    this.videoStage = document.getElementById('video-stage');

    // Status bar
    this.barFps = document.getElementById('bar-fps');
    this.barFrames = document.getElementById('bar-frames');

    // State
    this.currentVideoFilename = 'test.mp4';
    this.currentVideoMeta = null;
    this.activeJobId = null;
    this.pollInterval = null;

    this.init();
  }

  async init() {
    this.initEventListeners();
    await this.fetchHealth();
    // Load zone definitions from API so cards and canvas zones are ready
    if (window.zoneStore) {
      await window.zoneStore.refresh();
    }
    await this.fetchVideos();
  }

  initEventListeners() {
    // Video selector change
    this.videoSelect.addEventListener('change', (e) => {
      if (e.target.value) {
        this.selectVideo(e.target.value);
      }
    });

    // Start Process button (Optional Annotated MP4 Export)
    this.btnStartProcess.addEventListener('click', () => {
      this.startExportProcessing();
    });

    // File Upload handling
    this.btnUploadNew.addEventListener('click', () => {
      this.fileInput.click();
    });

    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        this.uploadFile(e.target.files[0]);
      }
    });

    // Drag and Drop on Video Stage
    ['dragenter', 'dragover'].forEach((eventName) => {
      this.videoStage.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.style.display = 'flex';
        this.dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.remove('dragover');
        if (eventName === 'dragleave') {
          this.dropzone.style.display = 'none';
        }
      });
    });

    this.dropzone.addEventListener('drop', (e) => {
      this.dropzone.style.display = 'none';
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        this.uploadFile(e.dataTransfer.files[0]);
      }
    });

    // Navigation items — view switching
    document.querySelectorAll('.nav-item').forEach((item) => {
      item.addEventListener('click', () => {
        const id = item.id;
        if (id === 'nav-overview') {
          this.switchView('overview');
        } else if (id === 'nav-zones') {
          this.switchView('zones');
        } else {
          // Other nav items — just highlight, no view change yet
          document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
          item.classList.add('active');
        }
      });
    });

    // Zone Editor buttons
    const btnSave = document.getElementById('ze-btn-save');
    const btnCancel = document.getElementById('ze-btn-cancel');
    const btnReset = document.getElementById('ze-btn-reset');
    if (btnSave) btnSave.addEventListener('click', () => {
      if (window.zoneEditor) window.zoneEditor.save();
    });
    if (btnCancel) btnCancel.addEventListener('click', () => {
      if (window.zoneEditor && window.zoneEditor.cancel()) {
        // Optionally switch back to overview
      }
    });
    if (btnReset) btnReset.addEventListener('click', () => {
      if (window.zoneEditor) window.zoneEditor.resetToDefaults();
    });
  }

  async fetchHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        const devName = data.device || 'CPU';
        this.gpuDeviceBadge.textContent = devName;
        this.gpuDeviceText.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
            <rect x="9" y="9" width="6" height="6"></rect>
          </svg>
          ${devName}
        `;
      }
    } catch (e) {
      console.warn('Could not fetch /api/health:', e);
    }
  }

  async fetchVideos(selectedFilename = null) {
    try {
      const res = await fetch('/api/videos');
      if (res.ok) {
        const data = await res.json();
        const videos = data.videos || [];
        this.videoSelect.innerHTML = '';

        let targetFound = false;
        videos.forEach((v) => {
          const opt = document.createElement('option');
          opt.value = v.filename;
          opt.textContent = `${v.filename} (${v.resolution}, ${v.fps} FPS)`;
          opt.dataset.meta = JSON.stringify(v);
          if (selectedFilename && v.filename === selectedFilename) {
            opt.selected = true;
            targetFound = true;
          }
          this.videoSelect.appendChild(opt);
        });

        if (selectedFilename && targetFound) {
          this.selectVideo(selectedFilename);
        } else if (videos.length > 0) {
          this.selectVideo(videos[0].filename);
        }
      }
    } catch (e) {
      console.error('Failed to load videos:', e);
    }
  }

  /**
   * Select a video and immediately initialize its real-time playback AI session.
   */
  async selectVideo(filename) {
    this.currentVideoFilename = filename;
    this.videoSourceBadge.textContent = filename;
    this.videoSelect.value = filename;

    const fvSourceEl = document.getElementById('full-video-source-filename');
    if (fvSourceEl) {
      fvSourceEl.textContent = filename;
    }

    // Reset analytics metrics for the new session
    if (window.cctvAnalytics) {
      window.cctvAnalytics.resetMetrics();
    }

    // Retrieve metadata from option dataset
    const opt = this.videoSelect.options[this.videoSelect.selectedIndex];
    if (opt && opt.dataset.meta) {
      try {
        const meta = JSON.parse(opt.dataset.meta);
        this.currentVideoMeta = meta;
        this.metaResolution.textContent = meta.resolution || '640x480';
        this.metaFps.textContent = `${meta.fps || 30.0}`;
        this.metaFrames.textContent = `${meta.total_frames || 90}`;
        this.metaDuration.textContent = this.formatSeconds(meta.duration_seconds || 3.0);
        this.barFrames.textContent = `Frame 0/${meta.total_frames || 90}`;
      } catch (e) {
        console.warn('Failed to parse video metadata:', e);
      }
    }

    this.setStatus('idle', 'SYSTEM READY — CLICK PLAY');

    // Initialize real-time playback session in player
    if (window.cctvPlayer) {
      await window.cctvPlayer.initSession(filename);
    }
  }

  formatSeconds(secs) {
    const mins = Math.floor(secs / 60);
    const rem = (secs % 60).toFixed(2);
    return `${String(mins).padStart(2, '0')}:${String(rem).padStart(5, '0')}`;
  }

  /**
   * Upload video file:
   * Validates file, saves original, extracts metadata, and immediately makes it playable.
   * Does NOT run full AI analysis at upload time.
   */
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    this.setStatus('processing', 'UPLOADING VIDEO...');
    this.progressContainer.style.display = 'block';
    this.progressBar.style.width = '100%';
    this.progressStatusLabel.textContent = `UPLOADING ${file.name}...`;

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        this.setStatus('completed', 'UPLOAD COMPLETE — READY TO PLAY');
        this.progressContainer.style.display = 'none';

        // Refresh video list and immediately select and initialize playback
        await this.fetchVideos(data.filename);
        await this.selectVideo(data.filename);
      } else {
        const err = await res.json();
        alert(`Upload error: ${err.detail || 'Failed to upload video'}`);
        this.setStatus('failed', 'UPLOAD FAILED');
      }
    } catch (e) {
      console.error('Upload error:', e);
      alert(`Network error during upload: ${e.message}`);
      this.setStatus('failed', 'NETWORK ERROR');
    }
  }

  /**
   * Optional: Start Batch Annotated MP4 Export
   */
  async startExportProcessing() {
    if (!this.currentVideoFilename) {
      alert('Please select or upload a video file first.');
      return;
    }

    this.btnStartProcess.disabled = true;
    this.setStatus('processing', 'EXPORTING ANNOTATED MP4');
    this.progressContainer.style.display = 'block';
    this.progressBar.style.width = '0%';
    this.progressPercent.textContent = '0%';
    this.progressStatusLabel.textContent = 'RENDERING ANNOTATED VIDEO EXPORT...';

    try {
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_filename: this.currentVideoFilename }),
      });

      if (res.ok) {
        const data = await res.json();
        const job = data.job;
        this.activeJobId = job.job_id;
        this.startJobPolling(job.job_id);
      } else {
        const err = await res.json();
        alert(`Failed to start export: ${err.detail || 'Unknown error'}`);
        this.btnStartProcess.disabled = false;
        this.setStatus('failed', 'EXPORT ERROR');
      }
    } catch (e) {
      console.error('Failed to submit export job:', e);
      alert(`Network error: ${e.message}`);
      this.btnStartProcess.disabled = false;
      this.setStatus('failed', 'NETWORK ERROR');
    }
  }

  startJobPolling(jobId) {
    if (this.pollInterval) clearInterval(this.pollInterval);

    this.pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;

        const job = await res.json();
        this.updateJobProgress(job);

        if (job.status === 'COMPLETED') {
          clearInterval(this.pollInterval);
          this.pollInterval = null;
          this.onExportCompleted(job);
        } else if (job.status === 'FAILED') {
          clearInterval(this.pollInterval);
          this.pollInterval = null;
          this.onJobFailed(job);
        }
      } catch (e) {
        console.warn('Polling error:', e);
      }
    }, 400);
  }

  updateJobProgress(job) {
    const pct = job.progress || 0;
    this.progressBar.style.width = `${pct}%`;
    this.progressPercent.textContent = `${Math.round(pct)}%`;
    this.progressStatusLabel.textContent = `RENDERING FRAME ${job.current_frame} / ${job.total_frames}...`;
    this.progressFramesStat.textContent = `Frame ${job.current_frame} / ${job.total_frames}`;
    this.progressFpsStat.textContent = `Speed: ${job.pipeline_fps} FPS`;
    this.progressElapsedStat.textContent = `Elapsed: ${job.elapsed_seconds}s`;
    this.barFps.textContent = `${job.pipeline_fps} FPS`;
  }

  onExportCompleted(job) {
    this.setStatus('completed', 'EXPORT COMPLETED');
    this.btnStartProcess.disabled = false;
    this.progressStatusLabel.textContent = 'ANNOTATED VIDEO EXPORT COMPLETE!';
    this.progressBar.style.width = '100%';
    this.progressPercent.textContent = '100%';

    if (job.annotated_video_filename && window.cctvPlayer) {
      window.cctvPlayer.annotatedVideoUrl =
        job.annotated_video_url || `/media/outputs/${encodeURIComponent(job.annotated_video_filename)}`;
    }

    setTimeout(() => {
      this.progressContainer.style.display = 'none';
    }, 2500);
  }

  onJobFailed(job) {
    this.setStatus('failed', 'EXPORT FAILED');
    this.btnStartProcess.disabled = false;
    this.progressStatusLabel.textContent = `FAILED: ${job.error_message || 'Unknown processing error'}`;
    alert(`Annotated video export failed: ${job.error_message}`);
  }

  setStatus(state, text) {
    this.processingStatusBadge.className = `status-badge ${state}`;
    this.processingStatusText.textContent = text;
  }

  /**
   * Switch between 'overview' (dashboard) and 'zones' (zone editor) views.
   */
  switchView(view) {
    const dashMain = document.querySelector('.dashboard-workspace');
    const editorView = document.getElementById('zone-editor-view');
    if (!dashMain || !editorView) return;

    // Guard: unsaved changes in zone editor
    if (this.currentView === 'zones' && view !== 'zones') {
      if (window.zoneEditor && !window.zoneEditor.confirmDiscard()) return;
      window.zoneEditor.close();
    }

    // Update nav highlight
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    if (view === 'zones') {
      document.getElementById('nav-zones')?.classList.add('active');
    } else {
      document.getElementById('nav-overview')?.classList.add('active');
    }

    if (view === 'zones') {
      // Hide dashboard, show editor
      dashMain.style.display = 'none';
      editorView.style.display = 'flex';
      this.currentView = 'zones';

      // Open editor
      if (window.zoneEditor) {
        const ok = window.zoneEditor.open();
        if (!ok) {
          // No video — switch back
          this.switchView('overview');
          return;
        }
      }
    } else {
      // Show dashboard, hide editor
      editorView.style.display = 'none';
      dashMain.style.display = '';
      this.currentView = 'overview';

      // Trigger canvas resize so player renders correctly after being hidden
      if (window.cctvPlayer) {
        window.cctvPlayer.resizeCanvas();
      }
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.cctvApp = new CCTVApp();
});
