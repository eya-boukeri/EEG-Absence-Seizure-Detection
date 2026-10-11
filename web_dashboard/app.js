// NeuroMining EEG Dashboard — Client Logic & Interactive Pipeline
let appState = {
  info: null,
  datasets: [],
  currentChannel: 'ALL',
  currentModel: 'Random Forest',
  signalData: null,
  activeLobe: 'ALL',
  viewRange: { start: 0, end: null },
  totalDuration: 3121.0,
  trainResult: null,
  isDragging: false,
  dragStartX: 0,
  dragStartRange: { start: 0, end: 0 }
};

// DOM Elements
const channelSelect = document.getElementById('channelSelect');
const modelSelect = document.getElementById('modelSelect');
const datasetSelect = document.getElementById('datasetSelect');
const btnTrain = document.getElementById('btnTrain');
const trainSpinner = document.getElementById('trainSpinner');
const lobeFilters = document.getElementById('lobeFilters');
const seizureQuickList = document.getElementById('seizureQuickList');
const cardChannelBadge = document.getElementById('cardChannelBadge');
const cardLobeBadge = document.getElementById('cardLobeBadge');
const cardChannelDesc = document.getElementById('cardChannelDesc');
const curSignalChannel = document.getElementById('curSignalChannel');
const crosshairInfo = document.getElementById('crosshairInfo');

// File Upload Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const btnBrowse = document.getElementById('btnBrowse');
const uploadedFilesList = document.getElementById('uploadedFilesList');

// Pipeline Action Buttons
const btnRunClean = document.getElementById('btnRunClean');
const btnRunExtract = document.getElementById('btnRunExtract');
const btnRunFullPipeline = document.getElementById('btnRunFullPipeline');
const btnResetPipeline = document.getElementById('btnResetPipeline');

// Stepper items
const stepperItems = document.querySelectorAll('.stepper-item');

// Toolbar buttons
const btnZoomIn = document.getElementById('btnZoomIn');
const btnZoomOut = document.getElementById('btnZoomOut');
const btnResetZoom = document.getElementById('btnResetZoom');
const btnZoomCrisis1 = document.getElementById('btnZoomCrisis1');
const btnZoomCrisis8 = document.getElementById('btnZoomCrisis8');

// Canvases
const eegCanvas = document.getElementById('eegCanvas');
const eegCtx = eegCanvas.getContext('2d');
const fftCanvas = document.getElementById('fftCanvas');
const fftCtx = fftCanvas.getContext('2d');

// Results elements
const resModelName = document.getElementById('resModelName');
const resChannelName = document.getElementById('resChannelName');
const resInterpretationText = document.getElementById('resInterpretationText');
const valRecall = document.getElementById('valRecall');
const barRecall = document.getElementById('barRecall');
const valF1 = document.getElementById('valF1');
const barF1 = document.getElementById('barF1');
const valPrecision = document.getElementById('valPrecision');
const barPrecision = document.getElementById('barPrecision');
const valAccuracy = document.getElementById('valAccuracy');
const barAccuracy = document.getElementById('barAccuracy');
const cmVN = document.getElementById('cmVN');
const cmFP = document.getElementById('cmFP');
const cmFN = document.getElementById('cmFN');
const cmVP = document.getElementById('cmVP');
const cmClinicalComment = document.getElementById('cmClinicalComment');
const featureImportanceList = document.getElementById('featureImportanceList');
const detectedSeizuresBody = document.getElementById('detectedSeizuresBody');
const toastContainer = document.getElementById('toastContainer');

// ==========================================================================
// Toast Notification Utility
// ==========================================================================
function showToast(message, icon = 'ℹ️') {
  if (!toastContainer) return;
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 400);
  }, 4000);
}

// ==========================================================================
// Initialization
// ==========================================================================
async function init() {
  setupCanvasDPI(eegCanvas, eegCtx);
  setupCanvasDPI(fftCanvas, fftCtx);
  setupEventListeners();

  try {
    // 1. Fetch info & datasets
    await loadDatasetsList();
    const res = await fetch('/api/info');
    appState.info = await res.json();
    appState.totalDuration = appState.info.total_duration;
    appState.viewRange.end = appState.totalDuration;

    document.getElementById('dispPatientId').textContent = appState.info.patient_id;
    if (document.getElementById('dispDebutTrace') && appState.info.debut_trace) {
      document.getElementById('dispDebutTrace').textContent = appState.info.debut_trace;
    }
    document.getElementById('dispDuration').textContent = `${(appState.totalDuration / 60).toFixed(1)} min (${appState.totalDuration.toFixed(0)}s)`;
    document.getElementById('dispNbCrises').textContent = `${appState.info.crises.length} enregistrées`;

    populateChannels();
    populateQuickSeizures();
    updateChannelCard();

    // 2. Initial signal load
    await loadSignal();
    
    // 3. Initial optimized training (Multi-channels, SMOTE, Temporal filter)
    await runTraining();
  } catch (err) {
    console.error("Erreur d'initialisation:", err);
  }
}

function setupCanvasDPI(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
}

window.addEventListener('resize', () => {
  setupCanvasDPI(eegCanvas, eegCtx);
  setupCanvasDPI(fftCanvas, fftCtx);
  renderSignalCanvas();
  if (appState.trainResult) renderFFTCanvas(appState.trainResult.fft_spectra);
});

// ==========================================================================
// Datasets & Files Handling
// ==========================================================================
async function loadDatasetsList() {
  try {
    const res = await fetch('/api/datasets');
    if (!res.ok) return;
    appState.datasets = await res.json();
    if (datasetSelect && appState.datasets.length > 0) {
      datasetSelect.innerHTML = '';
      appState.datasets.forEach(ds => {
        const opt = document.createElement('option');
        opt.value = ds.patient_id;
        opt.textContent = `Patient ${ds.patient_id} (${ds.size_mb} Mo) — ${ds.signal_file}`;
        if (ds.is_active) opt.selected = true;
        datasetSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn("Impossible de charger les datasets:", e);
  }
}

async function uploadFile(file) {
  showToast(`Envoi de ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} Mo)...`, '⏳');
  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      headers: {
        'X-Filename': file.name
      },
      body: file
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Fichier ${file.name} réceptionné dans data/ !`, '✓');
      addUploadedFilePill(file.name, `${(file.size / (1024*1024)).toFixed(1)} Mo`);
      await loadDatasetsList();
    } else {
      showToast(`Erreur d'envoi: ${data.error || 'Échec'}`, '⚠️');
    }
  } catch (err) {
    showToast(`Erreur réseau lors de l'upload`, '❌');
    console.error(err);
  }
}

function addUploadedFilePill(name, sizeStr) {
  if (!uploadedFilesList) return;
  const pill = document.createElement('div');
  pill.className = 'upload-file-pill';
  pill.innerHTML = `
    <span class="file-name">${name}</span>
    <span class="file-size">${sizeStr}</span>
  `;
  uploadedFilesList.appendChild(pill);
}

// ==========================================================================
// Event Listeners
// ==========================================================================
function setupEventListeners() {
  // Stepper items click to scroll
  stepperItems.forEach(item => {
    item.addEventListener('click', () => {
      stepperItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const step = item.getAttribute('data-step');
      const targetSec = document.getElementById(`sectionStep${step}`);
      if (targetSec) {
        targetSec.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else if (step === '5') {
        const resSec = document.getElementById('resultsSection');
        if (resSec) resSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Drag & drop handling
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          uploadFile(files[i]);
        }
      }
    });

    dropZone.addEventListener('click', () => {
      if (fileInput) fileInput.click();
    });
  }

  if (btnBrowse && fileInput) {
    btnBrowse.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        for (let i = 0; i < fileInput.files.length; i++) {
          uploadFile(fileInput.files[i]);
        }
      }
    });
  }

  // Dataset switch
  if (datasetSelect) {
    datasetSelect.addEventListener('change', async () => {
      showToast(`Chargement du patient ${datasetSelect.value}...`, '🔄');
      await executeCleaning();
    });
  }

  // Channel & Model selectors
  channelSelect.addEventListener('change', async (e) => {
    appState.currentChannel = e.target.value;
    updateChannelCard();
    await loadSignal();
  });

  modelSelect.addEventListener('change', (e) => {
    appState.currentModel = e.target.value;
  });

  lobeFilters.addEventListener('click', (e) => {
    if (e.target.classList.contains('chip')) {
      document.querySelectorAll('#lobeFilters .chip').forEach(c => c.classList.remove('active'));
      e.target.classList.add('active');
      appState.activeLobe = e.target.getAttribute('data-lobe');
      populateChannels();
    }
  });

  // Step Action buttons
  if (btnRunClean) {
    btnRunClean.addEventListener('click', executeCleaning);
  }
  if (btnRunExtract) {
    btnRunExtract.addEventListener('click', executeFeatureExtraction);
  }
  if (btnTrain) {
    btnTrain.addEventListener('click', runTraining);
  }
  if (btnRunFullPipeline) {
    btnRunFullPipeline.addEventListener('click', executeFullPipeline);
  }
  if (btnResetPipeline) {
    btnResetPipeline.addEventListener('click', async () => {
      try {
        await fetch('/api/pipeline/reset', { method: 'POST' });
        showToast("Cache du pipeline réinitialisé !", '🔄');
      } catch (e) {
        console.error(e);
      }
    });
  }

  // Zoom toolbar
  btnZoomIn.addEventListener('click', () => zoomCenter(0.5));
  btnZoomOut.addEventListener('click', () => zoomCenter(2.0));
  btnResetZoom.addEventListener('click', () => {
    appState.viewRange = { start: 0, end: appState.totalDuration };
    loadSignal();
  });

  if (btnZoomCrisis1) {
    btnZoomCrisis1.addEventListener('click', () => {
      appState.viewRange = { start: 0, end: 25 };
      loadSignal();
      showToast("Zoom sur Crise n°1 (0s)", '🔍');
    });
  }

  if (btnZoomCrisis8) {
    btnZoomCrisis8.addEventListener('click', () => {
      appState.viewRange = { start: 1450, end: 1485 };
      loadSignal();
      showToast("Zoom sur Crise n°8 (1460s - Absence 1465.5s)", '🔍');
    });
  }

  // Canvas Mouse Interactions
  eegCanvas.addEventListener('wheel', handleCanvasWheel, { passive: false });
  eegCanvas.addEventListener('mousedown', handleCanvasMouseDown);
  window.addEventListener('mousemove', handleCanvasMouseMove);
  window.addEventListener('mouseup', handleCanvasMouseUp);
}

// ==========================================================================
// Pipeline Step Execution Handlers
// ==========================================================================
async function executeCleaning() {
  showToast("Nettoyage & Synchronisation temporelle en cours...", '🧹');
  try {
    const selectedPid = datasetSelect ? datasetSelect.value : null;
    const res = await fetch('/api/pipeline/clean', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: selectedPid })
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Nettoyé : DébutTracé ${data.debut_trace}, ${data.nb_crises} crises synchronisées (${data.total_crisis_duration}s)`, '✓');
      appState.totalDuration = data.total_duration;
      appState.viewRange.end = data.total_duration;
      document.getElementById('dispDebutTrace').textContent = data.debut_trace;
      document.getElementById('dispDuration').textContent = `${(data.total_duration / 60).toFixed(1)} min`;
      document.getElementById('dispNbCrises').textContent = `${data.nb_crises} enregistrées`;
      await loadSignal();
    } else {
      showToast(`Erreur de nettoyage: ${data.error}`, '⚠️');
    }
  } catch (err) {
    showToast("Erreur lors du nettoyage", '❌');
    console.error(err);
  }
}

async function executeFeatureExtraction() {
  showToast("Extraction des 294 descripteurs spatiotemporels & ratios...", '🧬');
  try {
    const res = await fetch('/api/pipeline/extract', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      showToast(`${data.nb_features} features extraites sur ${data.total_windows} fenêtres (${data.crisis_windows} crises, ratio ${data.imbalance_ratio})`, '✓');
      if (document.getElementById('dispNbFeatures')) {
        document.getElementById('dispNbFeatures').textContent = `${data.nb_features} variables`;
      }
    } else {
      showToast(`Erreur: ${data.error}`, '⚠️');
    }
  } catch (err) {
    showToast("Erreur lors de l'extraction", '❌');
    console.error(err);
  }
}

async function executeFullPipeline() {
  showToast("Exécution complète du pipeline en 1-clic démarrée...", '⚡');
  await executeCleaning();
  await executeFeatureExtraction();
  await runTraining();
  showToast("Pipeline complet exécuté avec succès ! Diagnostic clinique à jour.", '🎉');
}

// ==========================================================================
// Populate Channel and Quick Seizures
// ==========================================================================
function populateChannels() {
  if (!appState.info) return;
  channelSelect.innerHTML = '';

  // Multi-channel option always on top
  const optAll = document.createElement('option');
  optAll.value = 'ALL';
  optAll.textContent = '⭐ TOUS LES CANAUX SIMULTANÉS (294 features) [Recommandé]';
  channelSelect.appendChild(optAll);

  const filtered = appState.info.channels.filter(ch => {
    if (appState.activeLobe === 'ALL') return true;
    return ch.lobe.toLowerCase().includes(appState.activeLobe.toLowerCase());
  });

  filtered.forEach(ch => {
    const opt = document.createElement('option');
    opt.value = ch.name;
    opt.textContent = `${ch.name} (${ch.lobe})`;
    channelSelect.appendChild(opt);
  });

  channelSelect.value = appState.currentChannel;
}

function updateChannelCard() {
  if (!appState.info) return;
  
  if (appState.currentChannel === 'ALL') {
    cardChannelBadge.textContent = 'ALL (21 canaux)';
    cardLobeBadge.textContent = 'Multi-Canaux Global';
    cardChannelDesc.textContent = 'Modèle multi-électrodes combinant 21 canaux (19 EEG standard + SLI + ECG) avec 294 variables spatiotemporelles et spectrales.';
    curSignalChannel.textContent = 'EEGFp1 (Canal pilote)';
    return;
  }

  const ch = appState.info.channels.find(c => c.name === appState.currentChannel);
  if (ch) {
    cardChannelBadge.textContent = ch.name;
    cardLobeBadge.textContent = ch.lobe;
    cardChannelDesc.textContent = ch.description;
    curSignalChannel.textContent = ch.name;
  }
}

function populateQuickSeizures() {
  if (!appState.info || !appState.info.crises) return;
  seizureQuickList.innerHTML = '';

  appState.info.crises.forEach(c => {
    const btn = document.createElement('button');
    btn.className = 'seizure-btn';
    btn.innerHTML = `
      <div class="seizure-num">Crise ${c.Crise}</div>
      <div class="seizure-times">
        ${c.Debut.toFixed(0)}s – ${c.Fin.toFixed(0)}s (${c.Duree.toFixed(0)}s)
        <br><span style="color:var(--text-dim);font-size:0.68rem;">Horaires: ${c.Debut_Horaire} ➔ ${c.Fin_Horaire}</span>
      </div>
    `;
    btn.addEventListener('click', () => {
      const margin = 10;
      appState.viewRange.start = Math.max(0, c.Debut - margin);
      appState.viewRange.end = Math.min(appState.totalDuration, c.Fin + margin);
      loadSignal();
      showToast(`Navigation vers Crise n°${c.Crise} (${c.Debut_Horaire})`, '🎯');
    });
    seizureQuickList.appendChild(btn);
  });
}

// ==========================================================================
// Signal Fetching & Canvas Rendering
// ==========================================================================
async function loadSignal() {
  try {
    const ch = (appState.currentChannel === 'ALL') ? 'EEGFp1' : appState.currentChannel;
    const start = appState.viewRange.start;
    const end = appState.viewRange.end || appState.totalDuration;

    const url = `/api/signal?channel=${ch}&start=${start}&end=${end}&max_points=2500`;
    const res = await fetch(url);
    appState.signalData = await res.json();
    renderSignalCanvas();
  } catch (err) {
    console.error("Erreur de chargement du signal:", err);
  }
}

function renderSignalCanvas() {
  if (!appState.signalData || !appState.signalData.signal) return;

  const rect = eegCanvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  eegCtx.clearRect(0, 0, w, h);

  // Background grid
  drawCanvasGrid(w, h);

  const times = appState.signalData.time;
  const sig = appState.signalData.signal;
  const crises = appState.signalData.crises || [];
  const minT = times[0];
  const maxT = times[times.length - 1];
  const rangeT = maxT - minT || 1;

  // Amplitude scaling
  let minV = -150, maxV = 150;
  if (sig.length > 0) {
    const sMin = Math.min(...sig);
    const sMax = Math.max(...sig);
    const pad = (sMax - sMin) * 0.1 || 20;
    minV = sMin - pad;
    maxV = sMax + pad;
  }
  const rangeV = maxV - minV || 1;

  // 1. Shading: Real Ground-truth crises (Green / Emerald)
  crises.forEach(c => {
    const x1 = Math.max(0, ((c.Debut - minT) / rangeT) * w);
    const x2 = Math.min(w, ((c.Fin - minT) / rangeT) * w);
    if (x2 > x1) {
      eegCtx.fillStyle = 'rgba(16, 185, 129, 0.22)';
      eegCtx.fillRect(x1, 0, x2 - x1, h);

      // Boundary lines
      eegCtx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
      eegCtx.lineWidth = 1.5;
      eegCtx.strokeRect(x1, 0, x2 - x1, h);

      // Label
      eegCtx.fillStyle = '#10b981';
      eegCtx.font = '10px JetBrains Mono';
      eegCtx.fillText(`Crise ${c.Crise} (${c.Duree}s)`, x1 + 4, 16);
    }
  });

  // 2. Shading: Model predictions (Crimson / Magenta)
  if (appState.trainResult && appState.trainResult.detected_crises) {
    appState.trainResult.detected_crises.forEach(p => {
      if (p.end >= minT && p.start <= maxT) {
        const x1 = Math.max(0, ((p.start - minT) / rangeT) * w);
        const x2 = Math.min(w, ((p.end - minT) / rangeT) * w);
        if (x2 > x1) {
          eegCtx.fillStyle = 'rgba(255, 51, 102, 0.20)';
          eegCtx.fillRect(x1, h - 28, x2 - x1, 28);
          eegCtx.strokeStyle = '#ff3366';
          eegCtx.lineWidth = 1.2;
          eegCtx.strokeRect(x1, h - 28, x2 - x1, 28);

          eegCtx.fillStyle = '#ff3366';
          eegCtx.font = '9px JetBrains Mono';
          eegCtx.fillText(`Pred ${p.id}`, x1 + 3, h - 10);
        }
      }
    });
  }

  // 3. Draw EEG Waveform line
  eegCtx.beginPath();
  eegCtx.strokeStyle = '#00f2fe';
  eegCtx.lineWidth = 1.4;

  for (let i = 0; i < times.length; i++) {
    const x = ((times[i] - minT) / rangeT) * w;
    const y = h - ((sig[i] - minV) / rangeV) * h;
    if (i === 0) eegCtx.moveTo(x, y);
    else eegCtx.lineTo(x, y);
  }
  eegCtx.stroke();

  // Draw time bounds text
  eegCtx.fillStyle = '#64748b';
  eegCtx.font = '11px JetBrains Mono';
  eegCtx.fillText(`${minT.toFixed(1)} s`, 8, h - 8);
  const endTxt = `${maxT.toFixed(1)} s`;
  const endWidth = eegCtx.measureText(endTxt).width;
  eegCtx.fillText(endTxt, w - endWidth - 8, h - 8);
}

function drawCanvasGrid(w, h) {
  eegCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  eegCtx.lineWidth = 1;

  // Vertical time grid lines
  const numVLines = 8;
  for (let i = 1; i < numVLines; i++) {
    const x = (w / numVLines) * i;
    eegCtx.beginPath();
    eegCtx.moveTo(x, 0);
    eegCtx.lineTo(x, h);
    eegCtx.stroke();
  }

  // Horizontal voltage zero line
  eegCtx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
  eegCtx.beginPath();
  eegCtx.moveTo(0, h / 2);
  eegCtx.lineTo(w, h / 2);
  eegCtx.stroke();
}

// ==========================================================================
// Zoom & Pan Handlers
// ==========================================================================
function zoomCenter(factor) {
  const currentStart = appState.viewRange.start;
  const currentEnd = appState.viewRange.end || appState.totalDuration;
  const currentSpan = currentEnd - currentStart;
  const center = (currentStart + currentEnd) / 2;

  const newSpan = Math.max(5, Math.min(appState.totalDuration, currentSpan * factor));
  let newStart = center - newSpan / 2;
  let newEnd = center + newSpan / 2;

  if (newStart < 0) {
    newStart = 0;
    newEnd = newSpan;
  }
  if (newEnd > appState.totalDuration) {
    newEnd = appState.totalDuration;
    newStart = Math.max(0, newEnd - newSpan);
  }

  appState.viewRange.start = newStart;
  appState.viewRange.end = newEnd;
  loadSignal();
}

function handleCanvasWheel(e) {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 1.3 : 0.7;
  zoomCenter(factor);
}

function handleCanvasMouseDown(e) {
  appState.isDragging = true;
  appState.dragStartX = e.clientX;
  appState.dragStartRange = { ...appState.viewRange };
  eegCanvas.style.cursor = 'grabbing';
}

function handleCanvasMouseMove(e) {
  const rect = eegCanvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  if (mouseX >= 0 && mouseX <= rect.width && mouseY >= 0 && mouseY <= rect.height) {
    const startT = appState.viewRange.start;
    const endT = appState.viewRange.end || appState.totalDuration;
    const curT = startT + (mouseX / rect.width) * (endT - startT);
    crosshairInfo.textContent = `Temps: ${curT.toFixed(1)}s (${(curT/60).toFixed(1)}min)`;
  }

  if (!appState.isDragging) return;
  const deltaX = e.clientX - appState.dragStartX;
  const span = appState.dragStartRange.end - appState.dragStartRange.start;
  const timeShift = -(deltaX / rect.width) * span;

  let newStart = appState.dragStartRange.start + timeShift;
  let newEnd = appState.dragStartRange.end + timeShift;

  if (newStart < 0) {
    newStart = 0;
    newEnd = span;
  }
  if (newEnd > appState.totalDuration) {
    newEnd = appState.totalDuration;
    newStart = appState.totalDuration - span;
  }

  appState.viewRange.start = Math.max(0, newStart);
  appState.viewRange.end = Math.min(appState.totalDuration, newEnd);
  renderSignalCanvas();
}

function handleCanvasMouseUp() {
  if (appState.isDragging) {
    appState.isDragging = false;
    eegCanvas.style.cursor = 'crosshair';
    loadSignal();
  }
}

// ==========================================================================
// Machine Learning Model Training & Evaluation
// ==========================================================================
async function runTraining() {
  btnTrain.disabled = true;
  trainSpinner.style.display = 'inline-block';

  const chkSMOTE = document.getElementById('chkSMOTE');
  const chkTemporal = document.getElementById('chkTemporal');
  const useSmote = chkSMOTE ? chkSMOTE.checked : true;
  const useTemporal = chkTemporal ? chkTemporal.checked : true;

  try {
    const payload = {
      channel: appState.currentChannel,
      model: appState.currentModel,
      use_smote: useSmote,
      use_temporal_filter: useTemporal
    };

    const res = await fetch('/api/pipeline/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (res.ok) {
      appState.trainResult = data;
      renderEvaluationDashboard(data);
      renderSignalCanvas();
      showToast(`Modèle ${data.model_name} entraîné avec succès (F1: ${data.metrics.f1}%)`, '⚡');
    } else {
      showToast(`Erreur d'entraînement: ${data.error}`, '❌');
    }
  } catch (err) {
    console.error("Erreur d'entraînement:", err);
    showToast("Erreur lors de l'entraînement", '❌');
  } finally {
    btnTrain.disabled = false;
    trainSpinner.style.display = 'none';
  }
}

function renderEvaluationDashboard(data) {
  resModelName.textContent = data.model_name;
  resChannelName.textContent = data.channel === 'ALL' ? 'Multi-Canaux (294 features)' : `${data.channel} (${data.lobe})`;
  
  if (data.is_multichannel) {
    resInterpretationText.textContent = `Modèle global entraîné sur l'ensemble des 21 canaux (294 descripteurs avec ratios SDR & SMOTE). Post-filtrage temporel actif.`;
  } else {
    resInterpretationText.textContent = `Modèle mono-électrode entraîné sur ${data.channel}. ${data.description}`;
  }

  // Animate KPI metrics
  animateValue(valRecall, data.metrics.recall, '%');
  barRecall.style.width = `${Math.min(100, data.metrics.recall)}%`;

  animateValue(valF1, data.metrics.f1, '%');
  barF1.style.width = `${Math.min(100, data.metrics.f1)}%`;

  animateValue(valPrecision, data.metrics.precision, '%');
  barPrecision.style.width = `${Math.min(100, data.metrics.precision)}%`;

  animateValue(valAccuracy, data.metrics.accuracy, '%');
  barAccuracy.style.width = `${Math.min(100, data.metrics.accuracy)}%`;

  // Confusion matrix
  const cm = data.confusion_matrix;
  cmVN.textContent = cm.vn;
  cmFP.textContent = cm.fp;
  cmFN.textContent = cm.fn;
  cmVP.textContent = cm.vp;

  if (cm.fp <= 2) {
    cmClinicalComment.innerHTML = `<span style="color:var(--emerald);font-weight:600;">Éradication des fausses alertes :</span> Seulement <strong>${cm.fp} faux positifs</strong> observés sur les 624 fenêtres de test grâce au post-filtrage temporel et au rééquilibrage SMOTE.`;
  } else {
    cmClinicalComment.textContent = `${cm.fp} faux positifs et ${cm.fn} faux négatifs sur le jeu de test indépendant.`;
  }

  // Render FFT Spectra
  renderFFTCanvas(data.fft_spectra);

  // Populate Feature Importances
  featureImportanceList.innerHTML = '';
  const entries = Object.entries(data.feature_importances);
  if (entries.length > 0) {
    const maxVal = Math.max(...entries.map(e => e[1])) || 1;
    entries.forEach(([name, val]) => {
      const item = document.createElement('div');
      item.className = 'feat-item';
      item.innerHTML = `
        <div class="feat-labels">
          <span class="feat-name">${name}</span>
          <span class="feat-val">${(val * 100).toFixed(1)}%</span>
        </div>
        <div class="feat-bar-bg">
          <div class="feat-bar-fill" style="width: ${(val / maxVal) * 100}%;"></div>
        </div>
      `;
      featureImportanceList.appendChild(item);
    });
  } else {
    featureImportanceList.innerHTML = `<p class="analytics-note">L'algorithme ${data.model_name} ne produit pas de poids explicites directs (basé sur des distances/noyaux SVM ou KNN).</p>`;
  }

  // Populate Detected Seizures Table
  detectedSeizuresBody.innerHTML = '';
  if (data.detected_crises.length === 0) {
    detectedSeizuresBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-dim);">Aucune crise détectée sur ce canal.</td></tr>`;
  } else {
    data.detected_crises.forEach(p => {
      const matched = data.real_crises.some(c => p.start <= c.Fin && p.end >= c.Debut);
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${p.id}</td>
        <td>${p.start.toFixed(1)} s</td>
        <td>${p.end.toFixed(1)} s</td>
        <td>${p.duration.toFixed(1)} s</td>
        <td>
          <span class="${matched ? 'badge-match' : 'badge-miss'}">
            ${matched ? '✓ Validée' : '⚠️ Non documentée'}
          </span>
        </td>
      `;
      detectedSeizuresBody.appendChild(row);
    });
  }
}

function renderFFTCanvas(fft) {
  if (!fft || !fft.freqs) return;

  const rect = fftCanvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  fftCtx.clearRect(0, 0, w, h);

  const freqs = fft.freqs;
  const sain = fft.sain;
  const crise = fft.crise;

  const maxPower = Math.max(...crise, ...sain, 100);

  // 3 Hz vertical reference line
  const x3Hz = (3.0 / 35.0) * w;
  fftCtx.strokeStyle = 'rgba(255, 51, 102, 0.45)';
  fftCtx.lineWidth = 1;
  fftCtx.setLineDash([3, 3]);
  fftCtx.beginPath();
  fftCtx.moveTo(x3Hz, 0);
  fftCtx.lineTo(x3Hz, h);
  fftCtx.stroke();
  fftCtx.setLineDash([]);

  fftCtx.fillStyle = '#ff3366';
  fftCtx.font = '9px JetBrains Mono';
  fftCtx.fillText('3 Hz (Pic)', x3Hz + 4, 14);

  // Sain Curve (Blue)
  fftCtx.beginPath();
  fftCtx.strokeStyle = '#3b82f6';
  fftCtx.lineWidth = 1.2;
  for (let i = 0; i < freqs.length; i++) {
    const x = (freqs[i] / 35.0) * w;
    const y = h - (sain[i] / maxPower) * (h - 20) - 10;
    if (i === 0) fftCtx.moveTo(x, y);
    else fftCtx.lineTo(x, y);
  }
  fftCtx.stroke();

  // Crise Curve (Red)
  fftCtx.beginPath();
  fftCtx.strokeStyle = '#ef4444';
  fftCtx.lineWidth = 1.8;
  for (let i = 0; i < freqs.length; i++) {
    const x = (freqs[i] / 35.0) * w;
    const y = h - (crise[i] / maxPower) * (h - 20) - 10;
    if (i === 0) fftCtx.moveTo(x, y);
    else fftCtx.lineTo(x, y);
  }
  fftCtx.stroke();
}

function animateValue(elem, targetVal, suffix = '') {
  let start = 0;
  const duration = 500;
  const startTime = performance.now();

  function update(time) {
    const elapsed = time - startTime;
    const progress = Math.min(1, elapsed / duration);
    const cur = (start + (targetVal - start) * progress).toFixed(1);
    elem.textContent = `${cur}${suffix}`;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Start application
window.addEventListener('DOMContentLoaded', init);
