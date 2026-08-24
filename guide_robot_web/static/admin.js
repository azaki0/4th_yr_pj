const modeButtons = document.querySelectorAll('.mode-button');
const remoteForm = document.getElementById('remote-form');
const remoteUrl = document.getElementById('remote-url');
const promptForm = document.getElementById('prompt-form');
const promptInput = document.getElementById('prompt-input');
const databaseStatus = document.getElementById('database-status');
const databaseButtons = document.querySelectorAll('[data-db-command]');
const manualMemoryForm = document.getElementById('manual-memory-form');
const manualMemory = document.getElementById('manual-memory');
const adminStatus = document.getElementById('admin-status');
const loadModelsBtn = document.getElementById('load-models-btn');
const modelLoadStatus = document.getElementById('model-load-status');
const latencyLogs = document.getElementById('latency-logs');
const latencyCount = document.getElementById('latency-count');
const clearLatencyBtn = document.getElementById('clear-latency-btn');

let currentMode = 'en';
let latencyPollInterval = null;

function setStatus(message) {
    adminStatus.textContent = message;
}

function renderMode(mode) {
    currentMode = mode;
    modeButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.mode === mode);
    });
}

async function loadMode() {
    const response = await fetch('/api/mode');
    const data = await response.json();
    renderMode(data.language || 'en');
}

async function loadRemoteConfig() {
    const response = await fetch('/api/remote');
    const data = await response.json();
    remoteUrl.value = data.url || '';
}

function renderDatabaseStatus(data) {
    databaseStatus.textContent = `${data.uniInfoChunks ?? 0} uni chunks | ${data.conversations ?? 0} memories`;
}

async function loadDatabaseStatus() {
    const response = await fetch('/api/admin/db');
    const data = await response.json();
    renderDatabaseStatus(data);
}

async function runDatabaseCommand(command, text = '') {
    setStatus('Running database command');
    const response = await fetch('/api/admin/db', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, text }),
    });
    const data = await response.json();

    if (!response.ok) {
        setStatus(data.error || 'Database command failed.');
        return;
    }

    renderDatabaseStatus(data.status || {});
    setStatus('Database command complete.');
}

function formatModelResult(label, result) {
    if (!result) return `${label}: no response`;
    if (result.ok) {
        const loaded = (result.loaded || []).join(', ') || 'ready';
        return `${label}: ${loaded}`;
    }

    const errorText = result.error || Object.values(result.errors || {}).join('; ') || 'failed';
    return `${label}: ${errorText}`;
}

modeButtons.forEach((button) => {
    button.addEventListener('click', async () => {
        const language = button.dataset.mode;
        renderMode(language);
        setStatus('Switching mode');

        const response = await fetch('/api/mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language }),
        });

        if (!response.ok) {
            setStatus('Could not switch mode.');
            return;
        }

        setStatus(language === 'mm' ? 'Myanmar mode active.' : 'English mode active.');
    });
});

remoteForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    setStatus('Saving Kaggle URL');
    const response = await fetch('/api/remote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: remoteUrl.value.trim(), token: '' }),
    });

    if (!response.ok) {
        setStatus('Could not save Kaggle URL.');
        return;
    }

    setStatus('Kaggle URL saved.');
});

loadModelsBtn.addEventListener('click', async () => {
    loadModelsBtn.disabled = true;
    modelLoadStatus.textContent = 'Loading local and Kaggle models...';
    setStatus('Loading models');

    try {
        const response = await fetch('/api/admin/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        const data = await response.json();

        modelLoadStatus.textContent = [
            formatModelResult('Local', data.local),
            formatModelResult('Voice', data.voice),
            formatModelResult('Kaggle', data.kaggle),
        ].join(' | ');

        if (!response.ok || !data.ok) {
            setStatus(data.error || 'Some models failed to load.');
            return;
        }

        setStatus('Models loaded.');
    } catch (error) {
        modelLoadStatus.textContent = 'Model load request failed.';
        setStatus('Model load request failed.');
    } finally {
        loadModelsBtn.disabled = false;
    }
});

databaseButtons.forEach((button) => {
    button.addEventListener('click', () => {
        const command = button.dataset.dbCommand;
        if (command === 'clear_conversations') {
            const confirmed = window.confirm('Clear all conversation memory?');
            if (!confirmed) return;
        }
        runDatabaseCommand(command);
    });
});

manualMemoryForm.addEventListener('submit', (event) => {
    event.preventDefault();

    const text = manualMemory.value.trim();
    if (!text) {
        setStatus('Type a memory first.');
        return;
    }

    runDatabaseCommand('add_memory', text);
    manualMemory.value = '';
});

promptForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus('Type a prompt first.');
        return;
    }

    setStatus('Sending...');
    const response = await fetch('/api/prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
        setStatus('Could not send prompt.');
        return;
    }

    promptInput.value = '';
    setStatus(`Queued in ${currentMode === 'mm' ? 'Myanmar' : 'English'} mode.`);
});

Promise.all([loadMode(), loadRemoteConfig(), loadDatabaseStatus()]).catch(() => setStatus('Could not load admin settings.'));

// Latency Logs
async function loadLatencyLogs() {
    try {
        const response = await fetch('/api/admin/latency');
        const data = await response.json();
        renderLatencyLogs(data.logs || []);
        if (latencyCount) {
            latencyCount.textContent = `${data.logs?.length || 0} requests`;
        }
    } catch (error) {
        console.error('Failed to load latency logs:', error);
        if (latencyLogs) {
            latencyLogs.textContent = 'Error loading latency logs';
        }
    }
}

function ttfrClass(ms) {
    if (ms < 2000) return 'ttfr-fast';
    if (ms < 5000) return 'ttfr-ok';
    return 'ttfr-slow';
}

function renderLatencyLogs(logs) {
    if (!latencyLogs) return;

    if (!logs || logs.length === 0) {
        latencyLogs.innerHTML = '<span style="color: var(--text-dim)">No latency data yet. Send a prompt to the robot to generate logs.</span>';
        return;
    }

    // Show latest first
    const reversedLogs = [...logs].reverse();
    const latest = reversedLogs[0];

    // Build summary stat card for latest TTFR
    const summaryHtml = `
        <div class="ttfr-summary">
            <div class="ttfr-label">Time to First Response (latest)</div>
            <div class="ttfr-value ${ttfrClass(latest.ttfr_ms || 0)}">${(latest.ttfr_ms || 0).toFixed(0)}<span class="ttfr-unit">ms</span></div>
        </div>
    `;

    const entriesHtml = reversedLogs.map(log => {
        const time = new Date(log.timestamp * 1000).toLocaleTimeString();
        const promptPreview = log.prompt || 'unknown';
        const ttfr = log.ttfr_ms || 0;
        return `
            <div class="request-header">
                [${time}] <span class="${ttfrClass(ttfr)}">${ttfr.toFixed(0)}ms</span> ttfr | ${log.total_ms.toFixed(0)}ms total | ${log.language} | "${promptPreview}"
            </div>
            ${formatStages(log.stages?.children || [], 0)}
        `;
    }).join('<hr style="border-color: var(--border); margin: 12px 0;">');

    latencyLogs.innerHTML = summaryHtml + entriesHtml;
}

function formatStages(stages, depth) {
    if (!stages || stages.length === 0) return '';

    const depthClass = depth === 0 ? 'stage-root' : depth === 1 ? 'stage-child' : 'stage-grandchild';
    const indent = '  '.repeat(depth);

    return stages.map(stage => {
        const metaHtml = stage.metadata && Object.keys(stage.metadata).length > 0
            ? `<span class="meta"> [${Object.entries(stage.metadata).map(([k,v]) => `${k}:${v}`).join(', ')}]</span>`
            : '';

        const childrenHtml = stage.children && stage.children.length > 0
            ? formatStages(stage.children, depth + 1)
            : '';

        return `<div class="${depthClass}">${indent}${stage.name}: <span class="duration">${stage.duration_ms.toFixed(1)}ms</span>${metaHtml}</div>${childrenHtml}`;
    }).join('');
}

async function clearLatencyLogs() {
    try {
        await fetch('/api/admin/latency', { method: 'DELETE' });
        latencyLogs.textContent = 'Logs cleared. Send a prompt to generate new logs.';
        if (latencyCount) latencyCount.textContent = '0 requests';
        setStatus('Latency logs cleared');
    } catch (error) {
        console.error('Failed to clear latency logs:', error);
    }
}

if (clearLatencyBtn) {
    clearLatencyBtn.addEventListener('click', clearLatencyLogs);
}

// Auto-refresh latency logs every 3 seconds when admin panel is visible
function startLatencyPolling() {
    if (latencyPollInterval) return;
    loadLatencyLogs();
    latencyPollInterval = setInterval(loadLatencyLogs, 3000);
}

function stopLatencyPolling() {
    if (latencyPollInterval) {
        clearInterval(latencyPollInterval);
        latencyPollInterval = null;
    }
}

// Start polling when page loads
document.addEventListener('DOMContentLoaded', () => {
    startLatencyPolling();
});

// Stop polling when page unloads
window.addEventListener('beforeunload', stopLatencyPolling);
