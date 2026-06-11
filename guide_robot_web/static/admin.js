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

let currentMode = 'en';

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
    setStatus('Running database command...');
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

modeButtons.forEach((button) => {
    button.addEventListener('click', async () => {
        const language = button.dataset.mode;
        renderMode(language);
        setStatus('Switching mode...');

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

    setStatus('Saving Kaggle URL...');
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
