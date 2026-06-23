const output = document.getElementById('stream-output');
const displayArea = document.getElementById('display-area');
const routeView = document.getElementById('route-view');
const routeLine = document.getElementById('route-line');
const routeMarkers = document.getElementById('route-markers');
const routeStart = document.getElementById('route-start');
const destinationName = document.getElementById('destination-name');
const routeDistance = document.getElementById('route-distance');
const routeNarration = document.getElementById('route-narration');
const mapDistance = document.getElementById('map-distance');
const mapTime = document.getElementById('map-time');
const routeHeading = document.querySelector('.direction-panel h3');
const streamCursor = document.getElementById('stream-cursor');
const routeBackBtn = document.getElementById('route-back-btn');

const micBtn = document.getElementById('mic-btn');
const listeningOverlay = document.getElementById('listening-overlay');
const transcriptPanel = document.getElementById('voice-transcript-panel');
const transcriptList = document.getElementById('transcript-list');
const modeButtons = document.querySelectorAll('.display-mode-button');
const mmControlPanel = document.getElementById('mm-control-panel');
const mmTextForm = document.getElementById('mm-text-form');
const mmTextInput = document.getElementById('mm-text-input');
const mmActionButtons = document.querySelectorAll('[data-mm-action]');

let isNewResponse = true;
let isResponding = false;
let currentLanguage = 'en';

let charQueue = [];
let typeInterval = null;
let currentSpan = null;
let currentScrollEl = null;
let currentTargetEl = null;
const CHAR_DELAY_MS = 18;

//Voice state
let isListening = false;

//Mode check
async function fetchMode() {
    try {
        const r = await fetch('/api/mode');
        const data = await r.json();
        currentLanguage = data.language || 'en';
    } catch {}
    updateMicVisibility();
    renderMode();
}

function renderMode() {
    modeButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.mode === currentLanguage);
    });
    mmControlPanel.classList.toggle('hidden', currentLanguage !== 'mm' || isResponding);
    routeBackBtn.classList.toggle('hidden', currentLanguage !== 'mm' || routeView.classList.contains('hidden'));
}

function updateMicVisibility() {
    if (currentLanguage !== 'en' || isResponding) {
        micBtn.classList.add('hidden');
    } else {
        micBtn.classList.remove('hidden');
    }
    if (mmControlPanel) {
        mmControlPanel.classList.toggle('hidden', currentLanguage !== 'mm' || isResponding);
    }
}

//Transcript
function addTranscript(text) {
    if (!routeView.classList.contains('hidden')) return;
    transcriptPanel.classList.remove('hidden');
    const entry = document.createElement('div');
    entry.className = 'transcript-entry';
    entry.textContent = text;
    transcriptList.appendChild(entry);
    transcriptList.scrollTop = transcriptList.scrollHeight;
}

//Push to talk
micBtn.addEventListener('click', async () => {
    if (isListening || isResponding) return;
    isListening = true;
    micBtn.classList.add('hidden');
    listeningOverlay.classList.remove('hidden');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 36000);
        const r = await fetch('/api/capture-voice', {
            method: 'POST',
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        const data = await r.json();
        if (data.text) {
            addTranscript(data.text);
        }
    } catch {}

    listeningOverlay.classList.add('hidden');
    isListening = false;
    updateMicVisibility();
});

async function setMode(language) {
    currentLanguage = language;
    renderMode();
    updateMicVisibility();

    const response = await fetch('/api/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language }),
    });

    if (!response.ok) {
        currentLanguage = language === 'en' ? 'mm' : 'en';
        renderMode();
        updateMicVisibility();
    }
}

async function sendPrompt(prompt) {
    const clean = prompt.trim();
    if (!clean || isResponding) return;

    isResponding = true;
    updateMicVisibility();

    try {
        const response = await fetch('/api/prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: clean, language: currentLanguage }),
        });

        if (response.ok) return;
    } catch {}

    isResponding = false;
    updateMicVisibility();
}

function showInteractiveHome() {
    routeView.classList.add('hidden');
    displayArea.classList.remove('hidden');
    transcriptPanel.classList.remove('route-mode');
    routeBackBtn.classList.add('hidden');
    updateMicVisibility();
}

modeButtons.forEach((button) => {
    button.addEventListener('click', () => {
        if (button.dataset.mode !== currentLanguage) {
            setMode(button.dataset.mode);
        }
    });
});

mmTextForm.addEventListener('submit', (event) => {
    event.preventDefault();
    sendPrompt(mmTextInput.value);
    mmTextInput.value = '';
});

mmActionButtons.forEach((button) => {
    button.addEventListener('click', () => {
        const action = button.dataset.mmAction;
        if (action === 'navigation') {
            const destination = mmTextInput.value.trim();
            if (!destination) {
                mmTextInput.focus();
                return;
            }
            sendPrompt(`${destination} ကို သွားချင်ပါတယ်။ လမ်းညွှန်ပေးပါ။`);
            mmTextInput.value = '';
        } else if (action === 'overview') {
            sendPrompt('NSPU အကြောင်း အကျဉ်းချုပ် ပြောပြပါ။');
        }
    });
});

routeBackBtn.addEventListener('click', () => {
    showInteractiveHome();
});

// --- Typing engine ---
function startTyping() {
    if (typeInterval) return;
    typeInterval = setInterval(drainQueue, CHAR_DELAY_MS);
}

function drainQueue() {
    if (charQueue.length === 0) {
        clearInterval(typeInterval);
        typeInterval = null;
        if (streamCursor) streamCursor.classList.remove('active');
        return;
    }

    const ch = charQueue.shift();

    if (ch === '\n') {
        currentSpan = null;
    } else {
        if (!currentSpan) {
            currentSpan = document.createElement('span');
            currentSpan.className = 'stream-line';
            currentSpan._target = currentTargetEl || output;
            currentSpan._target.appendChild(currentSpan);
        }
        currentSpan.textContent += ch;
    }

    if (currentScrollEl) {
        currentScrollEl.scrollTop = currentScrollEl.scrollHeight;
    }
}

function enqueueText(text, targetEl, scrollEl) {
    if (currentSpan && currentSpan._target !== targetEl) {
        currentSpan = null;
    }

    if (!currentSpan || currentSpan._target !== targetEl) {
        currentSpan = document.createElement('span');
        currentSpan.className = 'stream-line';
        currentSpan._target = targetEl;
        targetEl.appendChild(currentSpan);
    }

    currentScrollEl = scrollEl;
    currentTargetEl = targetEl;

    for (const ch of text) {
        charQueue.push(ch);
    }

    if (streamCursor) streamCursor.classList.add('active');
    startTyping();
}

function flushQueue() {
    clearInterval(typeInterval);
    typeInterval = null;
    charQueue = [];
    currentSpan = null;
    currentScrollEl = null;
    currentTargetEl = null;
    if (streamCursor) streamCursor.classList.remove('active');
}

// --- Stream ---
async function listenToStream() {
    try {
        const response = await fetch('/stream');
        if (!response.body) throw new Error('No response body');

        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += value;
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;

                let event;
                try {
                    event = JSON.parse(line);
                } catch (e) {
                    console.error('Stream parse error:', e, line);
                    continue;
                }

                if (event.type === 'route') {
                    showRoute(event.data);
                    continue;
                }

                if (event.type === 'reset') {
                    isNewResponse = true;
                    isResponding = true;
                    updateMicVisibility();
                    flushQueue();
                    routeView.classList.add('hidden');
                    routeBackBtn.classList.add('hidden');
                    displayArea.classList.remove('hidden');
                    transcriptPanel.classList.remove('route-mode');
                    continue;
                }

                if (event.type === 'text') {
                    if (isNewResponse) {
                        output.innerHTML = '';
                        currentSpan = null;
                        isNewResponse = false;
                    }

                    const inRoute = !routeView.classList.contains('hidden');
                    const targetEl = inRoute ? routeNarration : output;
                    const scrollEl = inRoute
                        ? document.querySelector('.direction-panel')
                        : document.querySelector('.stream-container');

                    enqueueText(event.data, targetEl, scrollEl);
                }

                if (event.type === 'memory' || event.type === 'done') {
                    isResponding = false;
                    updateMicVisibility();
                }
            }
        }
    } catch (error) {
        console.log('Connection lost. Retrying.');
        setTimeout(listenToStream, 2000);
    }
}

let routeAnimTimer = null;

function showRoute(data) {
    displayArea.classList.add('hidden');
    routeView.classList.remove('hidden');
    transcriptPanel.classList.add('hidden');
    routeBackBtn.textContent = currentLanguage === 'mm' ? 'နောက်သို့' : 'Back';
    routeBackBtn.classList.toggle('hidden', currentLanguage !== 'mm');

    const isMyanmar = data.displayLanguage === 'mm';
    const startName = data.startNameLocalized || data.startName;
    const destination = data.destinationNameLocalized || data.destinationName || (isMyanmar ? 'သွားမည့်နေရာ' : 'Destination');
    const distance = data.distanceLocalized || data.distance || '--';
    const distanceUnit = data.distanceUnitLocalized || data.distanceUnit || (isMyanmar ? 'ပေ' : 'feet');
    const walkingTimeText = data.walkingTimeTextLocalized || data.walkingTimeText || '--';

    routeHeading.innerText = isMyanmar ? 'လမ်းကြောင်း' : 'ROUTE';
    routeStart.innerText = startName
        ? `${isMyanmar ? 'မှ' : 'From'} ${startName}`
        : (isMyanmar ? 'လက်ရှိနေရာမှ' : 'From current location');
    destinationName.innerText = destination;
    routeDistance.innerText = `${distance} ${distanceUnit}`;
    mapDistance.innerText = `${distance} ${distanceUnit}`;
    mapTime.innerHTML = `${isMyanmar ? 'ခန့်မှန်းလမ်းလျှောက်ချိန်' : 'Estimated walking time'}:<br><span>${walkingTimeText}</span>`;

    const points = data.points || [];
    routeLine.setAttribute('points', points.map(p => `${p.x},${p.y}`).join(' '));
    routeMarkers.innerHTML = '';
    routeNarration.innerHTML = '';
    currentSpan = null;
    currentTargetEl = routeNarration;

    animateRouteLine();
}

function animateRouteLine() {
    clearTimeout(routeAnimTimer);
    routeLine.classList.remove('drawing', 'flowing');
    routeLine.style.strokeDasharray = '';
    routeLine.style.strokeDashoffset = '';

    const len = routeLine.getTotalLength();
    routeLine.style.strokeDasharray = len;
    routeLine.style.strokeDashoffset = len;

    routeLine.getBoundingClientRect();

    routeLine.classList.add('drawing');
    routeLine.style.strokeDashoffset = '0';

    routeAnimTimer = setTimeout(() => {
        routeLine.classList.remove('drawing');
        routeLine.style.strokeDasharray = '';
        routeLine.style.strokeDashoffset = '';
        routeLine.classList.add('flowing');
    }, 1500);
}

fetchMode();
listenToStream();
