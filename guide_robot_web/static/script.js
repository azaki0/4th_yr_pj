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

let isNewResponse = true;

let charQueue = [];
let typeInterval = null;
let currentSpan = null;
let currentScrollEl = null;
const CHAR_DELAY_MS = 18;

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
            (output === currentScrollEl?.firstChild ? output : (currentSpan._target || output)).appendChild(currentSpan);
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
    if (streamCursor) streamCursor.classList.remove('active');
}

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
                    flushQueue();
                    routeView.classList.add('hidden');
                    displayArea.classList.remove('hidden');
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

listenToStream();
