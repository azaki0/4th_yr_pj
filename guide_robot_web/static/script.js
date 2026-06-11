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
const qrSection = document.getElementById('qr-section');
const qrContainer = document.getElementById('qrcode-container');

let isNewResponse = true;
let cursorTimer = null;
let qrTimer = null;
let routeFallbackTimer = null;
let qrGenerated = false;

/* ── Cursor ── */
function showCursor() {
    if (!streamCursor) return;
    streamCursor.classList.add('active');
    clearTimeout(cursorTimer);
    clearTimeout(qrTimer);

    cursorTimer = setTimeout(() => {
        streamCursor.classList.remove('active');
        if (!routeView.classList.contains('hidden') && !qrGenerated) {
            qrTimer = setTimeout(generateDirectionsQR, 400);
        }
    }, 1800);
}

/* ── QR generation ── */
function generateDirectionsQR() {
    if (qrGenerated) return;
    if (typeof html2canvas === 'undefined' || typeof bwipjs === 'undefined') {
        console.warn('Aztec: libraries not loaded yet, retrying in 1s');
        qrTimer = setTimeout(generateDirectionsQR, 1000);
        return;
    }

    html2canvas(routeView, {
        scale: 1,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#090b14',
        logging: false,
    }).then(canvas => {
        const shrinkCanvas = document.createElement('canvas');
        const ctx = shrinkCanvas.getContext('2d');
        const scaleFactor = 260 / canvas.width;
        shrinkCanvas.width = 260;
        shrinkCanvas.height = canvas.height * scaleFactor;
        ctx.drawImage(canvas, 0, 0, shrinkCanvas.width, shrinkCanvas.height);

        const compressedImageBase64 = shrinkCanvas.toDataURL('image/jpeg', 0.15);

        const aztecCanvas = document.createElement('canvas');
        bwipjs.toCanvas(aztecCanvas, {
            bcid: 'azteccode',
            text: compressedImageBase64,
            scale: 2,
            backgroundcolor: 'ffffff',
        });

        qrContainer.innerHTML = '';
        aztecCanvas.style.width = '200px';
        aztecCanvas.style.height = '200px';
        qrContainer.appendChild(aztecCanvas);

        qrGenerated = true;
        qrSection.classList.remove('hidden');
    }).catch(err => {
        console.warn('Aztec generation failed:', err);
    });
}

function clearQR() {
    qrContainer.innerHTML = '';
    qrSection.classList.add('hidden');
    qrGenerated = false;
    clearTimeout(qrTimer);
    clearTimeout(routeFallbackTimer);
}

/* ── Stream listener ── */
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
                    routeView.classList.add('hidden');
                    displayArea.classList.remove('hidden');
                    if (streamCursor) streamCursor.classList.remove('active');
                    clearQR();
                    continue;
                }

                if (event.type === 'text') {
                    if (isNewResponse) {
                        output.innerHTML = '';
                        isNewResponse = false;
                    }

                    if (routeView.classList.contains('hidden')) {
                        appendText(event.data, output, document.querySelector('.stream-container'));
                    } else {
                        appendText(event.data, routeNarration, document.querySelector('.direction-panel'));
                    }

                    showCursor();
                }
            }
        }
    } catch (error) {
        console.log('Connection lost. Retrying.');
        setTimeout(listenToStream, 2000);
    }
}

function appendText(text, targetEl = output, scrollEl = displayArea) {
    let span = targetEl.lastElementChild;
    if (!span || !span.classList.contains('stream-line')) {
        span = document.createElement('span');
        span.className = 'stream-line';
        targetEl.appendChild(span);
    }
    span.textContent += text;
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
}

function showRoute(data) {
    displayArea.classList.add('hidden');
    routeView.classList.remove('hidden');
    clearQR();

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

    /* Fallback: generate QR 5s after route appears, in case no narration text comes */
    routeFallbackTimer = setTimeout(() => {
        if (!qrGenerated) generateDirectionsQR();
    }, 5000);
}

listenToStream();
