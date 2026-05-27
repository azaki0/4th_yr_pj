const output = document.getElementById('stream-output');
const displayArea = document.getElementById('display-area');
const routeView = document.getElementById('route-view');
const routeLine = document.getElementById('route-line');
const routeMarkers = document.getElementById('route-markers');
const destinationName = document.getElementById('destination-name');
const routeDistance = document.getElementById('route-distance');
const routeSteps = document.getElementById('route-steps');
const routeNarration = document.getElementById('route-narration');

let isNewResponse = true;

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
                    console.error("Stream parse error:", e, line);
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
                    continue;
                }

                if (event.type === 'text') {
                    if (isNewResponse) {
                        output.innerHTML = '';
                        isNewResponse = false;
                    }

                    if (routeView.classList.contains('hidden')) {
                        appendText(event.data, output, displayArea);
                    } else {
                        appendText(event.data, routeNarration, document.querySelector('.direction-panel'));
                    }
                }
            }
        }
    } catch (error) {
        console.log("Connection lost. Retrying.");
        setTimeout(listenToStream, 2000);
    }
}

function appendText(text, targetEl = output, scrollEl = displayArea) {
    let span = targetEl.lastElementChild;
    if (!span || !span.classList.contains('stream-line')) {
        span = document.createElement('span');
        span.className = 'stream-line';
        span.style.display = "block";
        span.style.marginBottom = "15px";
        targetEl.appendChild(span);
    }

    span.textContent += text;

    if (scrollEl) {
        scrollEl.scrollTop = scrollEl.scrollHeight;
    }
}

function typeText(text, targetEl = output, scrollEl = displayArea) {
    return new Promise((resolve) => {
        let i = 0;
        const span = document.createElement('span');
        span.style.display = "block";
        span.style.marginBottom = "15px";
        targetEl.appendChild(span);

        function type() {
            if (i < text.length) {
                span.innerHTML += text.charAt(i);
                i++;

                if (scrollEl) {
                    scrollEl.scrollTop = scrollEl.scrollHeight;
                }
                setTimeout(type, 30);
            } else {
                resolve();
            }
        }
        type();
    });
}

function showRoute(data) {
    displayArea.classList.add('hidden');
    routeView.classList.remove('hidden');

    destinationName.innerText = data.destinationName || "Destination";
    routeDistance.innerText = `${data.distance ?? "--"} ${data.distanceUnit || "feet"}`;

    const points = data.points || [];
    routeLine.setAttribute('points', points.map(point => `${point.x},${point.y}`).join(' '));
    routeMarkers.innerHTML = '';
    routeSteps.innerHTML = '';
    routeNarration.innerHTML = '';

    points.forEach((point, index) => {
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marker.setAttribute('cx', point.x);
        marker.setAttribute('cy', point.y);
        marker.setAttribute('r', index === 0 || index === points.length - 1 ? 9 : 6);
        marker.setAttribute('class', index === 0 ? 'start-marker' : index === points.length - 1 ? 'end-marker' : 'path-marker');
        routeMarkers.appendChild(marker);

        const step = document.createElement('div');
        step.className = 'route-step';
        step.innerHTML = `<span>${index + 1}</span><p>${point.label}</p>`;
        routeSteps.appendChild(step);
    });
}

listenToStream();
