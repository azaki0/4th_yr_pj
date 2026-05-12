const output = document.getElementById('stream-output');
const displayArea = document.getElementById('display-area');
const dashboardView = document.getElementById('dashboard-view');

let isNewResponse = true;

async function listenToStream() {
    try {
        const response = await fetch('/stream');
        if (!response.body) throw new Error('No response body');

        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            // 1. DASHBOARD TRIGGER
            if (value.includes("||JSON||")) {
                const rawData = value.replace("||JSON||", "");
                try {
                    const data = JSON.parse(rawData);
                    showDashboard(data);
                } catch (e) { console.error("Parse Error:", e); }
                continue;
            }

            // 2. RESET TRIGGER (New Query starts)
            if (value.includes("||RESET||")) {
                isNewResponse = true;
                // Instantly hide dashboard and show chat area again
                dashboardView.classList.add('hidden');
                displayArea.classList.remove('hidden');
                continue; 
            }

            // 3. NORMAL TEXT STREAMING
            if (isNewResponse) {
                output.innerHTML = '';
                isNewResponse = false;
            }
            
            // Call typeText safely with explicit arguments
            await typeText(value, output, displayArea);
        }
    } catch (error) {
        console.log("Connection lost. Retrying...");
        setTimeout(listenToStream, 2000);
    }
}

// Added default parameters so the old logic never breaks
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

                // Safely scroll whichever container is passed
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

async function showDashboard(data) {
    // Hide Chat, Show Dashboard
    displayArea.classList.add('hidden');
    dashboardView.classList.remove('hidden');

    const summaryTarget = document.getElementById('summary-text');
    summaryTarget.innerHTML = ''; // Clear previous summary
    
    // Update Stats on the right side
    document.getElementById('urgency-val').innerText = data.urgency;
    document.getElementById('result-val').innerText = data.suspected;
    
    // Animate the Bar
    const bar = document.getElementById('confidence-bar');
    const text = document.getElementById('confidence-text');
    bar.style.width = '0%';
    setTimeout(() => {
        bar.style.width = data.confidence + "%";
        text.innerText = data.confidence + "%";
    }, 100);

    // Type the summary out slowly on the left side
    const leftPanel = document.querySelector('.left-panel');
    await typeText(data.summary, summaryTarget, leftPanel);
}

listenToStream();