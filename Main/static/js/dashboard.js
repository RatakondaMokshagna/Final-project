document.addEventListener('DOMContentLoaded', () => {
    // Fake data logic to simulate incoming data initially
    initCharts();

    const analyzeBtn = document.getElementById('analyzeBtn');
    const liveInput = document.getElementById('liveInput');
    const resultsArea = document.getElementById('resultsArea');

    analyzeBtn.addEventListener('click', async () => {
        const textAreaVal = liveInput.value.trim();
        if (!textAreaVal) return;

        analyzeBtn.innerText = "Analyzing...";
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textAreaVal })
            });

            if (response.ok) {
                const data = await response.json();
                displayResults(data);
            } else {
                alert("Error analyzing text. Try again.");
            }
        } catch (e) {
            console.error(e);
            alert("API connection failed.");
        } finally {
            analyzeBtn.innerText = "Analyze Now";
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        resultsArea.classList.remove('hidden');
        const sentColor = data.sentiment === 'pos' ? 'var(--positive)' : (data.sentiment === 'neg' ? 'var(--negative)' : '#fbbf24');
        const misColor = data.is_misinformation ? 'var(--negative)' : 'var(--positive)';

        // Convert to readable
        let sentimentText = "Neutral/Unknown";
        if (data.sentiment === "pos") sentimentText = "Positive";
        if (data.sentiment === "neg") sentimentText = "Negative";
        if (data.sentiment === "neu") sentimentText = "Neutral";

        resultsArea.innerHTML = `
            <div class="res-item">
                <span>Sentiment</span>
                <span style="color: ${sentColor}">${sentimentText} (${(data.confidence * 100).toFixed(1)}%)</span>
            </div>
            <div class="res-item">
                <span>Misinformation Flag</span>
                <span style="color: ${misColor}">${data.is_misinformation ? 'DETECTED! 🚩' : 'Clean ✅'}</span>
            </div>
        `;
    }
});

let sentimentChartObj, misinfoChartObj;

function initCharts() {
    const ctx1 = document.getElementById('sentimentChart').getContext('2d');
    const ctx2 = document.getElementById('misinfoChart').getContext('2d');

    // Gradient setup
    let gradientPos = ctx1.createLinearGradient(0, 0, 0, 400);
    gradientPos.addColorStop(0, 'rgba(16, 185, 129, 0.5)');
    gradientPos.addColorStop(1, 'rgba(16, 185, 129, 0.05)');

    // Mock initial data covering recent hours simulating real-time
    const labels = ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "Live"];

    sentimentChartObj = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Positive',
                    data: [45, 48, 52, 50, 60, 58, 62, 65],
                    borderColor: '#10b981',
                    backgroundColor: gradientPos,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Negative',
                    data: [35, 30, 28, 30, 25, 28, 22, 20],
                    borderColor: '#ef4444',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#e2e8f0' } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    misinfoChartObj = new Chart(ctx2, {
        type: 'doughnut',
        data: {
            labels: ['Truthful', 'Misinformation'],
            datasets: [{
                data: [92, 8],
                backgroundColor: ['#6366f1', '#ef4444'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            cutout: '75%',
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Simulate realtime updates every 10 seconds
    setInterval(updateCharts, 10000);
}

function updateCharts() {
    // This function will fetch from \`/api/stats\` when ready. 
    // Right now, mock live updates slightly...
    const posLine = sentimentChartObj.data.datasets[0];
    const negLine = sentimentChartObj.data.datasets[1];

    // Shift arrays
    posLine.data.shift();
    negLine.data.shift();

    // push new random variation
    posLine.data.push(Math.max(40, posLine.data[posLine.data.length - 1] + (Math.random() * 6 - 2)));
    negLine.data.push(Math.max(10, negLine.data[negLine.data.length - 1] + (Math.random() * 6 - 3)));

    sentimentChartObj.update();
}
