document.addEventListener('DOMContentLoaded', function () {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFile');

    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const emptyState = document.getElementById('emptyState');
    const loadDemoBtn = document.getElementById('loadDemo');

    // Fetch model info on load
    fetchModelInfo();

    // Drag & Drop Handlers
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a valid CSV file.');
            return;
        }
        fileName.textContent = file.name;
        dropZone.classList.add('d-none');
        fileInfo.classList.remove('d-none');
        window.selectedFile = file;
    }

    removeFileBtn.addEventListener('click', () => {
        fileName.textContent = '';
        fileInput.value = '';
        window.selectedFile = null;
        fileInfo.classList.add('d-none');
        dropZone.classList.remove('d-none');
        resetResults();
    });

    function resetResults() {
        resultsContainer.classList.add('d-none');
        emptyState.classList.remove('d-none');
    }

    const manualForm = document.getElementById('manualForm');
    const manualResultContainer = document.getElementById('manualResultContainer');
    const manualEmptyState = document.getElementById('manualEmptyState');

    loadDemoBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileName.textContent = "sample_bearing_data.csv (Demo)";
        dropZone.classList.add('d-none');
        fileInfo.classList.remove('d-none');
        window.demoMode = true;
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!window.selectedFile && !window.demoMode) {
            alert("Please select a file first.");
            return;
        }

        emptyState.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        loadingSpinner.classList.remove('d-none');

        const formData = new FormData();
        if (window.selectedFile) {
            formData.append('file', window.selectedFile);
        } else if (window.demoMode) {
            formData.append('demo', 'true');
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayResults(result);
            } else {
                alert('Error: ' + (result.error || 'Prediction failed'));
                resetResults();
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the prediction server.');
            resetResults();
        } finally {
            loadingSpinner.classList.add('d-none');
        }
    });

    if (manualForm) {
        manualForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(manualForm);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict_manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    manualEmptyState.classList.add('d-none');
                    manualResultContainer.classList.remove('d-none');

                    document.getElementById('manualRulValue').textContent = (result.rul * 100).toFixed(1) + "%";
                    document.getElementById('manualRulLow').textContent = (result.rul_low * 100).toFixed(1) + "%";
                    document.getElementById('manualRulHigh').textContent = (result.rul_high * 100).toFixed(1) + "%";
                    document.getElementById('impactThermal').textContent = result.impact.thermal;
                    document.getElementById('impactLoad').textContent = result.impact.load;
                    document.getElementById('impactElec').textContent = result.impact.elec;

                    // Style impact indicators
                    styleImpact('impactThermal', result.impact.thermal);
                    styleImpact('impactLoad', result.impact.load);
                    styleImpact('impactElec', result.impact.elec);
                } else {
                    alert('Error: ' + (result.error));
                }
            } catch (error) {
                console.error("Error:", error);
            }
        });
    }

    function styleImpact(elementId, value) {
        const el = document.getElementById(elementId);
        el.classList.remove('text-success', 'text-warning', 'text-danger');
        if (value === 'CRITICAL') {
            el.classList.add('text-danger');
        } else if (value === 'High' || value === 'Abnormal') {
            el.classList.add('text-warning');
        } else {
            el.classList.add('text-success');
        }
    }

    function displayResults(data) {
        resultsContainer.classList.remove('d-none');

        const rulMedian = data.rul;
        const rulLow = data.rul_low || rulMedian;
        const rulHigh = data.rul_high || rulMedian;
        const hi = data.hi || rulMedian;

        const rulVal = (rulMedian * 100).toFixed(1);
        const rulLowVal = (rulLow * 100).toFixed(1);
        const rulHighVal = (rulHigh * 100).toFixed(1);

        const rulElem = document.getElementById('rulValue');
        const rulBar = document.getElementById('rulBar');
        const rulBarLow = document.getElementById('rulBarLow');
        const rulText = document.getElementById('rulText');
        const insightText = document.getElementById('insightText');
        const rulLowLabel = document.getElementById('rulLowLabel');
        const rulHighLabel = document.getElementById('rulHighLabel');

        rulElem.textContent = `${rulVal}%`;
        rulBar.style.width = `${rulVal}%`;
        rulBarLow.style.width = `${rulHighVal}%`;
        rulLowLabel.textContent = `10th: ${rulLowVal}%`;
        rulHighLabel.textContent = `90th: ${rulHighVal}%`;

        // Color coding
        rulBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
        rulBarLow.classList.remove('bg-success', 'bg-warning', 'bg-danger');

        if (rulMedian > 0.7) {
            rulBar.classList.add('bg-success');
            rulBarLow.classList.add('bg-success');
            rulText.textContent = "Motor is in excellent condition.";
            insightText.textContent = "Routine maintenance schedule is sufficient. Continue monitoring vibration patterns.";
        } else if (rulMedian > 0.3) {
            rulBar.classList.add('bg-warning');
            rulBarLow.classList.add('bg-warning');
            rulText.textContent = "Motor showing early signs of wear.";
            insightText.textContent = "Schedule inspection within next 2 weeks. Monitor vibration levels closely.";
        } else {
            rulBar.classList.add('bg-danger');
            rulBarLow.classList.add('bg-danger');
            rulText.textContent = "Critical condition detected!";
            insightText.textContent = "IMMEDIATE ACTION REQUIRED: Stop motor and inspect bearings to prevent catastrophic failure.";
        }

        // Health Indicator
        document.getElementById('hiValue').textContent = (hi * 100).toFixed(1) + '%';

        // Speed & Voltage
        document.getElementById('speedValue').textContent = data.speed.toFixed(3);
        document.getElementById('voltageValue').textContent = data.voltage.toFixed(3);

        // Model type
        const modelTypeInfo = document.getElementById('modelTypeInfo');
        if (modelTypeInfo) {
            const modelName = data.model_type === 'hybrid'
                ? 'Hybrid MSCAN + Transformer (Quantile)'
                : data.model_type === 'legacy'
                    ? 'Legacy MSCAN'
                    : 'Demonstration Mode';
            modelTypeInfo.textContent = modelName;
        }

        // Render Signal Chart
        if (data.signals) {
            const traceH = {
                y: data.signals.h,
                mode: 'lines',
                name: 'Horiz Vibration',
                line: { color: '#FF9644', width: 1 }
            };
            const traceV = {
                y: data.signals.v,
                mode: 'lines',
                name: 'Vert Vibration',
                line: { color: '#562F00', width: 1, opacity: 0.5 }
            };
            const traceT = {
                y: data.signals.t,
                mode: 'lines',
                name: 'Temperature',
                yaxis: 'y2',
                line: { color: '#FF4757', width: 2 }
            };

            const layout = {
                margin: { l: 40, r: 40, b: 30, t: 10 },
                showlegend: true,
                legend: { orientation: 'h', y: -0.2 },
                xaxis: { title: 'Samples', showgrid: false },
                yaxis: { title: 'Vibration (norm)', showgrid: true },
                yaxis2: {
                    title: 'Temp (°C)',
                    overlaying: 'y',
                    side: 'right',
                    range: [0, 120]
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                hovermode: 'x unified'
            };

            const config = { responsive: true, displayModeBar: false };
            Plotly.newPlot('signalChart', [traceH, traceV, traceT], layout, config);
        }
    }


    async function fetchModelInfo() {
        try {
            const response = await fetch('/model_info');
            const info = await response.json();

            const badge = document.getElementById('modelBadge');
            if (badge) {
                if (info.model_type === 'hybrid') {
                    badge.textContent = 'Hybrid Model';
                    badge.classList.add('bg-gradient-primary');
                } else if (info.model_type === 'legacy') {
                    badge.textContent = 'Legacy Model';
                    badge.style.background = 'linear-gradient(135deg, #f59e0b, #fbbf24)';
                } else {
                    badge.textContent = 'Demo Mode';
                    badge.style.background = 'linear-gradient(135deg, #94a3b8, #64748b)';
                }
            }
        } catch (err) {
            console.log('Could not fetch model info');
        }
    }
});
