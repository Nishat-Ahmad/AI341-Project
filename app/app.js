const slotConfig = [
    { key: 'front', label: 'Front' },
    { key: 'back', label: 'Back' },
    { key: 'left', label: 'Left Side' },
    { key: 'right', label: 'Right Side' },
    { key: 'roof', label: 'Roof' },
];

const files = new Map();

// DOM Elements
const landingPage = document.getElementById('landingPage');
const uploadStep = document.getElementById('uploadStep');
const destinationStep = document.getElementById('destinationStep');
const resultOverlay = document.getElementById('resultOverlay');

const getStartedBtn = document.getElementById('getStartedBtn');
const nextToDestination = document.getElementById('nextToDestination');
const backToUpload = document.getElementById('backToUpload');
const submitRide = document.getElementById('submitRide');
const closeOverlay = document.getElementById('closeOverlay');
const startOver = document.getElementById('startOver');

const startLocationInput = document.getElementById('startLocation');
const destinationInput = document.getElementById('destination');
const progressWrap = document.getElementById('progressWrap');

const rawJson = document.getElementById('rawJson');
const statusText = document.getElementById('statusText');
const badge = document.getElementById('verificationBadge');
const approvedPanel = document.getElementById('approvedPanel');
const rejectedPanel = document.getElementById('rejectedPanel');
const tierValue = document.getElementById('tierValue');
const etaValue = document.getElementById('etaValue');
const damagedAngles = document.getElementById('damagedAngles');
const heatmaps = document.getElementById('heatmaps');
const jsonOutput = document.getElementById('jsonOutput');

const slots = Array.from(document.querySelectorAll('.upload-slot'));

// Step Navigation with animations
function showPage(page) {
    const currentVisible = document.querySelector('.landing-page:not(.slide-up):not(.hidden), .step-page.slide-in');
    
    if (currentVisible === landingPage && page !== landingPage) {
        // Slide out landing
        landingPage.classList.add('slide-up');
        setTimeout(() => {
            landingPage.classList.add('hidden');
            page.style.display = 'flex';
            page.classList.add('slide-in');
        }, 600);
    } else if (currentVisible === uploadStep && page !== uploadStep) {
        // Slide out upload
        uploadStep.classList.remove('slide-in');
        setTimeout(() => {
            uploadStep.style.display = 'none';
            page.style.display = 'flex';
            page.classList.add('slide-in');
        }, 600);
    } else if (currentVisible === destinationStep && page !== destinationStep) {
        // Slide out destination (back to upload)
        destinationStep.classList.remove('slide-in');
        setTimeout(() => {
            destinationStep.style.display = 'none';
            uploadStep.style.display = 'flex';
            uploadStep.classList.add('slide-in');
        }, 600);
    } else {
        // Initial show
        landingPage.classList.add('hidden');
        page.style.display = 'flex';
        page.classList.add('slide-in');
    }
}

function checkAllFilesUploaded() {
    const allUploaded = slotConfig.every(s => files.has(s.key));
    nextToDestination.disabled = !allUploaded;
}

// File upload handling
function bindSlot(slot) {
    const input = slot.querySelector('input[type="file"]');
    const slotStatus = slot.querySelector('.slot-status');
    const slotPreview = slot.querySelector('.slot-preview');
    const previewImg = slot.querySelector('.slot-preview img');
    const key = slot.dataset.key;

    function setFile(file) {
        files.set(key, file);
        slot.classList.add('ready');
        
        // Show thumbnail preview
        if (slotPreview && previewImg) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
        
        // Update status text
        slotStatus.textContent = 'scanning';
        checkAllFilesUploaded();
    }

    input.addEventListener('change', () => {
        if (input.files && input.files[0]) {
            setFile(input.files[0]);
        }
    });

    slot.addEventListener('dragover', (event) => {
        event.preventDefault();
        slot.classList.add('drag');
    });

    slot.addEventListener('dragleave', () => {
        slot.classList.remove('drag');
    });

    slot.addEventListener('drop', (event) => {
        event.preventDefault();
        slot.classList.remove('drag');
        const dropped = event.dataTransfer?.files?.[0];
        if (dropped) {
            setFile(dropped);
        }
    });
}

// Clean Data Output Formatter
function formatCleanData(data) {
    const sections = [];
    
    // Status Section
    if (data.status) {
        sections.push(`
            <div class="clean-section">
                <div class="clean-section-title">Verification Status</div>
                <div class="clean-status">${data.status}</div>
            </div>
        `);
    }
    
    // Tier Info Section
    if (data.tier_info) {
        const tier = data.tier_info;
        const bodyConfidence = tier.body_confidence ? (parseFloat(tier.body_confidence) * 100).toFixed(1) + '%' : '-';
        
        sections.push(`
            <div class="clean-section">
                <div class="clean-section-title">Vehicle Classification</div>
                <div class="clean-data-grid">
                    <div class="clean-data-item">
                        <span class="clean-data-label">Service Tier</span>
                        <span class="clean-data-value">${tier.uber_tier || '-'}</span>
                    </div>
                    <div class="clean-data-item">
                        <span class="clean-data-label">Body Type</span>
                        <span class="clean-data-value">${tier.body_type || '-'}</span>
                    </div>
                    <div class="clean-data-item">
                        <span class="clean-data-label">Classification</span>
                        <div class="confidence-badge"><strong>${bodyConfidence}</strong> confidence</div>
                    </div>
                </div>
            </div>
        `);
    }
    
    // Route Section
    if (data.route) {
        const route = data.route;
        const distance = route.distance_km ? `${parseFloat(route.distance_km).toFixed(2)} km` : '-';
        
        sections.push(`
            <div class="clean-section">
                <div class="clean-section-title">Route Information</div>
                <div class="clean-route-summary">
                    <div class="clean-route-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"/>
                            <polyline points="12 6 12 12 16 14"/>
                        </svg>
                    </div>
                    <div class="clean-route-details">
                        <div class="clean-route-start">${route.start_location || '-'}</div>
                        <div class="clean-route-end">to ${route.destination || '-'}</div>
                    </div>
                    <div class="clean-route-distance">
                        <div class="clean-data-value">${distance}</div>
                        <div class="clean-data-label">Distance</div>
                    </div>
                </div>
                <div class="clean-data-grid" style="margin-top: var(--space-4);">
                    <div class="clean-data-item">
                        <span class="clean-data-label">ETA</span>
                        <span class="clean-data-value">${route.formatted_eta || '-'}</span>
                    </div>
                    ${route.duration_min ? `
                    <div class="clean-data-item">
                        <span class="clean-data-label">Duration</span>
                        <span class="clean-data-value">${Math.round(parseFloat(route.duration_min))} min</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `);
    }
    
    // Additional Data (model info)
    if (data.model_version || data.timestamp) {
        sections.push(`
            <div class="clean-section">
                <div class="clean-section-title">Inspection Details</div>
                <div class="clean-data-grid">
                    ${data.model_version ? `
                    <div class="clean-data-item">
                        <span class="clean-data-label">Model Version</span>
                        <span class="clean-data-value">${data.model_version}</span>
                    </div>
                    ` : ''}
                    ${data.timestamp ? `
                    <div class="clean-data-item">
                        <span class="clean-data-label">Timestamp</span>
                        <span class="clean-data-value">${data.timestamp}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `);
    }
    
    return sections.join('');
}

function resetPanels() {
    approvedPanel.classList.add('hidden');
    rejectedPanel.classList.add('hidden');
    damagedAngles.innerHTML = '';
    heatmaps.innerHTML = '';
    jsonOutput.classList.add('hidden');
    badge.className = 'verification-badge';
    statusText.className = 'status-text';
}

function normalizeHeatmapPath(path) {
    if (!path) return null;
    const normalized = path.replaceAll('\\', '/');
    const marker = '/outputs/';
    const idx = normalized.toLowerCase().indexOf(marker);
    if (idx >= 0) {
        return normalized.slice(idx);
    }
    return null;
}

function showApproved(data) {
    badge.className = 'verification-badge cleared';
    statusText.textContent = 'APPROVED';
    statusText.classList.add('cleared');
    approvedPanel.classList.remove('hidden');

    tierValue.textContent = `${data?.tier_info?.uber_tier || '-'} (${data?.tier_info?.body_type || '-'})`;
    etaValue.textContent = data?.route?.formatted_eta || '-';
    
    // Show clean formatted output
    jsonOutput.innerHTML = formatCleanData(data);
    jsonOutput.classList.remove('hidden');
}

function showRejected(data) {
    badge.className = 'verification-badge rejected';
    statusText.textContent = 'REJECTED';
    statusText.classList.add('rejected');
    rejectedPanel.classList.remove('hidden');

    const angles = data?.damaged_angles || [];
    angles.forEach((angle) => {
        const li = document.createElement('li');
        li.className = 'damage-tag';
        li.textContent = angle;
        damagedAngles.appendChild(li);
    });

    const map = data?.heatmaps || {};
    Object.entries(map).forEach(([angle, p]) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'heatmap-item';

        const title = document.createElement('div');
        title.className = 'heatmap-title';
        title.textContent = angle;
        wrapper.appendChild(title);

        const url = normalizeHeatmapPath(p);
        if (url) {
            const img = document.createElement('img');
            img.src = url;
            img.alt = `${angle} heatmap`;
            wrapper.appendChild(img);
        }
        heatmaps.appendChild(wrapper);
    });
    
    // Show clean formatted output
    jsonOutput.innerHTML = formatCleanData(data);
    jsonOutput.classList.remove('hidden');
}

function resetForm() {
    // Clear files
    files.clear();
    slots.forEach(slot => {
        slot.classList.remove('ready');
        const slotStatus = slot.querySelector('.slot-status');
        const previewImg = slot.querySelector('.slot-preview img');
        slotStatus.textContent = 'tap to capture';
        if (previewImg) {
            previewImg.src = '';
        }
    });
    
    // Clear inputs
    startLocationInput.value = '';
    destinationInput.value = '';
    
    // Reset button state
    checkAllFilesUploaded();
}

async function submitRequest() {
    const missing = slotConfig.filter((s) => !files.get(s.key));
    if (missing.length) {
        alert(`Missing files: ${missing.map((m) => m.label).join(', ')}`);
        return;
    }

    if (!startLocationInput.value.trim()) {
        alert('Please enter a starting location.');
        return;
    }

    if (!destinationInput.value.trim()) {
        alert('Please enter a destination.');
        return;
    }

    const formData = new FormData();
    formData.append('front', files.get('front'));
    formData.append('back', files.get('back'));
    formData.append('left', files.get('left'));
    formData.append('right', files.get('right'));
    formData.append('roof', files.get('roof'));
    formData.append('start_location', startLocationInput.value.trim());
    formData.append('destination', destinationInput.value.trim());

    submitRide.disabled = true;
    progressWrap.classList.remove('hidden');
    resetPanels();

    try {
        const response = await fetch('/request-ride', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        
        // Show result modal
        resultOverlay.classList.remove('hidden');

        if (!response.ok) {
            badge.className = 'verification-badge rejected';
            statusText.textContent = `ERROR ${response.status}`;
            statusText.classList.add('rejected');
            rejectedPanel.classList.remove('hidden');
            jsonOutput.innerHTML = formatCleanData(data);
            jsonOutput.classList.remove('hidden');
        } else if (data.status === 'REJECTED') {
            showRejected(data);
        } else {
            showApproved(data);
        }
    } catch (err) {
        resultOverlay.classList.remove('hidden');
        badge.className = 'verification-badge rejected';
        statusText.textContent = 'REQUEST FAILED';
        statusText.classList.add('rejected');
        rejectedPanel.classList.remove('hidden');
        jsonOutput.innerHTML = `<div class="clean-section"><div class="clean-section-title">Error</div><div class="clean-data-value" style="color: var(--error);">${err}</div></div>`;
        jsonOutput.classList.remove('hidden');
    } finally {
        submitRide.disabled = false;
        progressWrap.classList.add('hidden');
    }
}

// Event Listeners
getStartedBtn.addEventListener('click', () => showPage(uploadStep));
nextToDestination.addEventListener('click', () => showPage(destinationStep));
backToUpload.addEventListener('click', () => showPage(uploadStep));
submitRide.addEventListener('click', submitRequest);
closeOverlay.addEventListener('click', () => resultOverlay.classList.add('hidden'));
startOver.addEventListener('click', () => {
    resultOverlay.classList.add('hidden');
    resetForm();
    showPage(landingPage);
});

// Initialize
slots.forEach(bindSlot);