const slotConfig = [
    { key: 'front', label: 'Front' },
    { key: 'back', label: 'Back' },
    { key: 'left', label: 'Left Side' },
    { key: 'right', label: 'Right Side' },
    { key: 'roof', label: 'Roof' },
];

const files = new Map();

const slots = Array.from(document.querySelectorAll('.upload-slot'));
const startLocationInput = document.getElementById('startLocation');
const destinationInput = document.getElementById('destination');
const requestBtn = document.getElementById('requestRide');
const progressWrap = document.getElementById('progressWrap');
const overlay = document.getElementById('resultOverlay');
const closeOverlay = document.getElementById('closeOverlay');
const rawJson = document.getElementById('rawJson');
const statusText = document.getElementById('statusText');
const badge = document.getElementById('verificationBadge');
const approvedPanel = document.getElementById('approvedPanel');
const rejectedPanel = document.getElementById('rejectedPanel');
const tierValue = document.getElementById('tierValue');
const etaValue = document.getElementById('etaValue');
const damagedAngles = document.getElementById('damagedAngles');
const heatmaps = document.getElementById('heatmaps');

function bindSlot(slot) {
    const input = slot.querySelector('input[type="file"]');
    const fileLabel = slot.querySelector('.slot-file');
    const key = slot.dataset.key;

    function setFile(file) {
        files.set(key, file);
        fileLabel.textContent = file.name;
        slot.classList.add('ready');
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

function resetPanels() {
    approvedPanel.classList.add('hidden');
    rejectedPanel.classList.add('hidden');
    damagedAngles.innerHTML = '';
    heatmaps.innerHTML = '';
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
    statusText.textContent = 'CLEARED';
    approvedPanel.classList.remove('hidden');

    tierValue.textContent = `${data?.tier_info?.uber_tier || '-'} (${data?.tier_info?.body_type || '-'})`;
    etaValue.textContent = data?.route?.formatted_eta || '-';
}

function showRejected(data) {
    badge.className = 'verification-badge rejected';
    statusText.textContent = 'REJECTED';
    rejectedPanel.classList.remove('hidden');

    const angles = data?.damaged_angles || [];
    angles.forEach((angle) => {
        const li = document.createElement('li');
        li.textContent = angle;
        damagedAngles.appendChild(li);
    });

    const map = data?.heatmaps || {};
    Object.entries(map).forEach(([angle, p]) => {
        const url = normalizeHeatmapPath(p);

        const wrapper = document.createElement('div');
        wrapper.className = 'heatmap-item';

        const title = document.createElement('div');
        title.textContent = angle;
        title.style.marginBottom = '0.35rem';
        title.style.fontWeight = '700';
        wrapper.appendChild(title);

        if (url) {
            const img = document.createElement('img');
            img.src = url;
            img.alt = `${angle} heatmap`;
            wrapper.appendChild(img);
        } else {
            const pEl = document.createElement('p');
            pEl.textContent = p;
            pEl.className = 'muted';
            wrapper.appendChild(pEl);
        }

        heatmaps.appendChild(wrapper);
    });
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

    requestBtn.disabled = true;
    progressWrap.classList.remove('hidden');
    resetPanels();

    try {
        const response = await fetch('/request-ride', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        rawJson.textContent = JSON.stringify(data, null, 2);

        if (!response.ok) {
            badge.className = 'verification-badge rejected';
            statusText.textContent = `ERROR ${response.status}`;
            rejectedPanel.classList.remove('hidden');
        } else if (data.status === 'REJECTED') {
            showRejected(data);
        } else {
            showApproved(data);
        }

        overlay.classList.remove('hidden');
    } catch (err) {
        rawJson.textContent = String(err);
        badge.className = 'verification-badge rejected';
        statusText.textContent = 'REQUEST FAILED';
        rejectedPanel.classList.remove('hidden');
        overlay.classList.remove('hidden');
    } finally {
        requestBtn.disabled = false;
        progressWrap.classList.add('hidden');
    }
}

slots.forEach(bindSlot);
requestBtn.addEventListener('click', submitRequest);
closeOverlay.addEventListener('click', () => overlay.classList.add('hidden'));
