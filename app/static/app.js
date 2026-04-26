const form = document.getElementById('requestForm');
const output = document.getElementById('output');
const statusBadge = document.getElementById('statusBadge');
const loadDemo = document.getElementById('loadDemo');

function setStatus(text, kind = 'neutral') {
    statusBadge.textContent = text;
    statusBadge.className = `badge ${kind}`;
}

function setOutput(value) {
    output.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const required = ['front', 'back', 'left', 'right', 'roof'];
    for (const key of required) {
        if (!formData.get(key) || !formData.get(key).name) {
            setStatus('Missing files', 'danger');
            setOutput(`Please upload all 5 images: ${required.join(', ')}`);
            return;
        }
    }

    setStatus('Running...', 'neutral');
    setOutput('Sending request to /request-ride ...');

    try {
        const response = await fetch('/request-ride', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            setStatus(`Error ${response.status}`, 'danger');
            setOutput(data);
            return;
        }

        setStatus(data.status || 'Success', data.status === 'REJECTED' ? 'danger' : 'success');
        setOutput(data);
    } catch (error) {
        setStatus('Request failed', 'danger');
        setOutput(String(error));
    }
});

loadDemo.addEventListener('click', () => {
    const destination = form.querySelector('input[name="destination"]');
    destination.value = 'Lahore Airport';
    setStatus('Demo ready', 'neutral');
    setOutput('Select 5 sample images and click Run Dispatch Check.');
});
