document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const labelResults = document.getElementById('labelResults');
    const results = document.getElementById('results');

    imageUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze/collective', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                results.classList.remove('hidden');
                labelResults.innerHTML = '';
                if (data.error) {
                    labelResults.innerHTML = `<div class="bg-red-100 text-red-800 rounded p-4 font-semibold">${data.error}</div>`;
                } else if (data.success && data.labels) {
                    let html = '';
                    data.labels.forEach(item => {
                        html += `<div class="mb-2"><span class="font-semibold text-blue-900">${item.tooth}:</span> <span class="text-gray-800">${item.label}</span></div>`;
                    });
                    setTimeout(() => {
                        labelResults.innerHTML = html;
                    }, 30000); // 90 seconds delay
                }
            } else {
                throw new Error('Analysis failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis');
        }
    });
}); 