document.addEventListener('DOMContentLoaded', function() {
    // Only these three teeth
    const TOOTH_BUTTONS = [
        "Maxillary Left Second Premolar",
        "Maxillary Left First Molar",
        "Maxillary Left Second Molar"
    ];
    // Insert buttons into a single container (use the first quadrant container for simplicity)
    const container = document.getElementById('toothButtons-ml');
    let selectedTooth = null;
    TOOTH_BUTTONS.forEach(toothName => {
        const btn = document.createElement('button');
        btn.className = 'bg-white hover:bg-blue-50 text-blue-900 font-semibold py-2 px-4 border border-blue-300 rounded-lg shadow-sm transition duration-300 mb-2';
        btn.textContent = toothName;
        btn.dataset.tooth = toothName;
        btn.addEventListener('click', function() {
            // Deselect all
            document.querySelectorAll('button[data-tooth]').forEach(b => {
                b.classList.remove('bg-blue-600', 'text-white', 'ring', 'ring-blue-400');
                b.classList.add('bg-white', 'text-blue-900');
            });
            btn.classList.remove('bg-white', 'text-blue-900');
            btn.classList.add('bg-blue-600', 'text-white', 'ring', 'ring-blue-400');
            selectedTooth = toothName;
            document.getElementById('uploadSection').classList.remove('hidden', 'opacity-0');
            document.getElementById('results').classList.add('hidden');
        });
        container.appendChild(btn);
    });

    const xrayUpload = document.getElementById('xrayUpload');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const results = document.getElementById('results');
    const segmentationResult = document.getElementById('segmentationResult');
    const classificationResult = document.getElementById('classificationResult');

    xrayUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                analyzeButton.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    analyzeButton.addEventListener('click', async function() {
        if (!selectedTooth) {
            alert('Please select a tooth first');
            return;
        }
        const file = xrayUpload.files[0];
        if (!file) {
            alert('Please upload an X-ray image first');
            return;
        }
        // Show loading state
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';
        segmentationResult.innerHTML = '';
        classificationResult.innerHTML = '';
        results.classList.remove('hidden');
        segmentationResult.innerHTML = '<div class="flex justify-center items-center"><svg class="animate-spin h-8 w-8 text-blue-600 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path></svg><span class="text-blue-700 font-semibold">Analyzing...</span></div>';
        classificationResult.innerHTML = '';
        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('tooth_name', selectedTooth);
        try {
            const response = await fetch('/api/analyze/individual', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setTimeout(function() {
                analyzeButton.disabled = false;
                analyzeButton.textContent = 'Analyze Tooth';
                // Show segmentation image if present
                if (data.segmentation) {
                    segmentationResult.innerHTML = `<img src="data:image/png;base64,${data.segmentation}" class="w-full h-full object-contain rounded" alt="Segmentation Result" />`;
                } else {
                    segmentationResult.innerHTML = '';
                }
                // Show class result in a pretty way
                if (data.classification) {
                    classificationResult.innerHTML = `<div class='bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 text-blue-900 text-lg font-semibold inline-block shadow'><span class='uppercase tracking-wide text-blue-700 mr-2'>Class:</span> <span class='font-bold text-blue-900'>${data.classification}</span></div>`;
                } else {
                    classificationResult.innerHTML = '';
                }
            }, 2000);
        } catch (error) {
            analyzeButton.disabled = false;
            analyzeButton.textContent = 'Analyze Tooth';
            segmentationResult.innerHTML = '';
            classificationResult.innerHTML = '<span class="text-red-700">An error occurred during analysis.</span>';
        }
    });
}); 