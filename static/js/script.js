const fileElem = document.getElementById('fileElem');
const dropArea = document.getElementById('drop-area');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const classifyBtn = document.getElementById('classifyBtn');
const loading = document.getElementById('loading');
const resultDiv = document.getElementById('result');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');
const progressBar = document.getElementById('progress');

let selectedFile = null;

function preventDefaults (e) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)
})

dropArea.addEventListener('drop', (e) => {
  const dt = e.dataTransfer
  const files = dt.files
  handleFiles(files)
})

fileElem.addEventListener('change', (e) => {
  handleFiles(e.target.files)
})

function handleFiles(files) {
  if (!files || files.length === 0) return;
  const file = files[0];

  const valid = validateFile(file);
  if (!valid) return;

  selectedFile = file;
  showPreview(file);
  classifyBtn.disabled = false;
}

function validateFile(file) {
  const allowed = ['image/jpeg','image/png','image/jpg'];
  const maxSize = 5 * 1024 * 1024; // 5MB
  if (!allowed.includes(file.type)) {
    alert('Unsupported file format. Please upload JPG or PNG.')
    return false;
  }
  if (file.size > maxSize) {
    alert('File too large. Maximum 5MB allowed.')
    return false;
  }
  return true;
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    previewImg.src = e.target.result;
    preview.classList.remove('hidden');
    resultDiv.classList.add('hidden');
  }
  reader.readAsDataURL(file);
}

classifyBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Please choose an image first.');
    return;
  }

  classifyBtn.disabled = true;
  loading.classList.remove('hidden');
  resultDiv.classList.add('hidden');

  const formData = new FormData();
  formData.append('image', selectedFile, selectedFile.name);

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await resp.json();
    if (resp.ok && data.success) {
      predictionText.textContent = `Prediction: ${data.prediction}`;
      confidenceText.textContent = `Confidence: ${data.confidence}%`;
      progressBar.style.width = `${data.confidence}%`;
      resultDiv.classList.remove('hidden');
    } else {
      alert(data.error || 'Prediction failed');
    }
  } catch (err) {
    alert('Server unavailable. Please try again later.');
  } finally {
    loading.classList.add('hidden');
    classifyBtn.disabled = false;
  }
});
