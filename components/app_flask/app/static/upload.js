const fileInput = document.getElementById('fileInput');
const textInput = document.getElementById('textInput');
const uploadButton = document.getElementById('uploadButton');
const preview = document.getElementById('preview');
const modelSelect = document.getElementById('model_name');
const epochsSelect = document.getElementById('epochs');


uploadButton.addEventListener('click', () => {
    const files = fileInput.files;
    const text = textInput.value;
    const model_name = modelSelect.value;
    const epochs = epochsSelect.value;

    // Basic preview (you can enhance this)
    preview.innerHTML = ''; // Clear previous previews 
    for (let i = 0; i < files.length; i++) {
        preview.innerHTML += `<p>${files[i].name}</p>`;
    }

    // Create form data for sending to Flask Server
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    formData.append('text', text);
    formData.append('model_name', model_name);
    formData.append('epochs', epochs);
    console.log("Hasta aqui jala bien");
    // Send to Flask using Fetch API (example)
    fetch('/handle_upload', { 
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            console.log('File uploaded successfully!');
        } else {
            console.error('Upload failed:', response.statusText);
        }
    });
});

// Make sure epochsSelect is correctly referencing the DOM element

for (let i = 1; i <= 20; i++) {
    const option = document.createElement('option');
    option.value = i;
    option.text = i;
    epochsSelect.appendChild(option);
}