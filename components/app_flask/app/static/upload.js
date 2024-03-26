const fileInput = document.getElementById('fileInput');
const textInput = document.getElementById('textInput');
const uploadButton = document.getElementById('uploadButton');
const preview = document.getElementById('preview');

uploadButton.addEventListener('click', () => {
    const files = fileInput.files;
    const text = textInput.value;

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
