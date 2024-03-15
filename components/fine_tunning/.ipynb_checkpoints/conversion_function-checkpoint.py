import os
import time
import subprocess

def convert_checkpoints(
    weights_file, size, vocab_path, output_dir,
    convertion_https_dir="https://raw.githubusercontent.com/keras-team/keras-nlp/master/tools/gemma",
    conversion_script = "export_gemma_to_hf.py"
):
    """Downloads the conversion script and runs the Gemma to HuggingFace model conversion. 

    Args:
        f_weights_path (str): Path to the fine-tuned model weights.
        model_size (str):  The size of the model (e.g., "base", "large").
        f_vocab_path (str): Path to the fine-tuned vocabulary file.
        huggingface_model_dir (str): Output directory for the HuggingFace model.
    """
    os.environ["KERAS_BACKEND"] = "torch"
    # Download the conversion script
    if not os.path.exists(conversion_script):
        try:
            subprocess.run(["wget", "-nv", "-nc", f"{https://raw.githubusercontent.com/keras-team/keras-nlp/master/tools/gemma}/{conversion_script}"], check=True)
        except subprocess.SubprocessError as e:
            print(f"Download failed: {e}")
            exit(1) 

    start_time = time.time()

    # Run the conversion script (assuming 'KERAS_BACKEND' is set in the environment)
    try:
        subprocess.run([
            "python", 
            "export_gemma_to_hf.py", 
            "--weights_file", weights_file,
            "--size", size,
            "--vocab_path", vocab_path,
            "--output_dir", output_dir
        ], check=True)
    except subprocess.SubprocessError as e:
        print(f"Conversion failed: {e}")
        exit(1)
