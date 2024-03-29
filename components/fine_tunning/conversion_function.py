import os
import subprocess
import argparse

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
            subprocess.run(["wget", "-nv", "-nc", f"{convertion_https_dir}/{conversion_script}"], check=True)
        except subprocess.SubprocessError as e:
            print(f"Download failed: {e}")
            exit(1)

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
    return output_dir


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Checkpoint conversion tool.")

    parser.add_argument("--weights-file", dest="weights_file", type=str, required=True,
                        help="Path to the weights file.")
    parser.add_argument("--size", type=str, required=True,
                        help="Size of the model (e.g., '2b', '7b').")
    parser.add_argument("--vocab-path", dest="vocab_file", type=str, required=True,
                        help="Path to the vocabulary file.")
    parser.add_argument("--output-dir", dest='output_dir', type=str, required=True,
                        help="Output directory for the converted model.")
    parser.add_argument("--conversion-https-dir", type=str,
                        help="Base URL of the conversion script repository.")
    parser.add_argument("--conversion-script", type=str, 
                        help="Name of the conversion script within the repository.")

    args = parser.parse_args()
    hparams = args.__dict__
    convert_checkpoints(
        weights_file=args.weights_file, 
        size=args.size,
        vocab_path=args.vocab_file,
        output_dir=args.output_dir
    ) 