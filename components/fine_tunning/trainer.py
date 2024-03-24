import argparse
import sys
import keras
import keras_nlp
import os
import json

def finetune_gemma(
    data: list[str], model_paths:dict, fine_tune_flag: bool = True, model_name: str='gemma_2b_en',
    rank_lora: int=6, sequence_length: int=256, epochs: int=15, batch_size: int=1
) :
    # keras_nlp.models.GemmaCausalLM.from_preset(model)
    # Reduce the input sequence length to limit memory usage
    print("Print a few examples of the input data")
    print(data[:1])
    print("Fine tune Function section has ben called and started.")

    model = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
    model.summary()
    if fine_tune_flag:
        print("The model is been fine tuned. This condition is only until a GPU is available through cutom jobs vertex AI")
        model.backbone.enable_lora(rank=rank_lora)
        model.summary()
        model.preprocessor.sequence_length = sequence_length

        # Use AdamW (a common optimizer for transformer models)
        optimizer = keras.optimizers.AdamW(
            learning_rate=5e-5,
            weight_decay=0.01,
        )
        # Exclude layernorm and bias terms from decay
        optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
            sampler="greedy",
        )
        model.fit(data, epochs=epochs, batch_size=batch_size)
    else:
        print("The model is not fine tuned due to google cloud lacking support")

    os.makedirs(model_paths['finetuned_model_dir'], exist_ok=True)
    print("Saving the weights ", model_paths['finetuned_weights_path'])
    model.save_weights(model_paths['finetuned_weights_path'])
    model.preprocessor.tokenizer.save_assets(model_paths['finetuned_model_dir'])
    print("Saving model in ", model_paths['finetuned_model_dir'])
    finetuned_weights_path = model_paths['finetuned_weights_path']

    return finetuned_weights_path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training script arguments')  # Optional description

    # Data argument
    parser.add_argument('--data', dest='data', nargs='+', type=str, 
                        help='List of input data files (space-separated)')
    # Model paths (assuming you'll handle parsing the dictionary later)
    parser.add_argument('--model-paths', dest='model_paths', type=str, 
                        help='String representation of model paths dictionary')
    parser.add_argument("--fine-tune-flag", dest='fine_tune_flag', action="store_true",
                        help="Whether fine tunning should run")
    # Model name 
    parser.add_argument('--model-name', dest='model_name', 
                        default='gemma_2b_en', type=str, help='Name of the model')
    # Rank LoRA
    parser.add_argument('--rank-lora', dest='rank_lora', 
                        default=6, type=int, help='LoRA rank') 
    # Sequence length
    parser.add_argument('--sequence-length', dest='sequence_length', 
                        default=256, type=int, help='Input sequence length') 
    # Epochs
    parser.add_argument('--epochs', dest='epochs', 
                        default=2, type=int, help='Number of training epochs')
    # Batch size 
    parser.add_argument('--batch-size', dest='batch_size', 
                       default=1, type=int, help='Batch size')
    args = parser.parse_args()

    hparams = args.__dict__
    print(type(args.data), args.data)
    print(type(args.model_paths), json.loads(args.model_paths))
    print(args.fine_tune_flag)
    print(type(args.fine_tune_flag))

    finetune_gemma(
        data=args.data, 
        model_paths=json.loads(args.model_paths),
        fine_tune_flag=args.fine_tune_flag,
        model_name=args.model_name,
        rank_lora=args.rank_lora,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    ) 