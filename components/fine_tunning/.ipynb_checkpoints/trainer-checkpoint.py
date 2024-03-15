import keras
import keras_nlp
import os

def finetune_gemma(data: list[str], model_paths:dict, model_name: str, rank_lora: int=6, sequence_length: int=256, epochs: int=2, batch_size: int=1) :
    # keras_nlp.models.GemmaCausalLM.from_preset(model)
    # Reduce the input sequence length to limit memory usage
    model = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
    model.summary()
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

    os.makedirs(model_paths['finetuned_model_dir'], exist_ok=True)
    model.save_weights(model_paths['finetuned_weights_path'])
    model.preprocessor.tokenizer.save_assets(model_paths['finetuned_model_dir'])
    finetuned_weights_path = model_paths['finetuned_weights_path']

    return finetuned_weights_path

if __name__ == '__main__':
    model_paths = {
        'finetuned_model_dir': './gemma_2b_en',
        'finetuned_weights_path': './gemma_2b_en/model.weights.h5'
    }
    data = ['Sender:\nFooooddd\n\nAndres Perez:\nComing :)', "Sender:\nCan I maybe borrow your iron?\n\nAndres Perez:\nIt's not my iron But yeah haha Or is it? 🤔🧐"]
    finetune_gemma(data=data, model_paths=model_paths, model_name = 'gemma_2b_en')
