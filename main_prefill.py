from nano_sparse_attn.utils import (
    load_model_and_tokenizer,
    load_examples,
    model_forward,
    update_attention,
    CONSTANTS,
)

from nano_sparse_attn.attention import (
    InferenceHandler,
    DenseAttention,
    LocalAndSinksAttention,
    VerticalAndSlashAttention,
    BlockSparseAttention,
)


model, tokenizer = load_model_and_tokenizer()
model_inputs = load_examples(
    tokenizer,
    target_length_min=CONSTANTS['runtime_args']['target_length_min'],
    target_length_max=CONSTANTS['runtime_args']['target_length_max'],
    num_examples=CONSTANTS['runtime_args']['num_examples'],
)

inference_handlers = []

# Dense baseline
inference_handlers.append(
    InferenceHandler(
        prefill_attention=DenseAttention(),
        generation_attention=DenseAttention(),
    )
)

for window_size in range(128, 1024+1, 128):
    inference_handlers.append(
        InferenceHandler(
            prefill_attention=LocalAndSinksAttention(
                window_size=window_size,
                attention_sinks=16,
            ),
            generation_attention=DenseAttention(),
        )
    )

for top_k in range(128, 1024+1, 128):
    inference_handlers.append(
        InferenceHandler(
            prefill_attention=VerticalAndSlashAttention(
                top_tokens=top_k,
                top_slashes=top_k,
                window_size=128,
                approximation_window=64,
                attention_sinks=16,
            ),
            generation_attention=DenseAttention(),
        )
    )

for top_chunks in range(2, 16+1, 2):
    inference_handlers.append(
        InferenceHandler(
            prefill_attention=BlockSparseAttention(
                chunk_size=64,
                top_chunks=top_chunks,
            ),
            generation_attention=DenseAttention(),
        )
    )

for idx, inference_handler in enumerate(inference_handlers):
    update_attention(model, inference_handler)
    loss = model_forward(model, model_inputs, inference_handler)
    info_dict = inference_handler.info()
    print(f"InferenceHandler {idx}: \n-- Name: {info_dict['prefill']['name']}\n-- Loss: {loss}\n-- Sparsity: {info_dict['prefill']['sparsity']}\n-- Params: {info_dict['prefill']['params']}")
