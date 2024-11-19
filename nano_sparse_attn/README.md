## Technical Details & Extension

### Inference Handler & Attention Mechanism

The core of nanoSparseAttention is built around two key components:

1. **InferenceHandler** (`attention/inference_handler.py`): Manages the switching between prefilling and generation attention patterns. It allows you to:
   - Use different attention patterns for prefilling and generation stages
   - Track sparsity ratios and attention masks for visualization

2. **Base Attention** (`attention/abstract.py`): Provides the foundation for implementing sparse attention patterns with:
   - Common utility functions (`get_causal_mask`, `get_local_mask`, etc.)
   - Sparsity calculation and mask tracking
   - Standard attention computation methods

### Creating New Sparse Patterns

To implement a new sparse attention pattern:

1. Inherit from the base `Attention` class
2. Implement either/both:
   - `forward()` for prefilling attention
   - `generation_forward()` for generation attention
3. Create attention masks that specify which connections to keep/prune

Example skeleton:
```python
class MyNewAttention(Attention):
    def init(self, window_size):
        super().init()
        self.name = 'MyNewAttention'
        self.params = {"window_size": window_size}

    def forward(self, queries, keys, values, args, kwargs):
        # Create your attention mask
        attention_mask = ...

        # Track sparsity
        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))

        # Compute attention
        return self.attention(queries, keys, values, attention_mask)
```

### Supporting New Models

The repository only supports Llama-style attention mechanisms. To add support for a new model:

1. Locate the attention implementation in HuggingFace's transformers library (e.g., `modeling_llama.py` for Llama)
2. Copy and adapt the attention's forward pass to use our inference handler
3. Ensure correct tensor shapes for queries, keys, and values

### Using Custom Datasets

The repository can be adapted to work with any dataset. The key requirements are:

1. Implement a data loading function that returns examples in the format:
```python
{
    'input_ids': tensor, # Input token ids
    'attention_mask': tensor, # Attention mask
    'output_length': int, # Length of the output sequence
}
```

2. If using instruction-tuned models (like we do), format your prompts appropriately. Our implementation uses chat templates for Llama-style models, but this is configurable.

Example adaptation:
```python
def load_my_dataset(tokenizer, kwargs):
    # Load your data
    dataset = ...
    examples = []

    for item in dataset:
        # Format your prompt
        input_text = f"Your prompt: {item['input']}"

        # Tokenize (with or without chat template)
        tokens = tokenizer(
            input_text,
            return_tensors="pt",
        )

        # Get output length for loss calculation
        output_length = ...
        examples.append({
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'output_length': output_length,
        })

    return examples 
```
