import torch
import torch.nn as nn

class InferenceHandler(nn.Module):
    """Handles different attention patterns for prefilling and generation stages.
    
    This class manages the splitting of input sequences into prefilling and generation
    stages, and applies the appropriate attention pattern for each stage.
    
    Args:
        prefill_attention: Attention module to use during prefilling
        generation_attention: Attention module to use during generation
    """
    def __init__(self, prefill_attention, generation_attention):
        super().__init__()
        self.prefill_attention = prefill_attention
        self.generation_attention = generation_attention
        self.name = f"{prefill_attention.name}_to_{generation_attention.name}"
    
    def set_output_length(self, output_length):
        self.output_length = output_length
    
    def info(self):
        return {
            "prefill": self.prefill_attention.info(),
            "generation": self.generation_attention.info()
        }
    
    def forward(self, queries, keys, values, layer_idx=None):
        """Handles attention computation for both prefilling and generation stages.
        
        Args:
            queries: (batch_size, num_heads, seq_len, head_dim)
            keys: (batch_size, num_heads, seq_len, head_dim)
            values: (batch_size, num_heads, seq_len, head_dim)
            layer_idx: Optional layer index for some attention implementations
            
        Returns:
            attention_output: Combined output from both stages
        """
        # Split sequence into prefill and generation parts
        input_length = queries.size(-2) - self.output_length
        assert input_length > 0, "Input length must be > 0"

        # Prefilling stage
        prefill_output = self.prefill_attention.forward(
            queries=queries[..., :input_length, :],
            keys=keys[..., :input_length, :],
            values=values[..., :input_length, :],
            layer_idx=layer_idx
        )
        
        # Generation stage
        generation_output = self.generation_attention.generation_forward(
            prefilling_queries=queries[..., :input_length, :],
            prefilling_keys=keys[..., :input_length, :],
            prefilling_values=values[..., :input_length, :],
            generation_queries=queries[..., input_length:, :],
            generation_keys=keys[..., input_length:, :],
            generation_values=values[..., input_length:, :],
            layer_idx=layer_idx
        )
        
        # Combine outputs
        return torch.cat([prefill_output, generation_output], dim=-2)
