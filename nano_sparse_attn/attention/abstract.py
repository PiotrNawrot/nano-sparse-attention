import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Attention'
        self.masks = []
        self.sparsity_ratios = []
        self.params = {}
        self.layer_counter = 0

    def info(self):
        return {
            "name": self.name,
            "params": self.params,
            "masks": self.masks,
            "sparsity": self.calculate_average_sparsity_ratio(self.sparsity_ratios)
        }

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def maybe_save_mask(self, attention_mask, modulo_layers=8):
        if self.layer_counter % modulo_layers == 0:
            self.masks.append(attention_mask[0, 0].clone().cpu().numpy())

        self.layer_counter += 1

    def forward(self, queries, keys, values, *args, **kwargs):
        """Base forward method for prefilling attention patterns.
        
        Args:
            queries: (batch_size, num_heads, seq_len, head_dim)
            keys: (batch_size, num_heads, seq_len, head_dim)
            values: (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            attention_output: (batch_size, num_heads, seq_len, head_dim)
        """
        raise NotImplementedError

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        """Base forward method for generation attention patterns.
        
        Args:
            prefilling_queries: Queries from prefilling stage
            prefilling_keys: Keys from prefilling stage
            prefilling_values: Values from prefilling stage
            generation_queries: New queries for generation
            generation_keys: New keys for generation
            generation_values: New values for generation
            
        Returns:
            attention_output: Output for generation tokens
        """
        raise NotImplementedError

    @staticmethod
    def attention(queries, keys, values, attention_mask, return_attention_scores=False):
        """Standard attention computation."""
        attention_weights = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(queries.size(-1))
        attention_weights += attention_mask.to(queries.dtype) * torch.finfo(queries.dtype).min
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attention_output = torch.matmul(attention_weights, values)
        if return_attention_scores:
            return attention_output, attention_weights
        return attention_output
    
    @staticmethod
    def get_causal_mask(seq_len, device):
        """Creates a causal mask where future tokens cannot attend to past tokens.
        
        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, seq_len, seq_len) where True/1 indicates
                 that position (i,j) should be masked (set to -inf before softmax)
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def get_local_mask(seq_len, window_size, device):
        """Creates a local attention mask where tokens can only attend to nearby tokens
        within a fixed window size, plus the causal constraint.
        
        Args:
            seq_len: Length of the sequence
            window_size: Size of the local attention window including current token
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, seq_len, seq_len) where True/1 indicates
                 that position (i,j) should be masked (set to -inf before softmax)
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=-(window_size-1))
        mask = torch.tril(mask, diagonal=0)
        mask = (~mask)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def get_generation_mask(gen_len, prefill_len, device):
        """Creates a mask that allows generation tokens to:
        1. Attend to all prefilling tokens
        2. Attend causally to other generation tokens
        
        Args:
            gen_len: Number of tokens being generated
            prefill_len: Number of tokens in the prefill context
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, gen_len, prefill_len + gen_len)
                 where True indicates positions that should be masked
        """
        return torch.cat([
            torch.zeros((1, 1, gen_len, prefill_len), dtype=torch.bool, device=device),
            Attention.get_causal_mask(gen_len, device)
        ], dim=-1)

    @staticmethod
    def calculate_sparsity_ratio(mask):
        """Calculates the sparsity ratio of an attention mask.

        This method computes what fraction of the possible attention connections are masked 
        assuming that attention is causal, i.e., that tokens cannot attend to tokens before them.
        A higher ratio means more sparse attention. Asummes batch_size = 1.

        Args:
            mask: Boolean tensor of shape (batch_size, num_heads, queries_len, keys_len) where
                 True/1 indicates masked (disabled) attention connections

        Returns:
            float: The sparsity ratio between 0 and 1, where:
                  0 means all possible connections are enabled (dense attention)
                  1 means all possible connections are masked (completely sparse)
        """

        _, _, queries_len, keys_len = mask.shape

        if queries_len != keys_len:
            prefill_length = keys_len - queries_len
            total_connections = queries_len * (queries_len + 1) // 2 + prefill_length * queries_len
        else:
            total_connections = queries_len * (queries_len + 1) // 2

        connections_per_head = (~mask).long().sum(dim=(-1, -2))
        non_masked_ratio = (connections_per_head.float() / total_connections).mean(dim=-1).item()
        sparsity_ratio = 1 - non_masked_ratio

        return sparsity_ratio
    
    @staticmethod
    def calculate_average_sparsity_ratio(sparsity_ratios):
        return sum(sparsity_ratios) / len(sparsity_ratios) if sparsity_ratios else 0
    