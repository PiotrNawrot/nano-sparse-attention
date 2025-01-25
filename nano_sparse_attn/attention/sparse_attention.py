import math
import torch
import torch.nn.functional as F

from .abstract import Attention


class DenseAttention(Attention):
    def __init__(self):
        super().__init__()
        self.name = 'DenseAttention'

    def forward(self, queries, keys, values, *args, **kwargs):
        attention_mask = self.get_causal_mask(queries.size(-2), queries.device)
        self.maybe_save_mask(attention_mask)
        return self.attention(queries, keys, values, attention_mask)

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        keys = torch.cat([prefilling_keys, generation_keys], dim=-2)
        values = torch.cat([prefilling_values, generation_values], dim=-2)
        
        generation_mask = self.get_generation_mask(
            gen_len=generation_queries.size(-2),
            prefill_len=prefilling_keys.size(-2),
            device=generation_queries.device
        )
        
        return self.attention(generation_queries, keys, values, generation_mask)


class LocalAndSinksAttention(Attention):
    """Implements a sparse attention pattern combining local windows with global attention sinks.
    
    This attention mechanism reduces computational complexity by:
    1. Allowing each token to attend only to nearby tokens within a fixed window
    2. Designating the first K tokens as "attention sinks" that can be attended to by all tokens
    
    This creates a sparse pattern where most tokens have local connectivity, but important
    context tokens (sinks) maintain global connectivity. 

    Args:
        window_size (int): Size of the local attention window around each token
        attention_sinks (int): Number of initial tokens that serve as global attention sinks
    
    Reference Papers:
        - (Xiao et al, 2023) https://arxiv.org/abs/2309.17453
        - (Han et al, 2024) https://arxiv.org/abs/2308.16137
    """
    def __init__(self, window_size, attention_sinks):
        super().__init__()
        self.name = 'LocalAndSinksAttention'
        self.params = {
            "window_size": window_size,
            "attention_sinks": attention_sinks
        }
    
    def create_mask(self, seq_len, device):
        """Creates sparse attention mask with local windows and global sinks."""
        assert self.window_size <= seq_len, "Window size must be less or equal to sequence length"
        assert self.attention_sinks <= seq_len, "Number of attention sinks must be less or equal to sequence length"
        
        # Create base local attention window
        mask = self.get_local_mask(seq_len, self.window_size, device)

        # Allow attention to sink tokens
        mask[..., :, :self.attention_sinks] = 0
        mask = mask | self.get_causal_mask(seq_len, device)

        return mask
    
    def forward(self, queries, keys, values, *args, **kwargs):
        """Sparse attention for prefilling."""
        attention_mask = self.create_mask(queries.size(-2), queries.device)

        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))
        self.maybe_save_mask(attention_mask)

        return self.attention(queries, keys, values, attention_mask)

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        assert self.attention_sinks <= prefilling_queries.size(-2)
    
        total_keys = torch.cat([
            prefilling_keys,
            generation_keys,
        ], dim=-2)

        total_values = torch.cat([
            prefilling_values,
            generation_values,
        ], dim=-2)

        attention_mask = self.create_mask(total_keys.size(-2), generation_queries.device)
        attention_mask = attention_mask[..., -generation_queries.size(-2):, :]

        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))
        self.maybe_save_mask(attention_mask)

        return self.attention(generation_queries, total_keys, total_values, attention_mask)


class VerticalAndSlashAttention(Attention):
    """Implements a content-dependent sparse attention combining top tokens and diagonal patterns,
    known as Vertical-Slash, introduced by (Jiang et al, 2024) in the MInference 1.0 paper.
    
    This attention mechanism creates a sparse pattern by combining two types of connectivity:
    1. Dynamic attention to top-K most relevant tokens (content-based)
    2. Dynamic attention along top-K diagonal stripes (structure-based)
    
    The diagonal stripes (slashes) capture recurring patterns at fixed offsets, while
    top tokens capture semantic relevance. This combines structural and semantic sparsity.

    Args:
        attention_sinks (int): Number of initial tokens that serve as global attention sinks
        window_size (int): Size of the local attention window around each token
        top_tokens (int): Number of highest-scoring tokens to attend to globally
        top_slashes (int): Number of diagonal stripes to include in the pattern
        approximation_window (int): Size of the window used to approximate attention scores

    Reference Papers:
        - (Jiang et al, 2024) https://arxiv.org/abs/2407.02490
    """
    def __init__(self, attention_sinks, window_size, top_tokens, top_slashes, approximation_window):
        super().__init__()
        self.name = 'VerticalAndSlashAttention'
        self.params = {
            "attention_sinks": attention_sinks,
            "window_size": window_size,
            "top_tokens": top_tokens,
            "top_slashes": top_slashes,
            "approximation_window": approximation_window,
        }
    
    def fill_diagonals(self, L: int, indexes: torch.Tensor, device='cuda', chunk_size=1024):
        """
        Memory-efficient implementation to create a mask LxL with diagonals filled with False.
        Processes diagonals in chunks to reduce peak memory usage.
        
        Parameters:
        - L (int): The size of the square matrix.
        - indexes (torch.Tensor): Tensor of shape (B, H, M) with integer values which determine 
          the diagonals to be filled in the output matrix for each batch and head.
        - device: The device to perform computations on ('cuda' or 'cpu').
        - chunk_size: Size of chunks to process at once to manage memory usage.
        
        Returns:
        - mask (torch.Tensor): A boolean matrix of size (B, H, L, L) with specified diagonals filled with True.
        """
        assert (indexes <= 0).all().item(), "Indexes must be on or below diagonal."
        
        batch_size, num_heads, num_diagonals = indexes.shape
        
        # Create output tensor
        mask_dense = torch.ones((batch_size, num_heads, L, L), dtype=torch.bool, device=device)
        
        # Process the sequence length in chunks
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            chunk_len = chunk_end - chunk_start
            
            # Create row indices for this chunk
            row_indices = torch.arange(chunk_start, chunk_end, device=device, dtype=torch.int32)
            row_indices = row_indices.view(1, 1, 1, chunk_len)
            row_indices = row_indices.expand(batch_size, num_heads, num_diagonals, chunk_len)
            
            # Add the diagonal offsets to get column indices
            col_indices = row_indices + indexes.unsqueeze(-1).to(torch.int32)
            
            # Mask out indices that are out of bounds
            valid_mask = (col_indices >= 0) & (col_indices < L)
            
            if not valid_mask.any():
                continue
                
            # Create batch and head indices for valid positions only
            batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
            batch_idx = batch_idx.expand(-1, num_heads, num_diagonals, chunk_len)
            head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1)
            head_idx = head_idx.expand(batch_size, -1, num_diagonals, chunk_len)
            
            # Select only valid indices
            valid_batch_idx = batch_idx[valid_mask]
            valid_head_idx = head_idx[valid_mask]
            valid_row_idx = row_indices[valid_mask]
            valid_col_idx = col_indices[valid_mask]
            
            # Set the valid diagonal elements to False
            mask_dense[valid_batch_idx, valid_head_idx, valid_row_idx, valid_col_idx] = False
            
            # Free memory explicitly
            del row_indices, col_indices, valid_mask, batch_idx, head_idx
            
        return mask_dense

    def sum_over_diagonals(self, matrix):
        """Efficiently sum values along diagonals of the attention matrix.
        
        This method uses stride tricks to efficiently extract and sum diagonals:
        1. Pad the matrix with zeros on both sides to handle all possible diagonals
        2. Create a strided view that groups elements along diagonals
        3. Sum along each diagonal to get their total attention scores
        
        The strided operation creates a view where each row contains elements from
        one diagonal, allowing for efficient parallel summation.
        
        Args:
            matrix: Attention scores tensor of shape (batch_size, num_heads, queries, keys)
            
        Returns:
            Tensor of shape (batch_size, num_heads, queries + keys - 1) containing
            summed attention scores for each diagonal

        This function is based on the implementation from: 
            https://github.com/microsoft/MInference/blob/main/minference/modules/minference_forward.py#L101
        """
        batch_size, num_heads, queries, keys = matrix.shape
        zero_matrix = torch.zeros((batch_size, num_heads, queries, queries), device=matrix.device)
        matrix_padded = torch.cat((zero_matrix, matrix, zero_matrix), -1)
        matrix_strided = matrix_padded.as_strided(
            (
                batch_size,
                num_heads,
                queries,
                queries + keys
            ),
            (
                num_heads * queries * (2 * queries + keys),
                queries * (2 * queries + keys),
                2 * queries + keys + 1,
                1
            )
        )
        sum_diagonals = torch.sum(matrix_strided, 2)
        return sum_diagonals[:, :, 1:]

    def create_mask(self, queries, keys, seq_len, device):
        assert self.top_slashes <= seq_len
        assert self.top_tokens <= seq_len
        assert self.top_slashes >= self.window_size
        assert self.attention_sinks <= self.top_tokens

        # Approximate attention scores
        approx_queries = queries[..., -self.approximation_window:, :]
        attention_scores = torch.matmul(approx_queries, keys.transpose(-2, -1)) / math.sqrt(queries.size(-1))
        attention_scores[..., -self.approximation_window:] += self.get_causal_mask(self.approximation_window, device) * torch.finfo(queries.dtype).min
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(queries.dtype)
        
        # Get top tokens
        summed_scores = attention_scores.sum(dim=-2)
        summed_scores[..., :self.attention_sinks] = torch.inf
        top_k_indices = torch.topk(summed_scores, self.top_tokens, dim=-1, sorted=False).indices

        # Get top_k slashes
        top_k_slash = self.sum_over_diagonals(attention_scores)[..., :-self.approximation_window + 1]
        top_k_slash[..., -self.window_size:] = torch.inf
        top_k_slash = torch.topk(top_k_slash, self.top_slashes, -1).indices - (seq_len - 1)

        # Get final mask
        mask = self.fill_diagonals(seq_len, top_k_slash, device)
        mask.scatter_(-1, top_k_indices.unsqueeze(-2).expand(-1, -1, seq_len, -1), 0)
        mask = mask | self.get_causal_mask(seq_len, device)

        return mask

    def forward(self, queries, keys, values, *args, **kwargs):
        """Sparse attention for prefilling using top tokens and slashes."""
        attention_mask = self.create_mask(queries, keys, queries.size(-2), queries.device)
        
        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))
        self.maybe_save_mask(attention_mask)

        return self.attention(queries, keys, values, attention_mask)


class BlockSparseAttention(Attention):
    """Implements a Block-Sparse Attention pattern based on chunk-level relevance,
    introduced by (Jiang et al, 2024) in the MInference 1.0 paper.
    
    This attention mechanism reduces complexity by operating on chunks of tokens:
    1. Divides the sequence into fixed-size chunks.
    2. Computes chunk-level attention scores with averaged token representations.
    3. Allows each chunk to attend to the top-K most relevant chunks.
    
    This creates a coarse-grained sparse pattern where entire blocks of tokens attend
    to each other, making it efficient for long sequences while preserving approximate
    attention patterns.
    
    Contrary to the original paper, in this implementation we additionally:
    1. Include the local window of size chunk_size around each token because we found
       it crucial for performance and it is non trivial to be executed by this pattern
       for high sparsity ratios.
    2. Always pick the prefix chunk because we also found it crucial for performance but
       it is not always picked as top-1 chunk.

    For this reason, top_chunks = 1 BlockSparseAttention is equivalent to
    LocalAndSinksAttention(window_size=chunk_size, attention_sinks=chunk_size).

    Args:
        chunk_size (int): Size of each token chunk/block
        top_chunks (int): Number of highest-scoring chunks each chunk can attend to

    Reference Papers:
        - (Jiang et al, 2024) https://arxiv.org/abs/2407.02490
    """
    def __init__(self, chunk_size, top_chunks):
        super().__init__()
        self.name = 'BlockSparseAttention'
        self.params = {
            "chunk_size": chunk_size,
            "top_chunks": top_chunks,
        }

    def create_mask(self, queries, keys, seq_len, device):
        assert self.chunk_size < seq_len, "Chunk size must be smaller than sequence length"
        assert self.top_chunks > 0, "Must select at least one top chunk"
        assert self.chunk_size >= 8, "Recommended chunk size is >= 8."

        # Calculate number of chunks and padding needed
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        padded_seq_len = num_chunks * self.chunk_size
        padding_len = padded_seq_len - seq_len

        assert self.top_chunks <= num_chunks, "Cannot select more top chunks than available chunks"

        # Pad queries and keys if needed
        if padding_len > 0:
            queries_padded = torch.nn.functional.pad(queries, (0, 0, 0, padding_len))
            keys_padded = torch.nn.functional.pad(keys, (0, 0, 0, padding_len))
        else:
            queries_padded = queries
            keys_padded = keys

        # Reshape padded queries and keys to chunks
        query_chunks = queries_padded.reshape(queries.size(0), queries.size(1), num_chunks, self.chunk_size, -1)
        key_chunks = keys_padded.reshape(keys.size(0), keys.size(1), num_chunks, self.chunk_size, -1)

        # Compute chunk representations by averaging
        query_chunk_representations = query_chunks.mean(dim=-2)
        key_chunk_representations = key_chunks.mean(dim=-2)

        # Compute attention scores between chunk representations
        chunk_attention_scores = torch.matmul(query_chunk_representations, key_chunk_representations.transpose(-2, -1))
        
        # Add causal masking of upper triangle
        chunk_attention_scores.masked_fill_(self.get_causal_mask(num_chunks, device), float('-inf'))

        # Always pick the prefix chunk
        chunk_attention_scores[..., 0] = float('inf')

        # Get top-k key chunks for each query chunk
        top_k_chunk_indices = torch.topk(chunk_attention_scores, self.top_chunks, dim=-1, sorted=False).indices

        # Create a mask for top-k interactions
        top_k_mask = torch.ones((num_chunks, num_chunks), dtype=torch.bool, device=device)
        top_k_mask = top_k_mask.unsqueeze(0).unsqueeze(0).repeat(queries.size(0), queries.size(1), 1, 1)
        top_k_mask.scatter_(-1, top_k_chunk_indices, 0)
        
        # Expand mask to padded sequence length
        mask = top_k_mask.repeat_interleave(self.chunk_size, dim=-2).repeat_interleave(self.chunk_size, dim=-1)

        # Include the local window of size chunk_size around each token
        mask = mask & self.get_local_mask(padded_seq_len, self.chunk_size, queries.device)

        # Include the causal mask
        mask = mask | self.get_causal_mask(padded_seq_len, queries.device)

        # Remove padding from mask if needed
        if padding_len > 0:
            mask = mask[..., :seq_len, :seq_len]

        return mask

    def forward(self, queries, keys, values, *args, **kwargs):
        """Block-sparse attention for prefilling."""
        attention_mask = self.create_mask(queries, keys, queries.size(-2), queries.device)
        
        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))
        self.maybe_save_mask(attention_mask)

        return self.attention(queries, keys, values, attention_mask)


class SnapKVAttention(Attention):
    """Implements SnapKV's attention pattern for efficient text generation.
    
    This attention mechanism compresses the Pre-Filling KV Cache, therefore reducing
    memory usage during generation, by:
    1. Approximating attention scores using a suffix window of queries
    2. Using attention score pooling to identify important token clusters
    3. Keeping only the most relevant tokens plus a recent window
    
    Args:
        token_capacity (int): Maximum number of tokens to keep in compressed history
        approximation_window (int): Number of suffix tokens used to approximate attention scores
        kernel_size (int): Size of the pooling kernel for score aggregation
        
    Reference:
        - (Li et al, 2024) https://arxiv.org/abs/2404.14469
    """
    def __init__(self, token_capacity, approximation_window=64, kernel_size=7):
        super().__init__()
        self.name = 'SnapKVAttention'
        self.params = {
            "token_capacity": token_capacity,
            "approximation_window": approximation_window,
            "kernel_size": kernel_size,
        }

    def create_mask_for_prefill(self, queries, keys, seq_len, device):
        """Create mask for generation with compressed KV cache.
        
        It uses prefilling queries and keys to estimate which tokens will be important
        during generation and keeps only the most important tokens plus a recent window.

        The output mask only concerns the prefilling tokens and has to be extended to
        account for the tokens from the generation phase.
        """
        assert self.token_capacity >= self.approximation_window

        # Approximate attention scores
        approx_queries = queries[..., -self.approximation_window:, :]
        attention_scores = torch.matmul(approx_queries, keys.transpose(-2, -1)) / math.sqrt(queries.size(-1))
        attention_scores[..., -self.approximation_window:] += self.get_causal_mask(self.approximation_window, device) * torch.finfo(queries.dtype).min
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(queries.dtype)
        
        # Sum over queries to get per-key importance and apply average pooling
        key_importance = attention_scores.sum(dim=-2)
        key_importance = F.avg_pool1d(
            key_importance,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        )
        
        # Always keep the window tokens
        key_importance[..., -self.approximation_window:] = torch.inf

        # Keep only the top_k tokens
        mask = torch.ones((queries.size(0), queries.size(1), seq_len), dtype=torch.bool, device=device)
        top_indices = key_importance.topk(self.token_capacity, dim=-1).indices
        mask.scatter_(-1, top_indices, False)
        
        # Expand mask to proper attention shape
        mask = mask.unsqueeze(-2)
        
        return mask

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        # Concatenate prefilling and generation KV states
        keys = torch.cat([prefilling_keys, generation_keys], dim=-2)
        values = torch.cat([prefilling_values, generation_values], dim=-2)
        
        # Create mask for prefilling tokens
        prefill_mask = self.create_mask_for_prefill(prefilling_queries, prefilling_keys, prefilling_keys.size(-2), generation_queries.device)
        
        # Create dense causal mask for generation tokens
        attention_mask = self.get_generation_mask(
            gen_len=generation_queries.size(-2),
            prefill_len=prefilling_keys.size(-2),
            device=generation_queries.device
        ).repeat(generation_queries.size(0), generation_queries.size(1), 1, 1)

        # Combine masks
        attention_mask[..., :prefilling_keys.size(-2)] |= prefill_mask

        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))
        self.maybe_save_mask(attention_mask)
        
        return self.attention(generation_queries, keys, values, attention_mask)


class TOVAAttention(Attention):
    """Implements Token Omission Via Attention (TOVA) for efficient text generation.
    
    This attention mechanism dynamically prunes the KV cache during generation by:
    1. Using the last prefilling token to identify initially important tokens
    2. During generation, maintaining a fixed-size cache by removing the least attended token
       after processing each new token
    
    Args:
        token_capacity (int): Maximum number of tokens to keep in the pruned KV cache
        
    Reference:
        - (Oren et al, 2023) https://arxiv.org/abs/2401.06104
    """
    def __init__(self, token_capacity):
        super().__init__()
        self.name = 'TOVAAttention'
        self.params = {
            "token_capacity": token_capacity
        }

    def create_mask_for_prefill(self, queries, keys, seq_len, device):
        """Create initial mask based on attention scores from the last prefilling token."""
        if self.token_capacity >= seq_len:
            return torch.zeros((queries.size(0), queries.size(1), 1, seq_len), dtype=torch.bool, device=device)
        
        # Get attention scores for the last prefilling token
        last_query = queries[..., -1:, :]
        attention_scores = torch.matmul(last_query, keys.transpose(-2, -1)) / math.sqrt(queries.size(-1))
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(queries.dtype)
        
        # Average attention scores across heads
        mean_attention_scores = attention_scores.mean(dim=1)
        
        # Create mask keeping only top-k tokens
        mask = torch.ones((queries.size(0), queries.size(1), seq_len), dtype=torch.bool, device=device)
        top_indices = mean_attention_scores.topk(self.token_capacity, dim=-1).indices
        mask.scatter_(-1, top_indices.expand(-1, queries.size(1), -1), False)
        
        # Expand mask to proper attention shape
        mask = mask.unsqueeze(-2)

        return mask

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        # Get initial mask for prefilling tokens
        current_mask = self.create_mask_for_prefill(
            prefilling_queries, 
            prefilling_keys, 
            prefilling_keys.size(-2), 
            generation_queries.device
        )
        
        # Initialize lists to store outputs and updated keys/values
        outputs = []
        current_keys = prefilling_keys
        current_values = prefilling_values

        # Intialise final mask we'll output
        attention_mask = torch.ones((
            generation_queries.size(0),
            generation_queries.size(1),
            generation_queries.size(2),
            prefilling_keys.size(-2) + generation_keys.size(-2),
        ), dtype=torch.bool, device=generation_queries.device)
        
        # Process generation tokens one by one
        for idx in range(generation_queries.size(-2)):
            # Get current generation token
            current_query = generation_queries[..., idx:idx+1, :]
            current_gen_key = generation_keys[..., idx:idx+1, :]
            current_gen_value = generation_values[..., idx:idx+1, :]
            
            # Extend keys and values
            current_keys = torch.cat([current_keys, current_gen_key], dim=-2)
            current_values = torch.cat([current_values, current_gen_value], dim=-2)
            
            # Extend mask for the new token (always attended)
            current_mask = torch.cat([
                current_mask,
                torch.zeros((current_query.size(0), current_query.size(1), 1, 1), dtype=torch.bool, device=current_query.device)
            ], dim=-1)

            attention_mask[..., idx:idx+1, :current_keys.size(-2)] = current_mask
            
            # Compute attention with scores
            output, attention_scores = self.attention(
                current_query, 
                current_keys,
                current_values,
                current_mask,
                return_attention_scores=True
            )
            outputs.append(output)

            # If we exceed capacity, mask the token with lowest attention score
            if current_keys.size(-2) > self.token_capacity:
                # Set scores to inf where tokens were already masked
                attention_scores = attention_scores.masked_fill(current_mask, float('inf'))

                # Average attention scores across heads
                mean_scores = attention_scores.mean(dim=1, keepdim=True)

                # Find token with lowest attention score
                min_indices = mean_scores.argmin(dim=-1, keepdim=True)
                min_indices = min_indices.expand(-1, current_query.size(1), -1, -1)

                # Update mask to exclude the lowest scoring token
                current_mask.scatter_(-1, min_indices, True)
        
        # Concatenate all outputs
        final_output = torch.cat(outputs, dim=-2)
        
        self.sparsity_ratios.append(self.calculate_sparsity_ratio(attention_mask))
        self.maybe_save_mask(attention_mask)

        return final_output
