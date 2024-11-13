from typing import Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import Cache, repeat_kv, apply_rotary_pos_emb

from .constants import CONSTANTS


def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    batch_size, query_length, _ = hidden_states.size()
    num_heads, head_dim = self.num_heads, self.head_dim
    num_kv_heads, num_kv_groups = self.num_key_value_heads, self.num_key_value_groups

    queries = self.q_proj(hidden_states).view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    keys = self.k_proj(hidden_states).view(batch_size, query_length, num_kv_heads, head_dim).transpose(1, 2)
    values = self.v_proj(hidden_states).view(batch_size, query_length, num_kv_heads, head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(values, position_ids)
    else:
        cos, sin = position_embeddings

    queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        keys, values = past_key_value.update(keys, values, self.layer_idx, cache_kwargs)

    keys = repeat_kv(keys, num_kv_groups)
    values = repeat_kv(values, num_kv_groups)

    attention_output = self.attn(queries, keys, values, layer_idx=self.layer_idx)

    attention_output = attention_output.transpose(1, 2).reshape(batch_size, query_length, -1).contiguous()
    attention_output = self.o_proj(attention_output)

    return attention_output, None, past_key_value


def update_attention(model, inference_handler):
    def update_attention_recursive(module):
        if isinstance(module, LlamaAttention):
            module.forward = llama_attention_forward.__get__(module, LlamaAttention)
            module.attn = inference_handler

    model.apply(update_attention_recursive)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        CONSTANTS['model_name'],
        **CONSTANTS['hf_kwargs'],
    )

    model = AutoModelForCausalLM.from_pretrained(
        CONSTANTS['model_name'],
        torch_dtype=CONSTANTS['runtime_args']['dtype'],
        attn_implementation="eager",
        **CONSTANTS['hf_kwargs'],
    ).to(CONSTANTS['runtime_args']['device'])


    return model, tokenizer


def load_examples(tokenizer, target_length_min=3500, target_length_max=4096, num_examples=1):
    dataset = load_dataset(
        CONSTANTS['dataset_name'],
        split="train",
        **CONSTANTS['hf_kwargs'],
    )

    examples = []
    for _, example in enumerate(dataset):
        # Separate input prompt and output
        input_prompt = f"""Below is a US Congressional and California state bill. Please provide a concise summary of the bill.

Bill:
{example['text']}"""
        
        output_text = example['summary']
        
        # Apply chat template to input
        templated_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Create full sequence with answer
        full_sequence = templated_input + output_text
        
        # Check total length
        tokens = tokenizer(
            full_sequence,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        total_length = tokens['input_ids'].shape[1]
        if target_length_min <= total_length <= target_length_max:
            # Get output length for loss calculation
            output_length = len(tokenizer(output_text, add_special_tokens=False)['input_ids'])
            
            # Create final tokens
            model_inputs = tokens.to(CONSTANTS['runtime_args']['device'])
            
            examples.append({
                'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
                'output_length': output_length,
            })

            if len(examples) >= num_examples:
                break

    return examples


def model_forward(model, model_inputs, inference_handler):
    total_loss = 0
    
    for example in model_inputs:
        # Set output length for current example
        inference_handler.set_output_length(example['output_length'])
        
        with torch.no_grad():
            outputs = model(
                input_ids=example['input_ids'],
                attention_mask=example['attention_mask'],
            )

        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = example['input_ids'][..., 1:]

        # Calculate loss only over the output length
        output_length = example['output_length'] 
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1))[-output_length:],
            shift_labels.view(-1)[-output_length:]
        )
        total_loss += loss.item()
    
    # Return average loss across examples
    return total_loss / len(model_inputs)
