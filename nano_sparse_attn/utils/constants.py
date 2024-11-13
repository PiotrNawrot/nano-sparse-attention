import torch


CONSTANTS = {
    'model_name': 'unsloth/Llama-3.2-1B-Instruct',
    'dataset_name': 'FiscalNote/billsum',
    'runtime_args': {
        'dtype': torch.bfloat16,
        'num_examples': 3,
        'target_length_min': 4096,
        'target_length_max': 4096 + 512,
        'device': "cuda",
    },
    'hf_kwargs': {
        'trust_remote_code': True,
    },
    'save_results': True,
}
