# nanoSparseAttention

<!-- <img src="./assets/logo.jpeg" width="400" alt="logo"> -->
<!-- <p align="center">
  <img src="./assets/logo.jpeg" width="400" alt="logo">
</p> -->
<!-- ![logo](./assets/logo.jpeg) -->

## Overview

nanoSparseAttention provides clean, educational implementations of recent Sparse Attention mechanisms for both prefilling and generation stages of LLM inference. The repository prioritizes clarity and understanding over performance, making it ideal for learning and experimentation.

We implemented a [Jupyter notebook](./notebooks/tutorial.ipynb) that provides:
1. Detailed explanation of Sparse Attention concepts
2. Step-by-step implementation walkthrough
3. Visualization of attention patterns
4. Performance comparisons between different methods

The notebook has been prepared for the purpose of [NeurIPS 2024 Dynamic Sparsity Workshop](https://dynamic-sparsity.github.io/) - check it out if you want to learn more about dynamic execution, not only in the context of self-attention!

### Key Features

- **Pure PyTorch Implementation**: All attention mechanisms are implemented in pure PyTorch for maximum clarity and ease of understanding.
- **Real-world Testing**: Uses Llama-3.2-1B-Instruct model and FiscalNote/billsum dataset for practical experiments.
- **Comprehensive Tutorial**: Includes a detailed Jupyter notebook explaining core concepts and implementations.
- TODO - easily extensible to new models, datasets, and attention patterns
- TODO - it supports both prefilling and generation stages, as well as combining the two

### Implemented Methods

#### Prefilling Stage
- **Local Window + Attention Sinks** [(Xiao et al, 2023)](https://arxiv.org/abs/2309.17453), [(Han et al, 2024)](https://arxiv.org/abs/2308.16137)
- **Vertical-Slash Attention** [(Jiang et al, 2024)](https://arxiv.org/abs/2407.02490)
- **Block-Sparse Attention** [(Jiang et al, 2024)](https://arxiv.org/abs/2407.02490)

#### Generation Stage
- **Local Window + Attention Sinks** [(Xiao et al, 2023)](https://arxiv.org/abs/2309.17453), [(Han et al, 2024)](https://arxiv.org/abs/2308.16137)
- **SnapKV** [(Li et al, 2024)](https://arxiv.org/abs/2404.14469)
- **TOVA** [(Oren et al, 2023)](https://arxiv.org/abs/2401.06104)

## Installation

```
    git clone https://github.com/PiotrNawrot/nano-sparse-attention
    cd nano-sparse-attention
    pip install -e .
```

## Example Usage

TODO

```
inference_handler = InferenceHandler(
    prefill_attention=LocalAndSinksAttention(
        window_size=640,
        attention_sinks=16,
    ),
    generation_attention=DenseAttention(),
)
```

## Contributing

Contributions are welcome! Our goal is to keep this repository up-to-date with the latest Sparse Attention methods, by consistently adding new methods. Feel free to submit a Pull Request if 1) you want a new method to be added or 2) [even better] you have an implementation of a new Sparse Attention method!

## Authors

Piotr Nawrot - [Website](https://piotrnawrot.github.io/) - piotr@nawrot.org

Edoardo Maria Ponti - [Website](https://ducdauge.github.io/) - eponti@ed.ac.uk
