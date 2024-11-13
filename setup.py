from setuptools import setup, find_packages

setup(
    name="nano_sparse_attn",
    version="0.99",
    author="Piotr Nawrot",
    author_email="piotr@nawrot.org",
    description="nanoSparseAttention: PyTorch implementation of novel sparse attention mechanisms",
    url="https://github.com/PiotrNawrot/nanoSparseAttention",
    packages=find_packages(include=['nano_sparse_attn', 'nano_sparse_attn.*']),
    install_requires=[
        "torch",
        "datasets",
        "transformers", 
        "matplotlib"
    ],
    python_requires=">=3.10",
)
