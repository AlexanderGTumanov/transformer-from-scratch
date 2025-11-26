# Transformer from Scratch

This project implements a minimal Transformer language model using PyTorch. The goal is educational rather than performance-oriented: every major component of a modern Transformer is written manually, including token and positional embeddings, causal self-attention, residual connections, layer normalization, and the feed-forward network. The implementation is fully transparent and does not rely on pre-built convenience layers.

The model uses character-level tokenization and is trained on the full text of The Wizard of Oz. It is a context model trained to predict the next character in a sequence. Although small, it successfully learns common English patterns and can generate short, coherent continuations. Sampling utilities (temperature scaling and top-k filtering) are included to improve the quality of the generated text. A pre-trained model (trained overnight on a Mac M1 using the MPS backend) is provided for experimentation.

The project is organized into three directories. The `/notebooks` folder contains a Jupyter notebook that walks through the model’s architecture, training, and output decoding. The `/src` folder includes the full PyTorch implementation in `utils.py`. The `/model` folder contains a pre-trained model that can be used without retraining.

---

## What It Does

- Loads and preprocesses a text dataset
- Builds a Transformer language model from first principles using PyTorch
- Implements token and positional embeddings, causal self-attention, residual connections, and feed-forward blocks
- Trains the model to predict the next character in a sequence
- Provides sampling utilities (temperature scaling and top-k filtering) for controlled text generation
- Illustrates the model’s behavior by generating continuations for a selection of prompts

---
