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

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/AlexanderGTumanov/transformer-from-scratch>
   cd transformer-from-scratch

---

## Contents of the `/src` folder

The `utils.py` file contains all the functions and classes used to prepare data, build the Transformer model, train it, and generate forecasts. Below is a brief description of each component.

### `CharDataset(data, block_size)`

A PyTorch `Dataset` for character-level modeling.

- `__init__(self, data, block_size)`  
  Initializes the dataset with the encoded text `data` and a fixed context length `block_size`.

- `__len__(self)`  
  Returns the number of available training samples, determined by all possible positions of the context window within the data  
  (`len(self.data) - self.block_size - 1`).

- `__getitem__(self, idx)`  
  Returns the sample whose context window starts at position `idx`. The method returns `(x, y)` where:
  - `x` is the context window: `data[idx : idx + block_size]`  
  - `y` is the same window shifted by one position: `data[idx + 1 : idx + 1 + block_size]`

### `tokenize_text(filepath, process = True)`

Loads the text file at `filepath` and converts it to lowercase. If `process = True`, the function also removes line breaks, collapses repeated spaces, and strips specialized Unicode characters. The cleaned text is then tokenized at the character level to produce integer IDs. The function returns `(text, encoded, encoder, decoder)`, where:

- `text` is the cleaned text string  
- `encoded` is the list of integer token IDs  
- `encoder` maps characters to IDs  
- `decoder` maps IDs back to characters

### `prepare_dataloaders(encoded, block_size, batch_size = 16, valid_split = 0.2, seed = 42)`

Constructs a `CharDataset` from the encoded text, splits it into training and validation subsets, and wraps them in PyTorch `DataLoader` objects. The training loader is shuffled, while the validation loader is not. The arguments `batch_size` and `valid_split` control the batch size and the proportion of data allocated to validation. Returns `(train_loader, valid_loader)`.

### `TransformerBlock(dim, dropout = 0.1)`

A single Transformer block implementing:
- layer normalization before attention
- linear projections for queries (W_Q), keys (W_K), and values (W_V)
- causal (triangular) masking to prevent attention to future tokens
- dropout applied to attention outputs and MLP outputs
- a feed-forward network (`dim -> 4 * dim -> dim` with ReLU)

### `LanguageModel(vocab_size, block_size, dim, n_transformers)`

A PyTorch `nn.Module` class used to define the model.

- `__init__(self, vocab_size, block_size, dim, n_transformers)`  
  Initializes the model by creating token embeddings, positional embeddings, a stack of `n_transformers` Transformer blocks, and a final linear projection layer. Here, `vocab_size` is the number of unique characters, `block_size` is the context length, `dim` is the internal embedding dimension representing tokens and their positions, and `n_transformers` is the number of Transformer blocks in the architecture.

- `forward(self, batch)`  
  Takes input of shape `(batch_size, sequence_length)` and returns logits of shape `(batch_size, sequence_length, vocab_size)`, where each `vocab_size`-dimensional vector contains unnormalized log-probabilities for the next character. During generation, only the final logit vector in each sequence is used.

### `train_model(model, train_loader, valid_loader, lr, min_epochs = 100, max_epochs = None, patience = 50)`

Trains the language model using Adam and cross-entropy loss, tracking both training and validation loss at each epoch. The model trains for up to `max_epochs` epochs, with early stopping triggered if validation loss fails to improve for more than `patience` epochs, while still guaranteeing at least `min_epochs` epochs of training. The function automatically selects the best available device (CUDA, MPS, or CPU). Returns `(model, history)`, where `history` contains the recorded loss curves.

### `forecast(model, text, horizon, encoder, decoder, block_size, temperature = None, top_k = None)`

Generates text by repeatedly predicting the next character until `horizon` additional characters are produced. The function encodes the prompt, feeds the last `block_size` characters into the model at each step, and selects the next character either greedily (if `temperature` is `None`) or via temperature-scaled softmax with optional top-k filtering. Returns the generated string.

### `forecast_sentence(model, text, encoder, decoder, block_size, temperature = None, top_k = None, cutoff = 300)`

Similar to `forecast`, but generation continues until the model produces a sentence-ending character (`.`, `?`, or `!`) or the optional `cutoff` limit is reached. Returns the generated sentence.

### `plot_loss_history(history)`

Given a `history` dictionary from `train_model`, plots training and validation loss per epoch.


