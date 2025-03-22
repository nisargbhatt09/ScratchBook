# Self-Attention with Word2Vec

This project implements a Self-Attention mechanism using Word2Vec embeddings in PyTorch. It demonstrates how to process text data through word embeddings and apply self-attention to capture relationships between words in a sequence.

## Features

- Text to Word2Vec embedding conversion
- Self-Attention implementation using PyTorch
- Batch processing support
- Attention weight visualization capability

## Requirements

```bash
torch>=1.7.0
numpy>=1.19.2
gensim>=4.0.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/nisargbhatt09/ScratchBook
cd ScratchBook/SelfAttention
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the Word2Vec model:
   - Download Google's pre-trained Word2Vec model from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
   - Extract the `.bin.gz` file
   - Place the `.bin` file in your project directory

## Usage

### Basic Usage

```python
from SelfAttention import TextProcessor, SelfAttention

# Initialize text processor with Word2Vec model
text_processor = TextProcessor("path_to_word2vec_model.bin")

# Prepare your text
text = "Give a star to this repo, if you like it!"

# Convert text to embeddings
embeddings = text_processor.text_to_embeddings(text)
embeddings = embeddings.unsqueeze(0)  # Add batch dimension

# Initialize self-attention
self_attention = SelfAttention(
    embedding_dim=text_processor.embedding_dim,
    hidden_dim=64
)

# Get output and attention weights
output, attention_weights = self_attention(embeddings)
```

### Code Structure

- `SelfAttention.py`: Contains the main implementation
  - `SelfAttention`: PyTorch module implementing the self-attention mechanism
  - `TextProcessor`: Handles text processing and embedding conversion
  - `main()`: Example usage and demonstration

## How It Works

1. **Text Processing**:
   - Input text is split into words
   - Each word is converted to its Word2Vec embedding
   - Unknown words are handled by zero vectors

2. **Self-Attention**:
   - Converts input embeddings into Query (Q), Key (K), and Value (V) vectors
   - Computes attention scores using scaled dot-product attention
   - Applies softmax to get attention weights
   - Produces final output by combining attention weights with values

## Architecture Details

- Input Shape: `(batch_size, sequence_length, embedding_dim)`
- Output Shape: `(batch_size, sequence_length, hidden_dim)`
- Attention Weights Shape: `(batch_size, sequence_length, sequence_length)`

## Example Output

```python
Input shape: torch.Size([1, 7, 300])  # For a 7-word sentence with 300d embeddings
Output shape: torch.Size([1, 7, 64])  # Hidden dimension of 64
Attention weights shape: torch.Size([1, 7, 7])  # Attention matrix for 7 words
```

## License

MIT

## Contributing

Feel free to open issues or submit pull requests for any improvements.
