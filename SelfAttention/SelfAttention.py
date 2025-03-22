from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
       
        self.W_query = nn.Linear(embedding_dim, hidden_dim)
        self.W_key = nn.Linear(embedding_dim, hidden_dim)
        self.W_value = nn.Linear(embedding_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    
    def forward(self, x):
        Q = self.W_query(x)
        print("Q:", Q)
        K = self.W_key(x)
        print("K: ", K)
        V = self.W_value(x)
        print("V:", V)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        attention_weights = F.softmax(attention, dim = -1)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights
    
class TextProcessor:
    def __init__(self, word2vec_path):
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary = True)
        self.embedding_dim = self.word2vec.vector_size

    def text_to_embeddings(self, text):
        words = text.lower().split()

        embeddings = []

        for word in words:
            try:
                embedding = self.word2vec[word]
                embeddings.append(embedding)

            except KeyError:
                embeddings.append(np.zeros(self.embedding_dim))

        return torch.FloatTensor(embeddings)
    
def main():
    word2vec_path = "GoogleNews-vectors-negative300.bin" #download this from "GoogleNews-vectors-negative300.bin.gz"

    text_processor = TextProcessor(word2vec_path)

    text = "Give a star to this repo, if you like it!"

    embeddings = text_processor.text_to_embeddings(text)

    embeddings = embeddings.unsqueeze(0)

    embedding_dim = text_processor.embedding_dim
    hidden_dim = 64
    self_attention = SelfAttention(embedding_dim, hidden_dim)

    output, attention_weights = self_attention(embeddings)

    print("Input_shape:", embeddings.shape)
    print("Output_shape:", output.shape)
    print("Attention_weights_shape:", attention_weights.shape)

if __name__ == "__main__":
    main()