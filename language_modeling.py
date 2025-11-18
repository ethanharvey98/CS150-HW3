import torch
from torch import nn
import math
import time

# TODO: you are supposed to implement a language model and the training function to support the notebook.  
# The minimum starter code is given below. 

# NOTE: Please refer to the following resources for help: 
# * d2l Chapter 11.7 [link](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)

# NOTE: We provide the code below and try to best help you to construct the language model. But you are 
# suppose to guarantee the correctness of the code.  

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This is a standard component for Transformer models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SmallLanguageModel(torch.nn.Module):
    """
    A small language model using the transformer architecture
    
    *** REVISED to use nn.TransformerEncoder ***
    """

    def __init__(self, vocabulary, d_model=512, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        args: 
            vocabulary: a list or dict of characters or word roots.
            d_model: the number of expected features in the encoder/decoder inputs
            nhead: the number of heads in the multiheadattention models
            num_encoder_layers: the number of sub-encoder-layers in the encoder
            dim_feedforward: the dimension of the feedforward network model
            dropout: the dropout value
        """
        super().__init__()
        
        self.vocab_size = len(vocabulary)
        self.d_model = d_model
        
        # TODO: Please implement the following steps

        # 1. Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 3. Transformer with causal attention 
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        # 4. Final Linear Layer (Output)
        self.decoder = nn.Linear(d_model, self.vocab_size)
        

    def _generate_square_subsequent_mask(self, sz, device):
        """
        Generates the causal mask required for language modeling
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X):
        """
        The forward function of the model.
        args:
            X: a tensor with shape `[seq_len, batch_size]`.
        returns:
            out: a tensor with shape `[seq_len, batch_size, len(vocabulary)]`.
        """

        # TODO: Please implement the function that compute the logits of the probability
        # p(x_t | x_{<t})

        device = X.device
        seq_len = X.size(0)
        mask = self._generate_square_subsequent_mask(seq_len, device)
        emb = self.embedding(X) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, mask)
        out = self.decoder(out)
        return out

# --- Helper Functions for Training (Unchanged) ---

def batchify(data, bsz):
    """
    Reshapes the 1D data tensor into [seq_len, batch_size]
    Args:
        data: a 1D tensor
        bsz: batch size
    """
    device = data.device
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit.
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt):
    """
    Gets a batch of data for training
    Args:
        source: data from batchify, shape [full_seq_len, batch_size]
        i: the index of the batch
        bptt: sequence length (backprop through time)
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # Target is the data shifted by one position
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def evaluate_loop(model, data_source,loss_func, bptt):
    model.eval() # Set model to evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            if len(data.shape) == 1: data = data.unsqueeze(1)
            output = model(data)
            # Reshape for loss function
            # output: [seq_len, batch_size, vocab_size] -> [seq_len * batch_size, vocab_size]
            # targets: [seq_len * batch_size]
            output_flat = output.view(-1, model.vocab_size)
            total_loss += len(data) * loss_func(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def train(model, train_data, val_data, loss_func, optimizer, scheduler, num_epochs = 2, bptt = 50):
    """
    The training function for language modeling
    """
    # TODO: Please implement the training function
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.
        for i in range(0, train_data.size(0) - 1, bptt):
            data, targets = get_batch(train_data, i, bptt)
            if len(data.shape) == 1: data = data.unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            output_flat = output.view(-1, model.vocab_size)
            loss = loss_func(output_flat, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += len(data) * loss.item()
        val_loss = evaluate_loop(model, val_data, loss_func, bptt)
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_data):.4f}, Val Loss: {val_loss:.4f}')
        if scheduler:
            scheduler.step()
