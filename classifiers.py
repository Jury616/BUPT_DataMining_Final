# classifiers.py
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

import torch
import torch.nn as nn

def get_multinomial_nb_classifier():
    """
    获取朴素贝叶斯分类器 (MultinomialNB) 实例
    """
    return MultinomialNB()

def get_gaussian_nb_classifier():
    """
    获取高斯朴素贝叶斯分类器 (GaussianNB) 实例
    """
    return GaussianNB()

def get_svm_classifier(C=1.0, max_iter=1000):
    """
    获取 SVM 分类器 (LinearSVC) 实例
    """
    return LinearSVC(C=C, max_iter=max_iter, penalty='l2', loss='squared_hinge', dual=False, random_state=42)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 n_layers=1, bidirectional=False, dropout_rate=0.5,
                 embedding_matrix=None, trainable_embedding=True, pad_idx=None):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=not trainable_embedding,
                padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout_rate if n_layers > 1 else 0, # Dropout between LSTM layers if n_layers > 1
                            batch_first=True)

        self.dropout = nn.Dropout(dropout_rate) # Dropout after LSTM layer

        # Adjust linear layer input dimension based on bidirectionality
        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(linear_input_dim, num_classes)

    def forward(self, text_sequences):
        # text_sequences shape: (batch_size, seq_len)
        embedded = self.embedding(text_sequences)
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Using simple sequence (assuming padding is handled or less critical for this setup)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden shape: (n_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (n_layers * num_directions, batch_size, hidden_dim)

        # We use the output of the last time step for classification
        # If bidirectional, lstm_out[:, -1, :] contains concatenated forward and backward hidden states of the last layer
        final_output = lstm_out[:, -1, :]

        final_output_dropped = self.dropout(final_output)
        logits = self.fc(final_output_dropped)
        # logits shape: (batch_size, num_classes)
        return logits

# Helper function to create the LSTM model instance
def create_lstm_model(vocab_size, embedding_dim, hidden_dim, num_classes,
                              n_layers=1, bidirectional=False, dropout_rate=0.5,
                              embedding_matrix=None, trainable_embedding=True, pad_idx=None):
    model = LSTM(vocab_size, embedding_dim, hidden_dim, num_classes,
                        n_layers, bidirectional, dropout_rate,
                        embedding_matrix, trainable_embedding, pad_idx)
    return model