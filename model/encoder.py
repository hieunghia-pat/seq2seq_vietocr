import torch 
from torch import nn 

class Encoder(nn.Module):
    def __init__(self, image_h, enc_hid_dim, dec_hid_dim, dropout=0.25):
        super().__init__()

        emb_dim = (image_h // 32) * enc_hid_dim
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        outputs: src_len x batch_size x hid_dim 
        hidden: batch_size x hid_dim
        """

        embedded = self.dropout(src)
        
        outputs, hidden = self.rnn(embedded)
                                 
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden