import torch
from torch import nn
from torch.nn import functional as F

from data_utils.vocab import Vocab
from model.feature_extractor import FeatureExtractor
from model.encoder import Encoder
from model.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, extractor, vocab, image_h, enc_hid_dim, dec_hid_dim, emb_dim, dropout=0.1):
        super(Seq2Seq, self).__init__()
        
        self.feature_extractor = FeatureExtractor(extractor, enc_hid_dim)
        self.encoder = Encoder(image_h, enc_hid_dim, dec_hid_dim, dropout)
        self.decoder = Decoder(vocab, emb_dim, enc_hid_dim, dec_hid_dim, dropout)

    def forward(self, images, targets):
        '''
            images: (bs, c, h, w)
            targets: (bs, w)
        '''
        features = self.feature_extractor(images)
        encoded_features, hidden_states = self.encoder(features)
        # applying the teacher-forcing mechanisms
        outputs = torch.tensor([], dtype=images.dtype, device=images.device)
        for t in range(targets.shape[1]):
            input = targets[:, t].unsqueeze(-1)
            output, hidden_states, _ = self.decoder(input, hidden_states, encoded_features)
            outputs = torch.cat([outputs, output.unsqueeze(1)], dim=1) # append new predicted word to the sequence

        return F.log_softmax(outputs, dim=-1) # (bs, w, vocab_size)

    def get_predictions(self, images, vocab: Vocab, max_len: int):
        self.eval()

        features = self.feature_extractor(images)
        encoded_features, hidden_states = self.encoder(features)
        bs = images.shape[0]
        targets = torch.tensor([vocab.sos_idx]*bs, dtype=torch.int, device=images.device).reshape(bs, 1)
        eos_mark = torch.tensor([0]*bs).to(float).to(images.device)
        for t in range(max_len):
            input = targets[:, -1].unsqueeze(-1)
            output, hidden_states, _ = self.decoder(input, hidden_states, encoded_features)
            # append new predicted word to the sequence
            target = output.argmax(dim=-1) # (bs, 1)
            target = torch.where(eos_mark == 1, vocab.padding_idx, target)
            targets = torch.cat([targets, target.unsqueeze(1)], dim=-1)
            eos_mark = torch.where(target.squeeze(-1) == vocab.eos_idx, 1., eos_mark)
            if eos_mark.mean() == 1:
                break

        self.train()

        return targets # (bs, w)