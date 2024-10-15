import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ViTModel

from helpers.dataloader import get_dataset
from helpers.train_utils import train, infer
from helpers.utils import *


class ViTEncoder(nn.Module):
    def __init__(self, hid_dim=256, dropout=0.5):
        super(ViTEncoder, self).__init__()
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.encoder.pooler.dense = nn.Linear(768, hid_dim)
        self.dropout = nn.Dropout(p=dropout)

        # freeze the encoder
        for name, param in self.encoder.named_parameters():
            if "pooler.dense" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, images):
        return self.dropout(self.encoder(images)[0])


class RNNDecoder(nn.Module):
    def __init__(
        self, vocab_size, hid_dim=256, num_layers=1, max_length=25, dropout=0.3
    ):
        super(RNNDecoder, self).__init__()
        self.num_rnn_layers = num_layers
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.rnn_dec = nn.LSTM(hid_dim, hid_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.umbedding = nn.Linear(hid_dim, vocab_size)
        self.max_length = max_length
    
    def forward(self, image_features, gt_captions=None):
        device = image_features.device
        encoder_outputs = image_features[:, :1, :].permute(1, 0, 2).contiguous()
        decoder_input = torch.ones(
            size=(image_features.size(0), 1), dtype=torch.long).to(device)
        decoder_hidden = torch.zeros(
            size=(self.num_rnn_layers, image_features.size(0), self.hid_dim),
            device=device)
        for i in range(self.num_rnn_layers):
            decoder_hidden[i] = encoder_outputs
        rnn_state = (decoder_hidden, torch.zeros_like(decoder_hidden))
        decoder_outputs = []
        
        for i in range(self.max_length):
            decoder_output, rnn_state =\
                self.forward_step(decoder_input, rnn_state)
            
            decoder_outputs.append(decoder_output)
            
            if gt_captions is not None:
                decoder_input = gt_captions[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs
    
    def forward_step(self, decoder_input, rnn_state):
        tok_embeds = self.dropout(self.embedding(decoder_input))
            
        rnn_hidden, rnn_state = self.rnn_dec(tok_embeds, rnn_state)
        
        decoder_output = self.umbedding(self.dropout(rnn_hidden))

        return decoder_output, rnn_state


def main(args):
    set_seed(args.seed)
    
    device = torch.device(
        'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # generate/load vocab
    word2idx, idx2word = create_vocab(args.caption_path, args.min_word_count)
    vocab_size = len(idx2word)
    print('Generated vocab of size: {}'.format(vocab_size))

    # get dataset
    train_transform, test_transform = get_transforms(args.image_size)
    trainset = get_dataset(
        train_transform, args.image_dir, args.train_caption_path)
    valset = get_dataset(
        test_transform, args.image_dir, args.val_caption_path)
    print("Training set size: {}".format(len(trainset)))
    
    # get model
    encoder = ViTEncoder(hid_dim=args.hid_dim, dropout=args.dropout).to(device)
    decoder = RNNDecoder(
        vocab_size, hid_dim=args.hid_dim, num_layers=args.num_rnn_layers,
        dropout=args.dropout, max_length=args.max_len).to(device)
    model = (encoder, decoder)
    
    # train / infer
    if args.is_train:
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
        train(model, optimizer, trainset, valset, device, word2idx, idx2word, args)
    else:
        infer(model, valset, device, word2idx, idx2word, args)


if __name__ == '__main__':
    import sys
    is_train = False
    if len(sys.argv) > 1:
        is_train = sys.argv[1] == 'train'
    args = ArgStorage(
        is_train=is_train,
        seed=2024,
        gpu_id=0,
        epochs=30,
        batch_size=384,
        image_size=224,
        min_word_count=5,
        max_len=25,
        lr=5e-3,
        dropout=0.5,
        max_norm=1.0,
        hid_dim=768,
        num_rnn_layers=2,
        num_workers=4,
        image_dir='data/flickr8k/images',
        caption_path='data/flickr8k/captions.txt',
        train_caption_path='data/flickr8k/train_captions.pkl',
        val_caption_path='data/flickr8k/val_captions.pkl',
        model_name='vit_lstm',
        save_models=True,
        save_encoder=True
    )
    main(args)
