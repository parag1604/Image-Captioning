import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ViTModel

from helpers.dataloader import get_dataset
from helpers.train_utils import train, infer
from helpers.utils import *


class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.encoder.pooler.dense = nn.Identity()

        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        return self.encoder(images)[0]


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size, hid_dim=768, num_layers=3, num_heads=8,
        max_length=25, dropout=0.1, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.num_rnn_layers = num_layers
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.tf_dec = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hid_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.umbedding = nn.Linear(hid_dim, vocab_size)
        self.max_length = max_length
        self.tgt_causal_mask = torch.triu(
            torch.ones(max_length, max_length) * float('-inf'), diagonal=1)
        self.tgt_causal_mask = self.tgt_causal_mask.to(device)
    
    def forward(self, enc_memory, gt_captions=None):
        device = enc_memory.device
        
        if gt_captions is not None:
            # teacher forcing
            decoder_inputs = torch.ones_like(gt_captions)
            masks = (gt_captions == 0).float()
            decoder_inputs[:, 1:] = gt_captions[:, :-1]
            
            decoder_hidden = self.dropout(self.embedding(decoder_inputs))
            for layer in self.tf_dec:
                decoder_hidden = layer(
                    tgt=decoder_hidden,
                    memory=enc_memory,
                    tgt_mask=self.tgt_causal_mask,
                    tgt_key_padding_mask=masks
                )
            
            decoder_outputs = self.umbedding(self.dropout(decoder_hidden))
        else:
            # autoregressive
            decoder_inputs = torch.ones(
                size=(enc_memory.size(0), 1),
                dtype=torch.long, device=device)
            for i in range(1, self.max_length + 1):
                decoder_hidden = self.dropout(self.embedding(decoder_inputs))
                for layer in self.tf_dec:
                    decoder_hidden = layer(
                        tgt=decoder_hidden,
                        memory=enc_memory,
                        tgt_mask=self.tgt_causal_mask[:i, :i],
                    )
                decoder_outputs = self.umbedding(self.dropout(decoder_hidden))
                
                _, topi = decoder_outputs[:, -1].topk(1)
                decoder_input = topi.squeeze(-1).detach().reshape(-1, 1)
                decoder_inputs = torch.cat([decoder_inputs, decoder_input], dim=1)

        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs
    

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
    encoder = ViTEncoder().to(device)
    decoder = TransformerDecoder(
        vocab_size, hid_dim=args.hid_dim, num_layers=args.num_rnn_layers,
        dropout=args.dropout, max_length=args.max_len, device=device).to(device)
    model = (encoder, decoder)

    # train / infer
    if args.is_train:
        optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
        train(model, optimizer, trainset, valset, device, word2idx, idx2word, args)
    else:
        infer(model, valset, device, word2idx, idx2word, args)


if __name__ == '__main__':
    args = ArgStorage(
        is_train=False,
        seed=2024,
        gpu_id=0,
        epochs=30,
        batch_size=256,
        image_size=224,
        min_word_count=5,
        max_len=25,
        lr=3e-4,
        dropout=0.2,
        max_norm=1.0,
        hid_dim=768,
        num_rnn_layers=4,
        num_workers=4,
        image_dir='data/flickr8k/images',
        caption_path='data/flickr8k/captions.txt',
        train_caption_path='data/flickr8k/train_captions.pkl',
        val_caption_path='data/flickr8k/val_captions.pkl',
        model_name='vit_tf',
        save_models=True,
        save_encoder=False
    )
    main(args)
