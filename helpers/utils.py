import re
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from matplotlib import pyplot as plt


class ArgStorage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_transforms(image_size):
    image_size2 = int(image_size * 1.143)
    train_transform = transforms.Compose([
        transforms.Resize(image_size2, transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, test_transform


def extract_image_path_and_captions(line):
    idx = line.find('.jpg,')
    filename = line[:idx + 4]
    caption = line[idx + 5:]
    caption = caption.strip().lower()
    # replace everything except (a-z and space)
    caption = re.sub(r'[^a-z\s]', '', caption).strip()
    # remove multiple spaces
    caption = re.sub(r'\s+', ' ', caption)
    return filename, caption


def create_vocab(captions_path, min_word_count=5):
    word_counts = dict()
    with open(captions_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            _, caption = extract_image_path_and_captions(line)
            # split the caption into words
            words = caption.split(' ')
            for word in words:
                # check whether word is in vocab or not
                if word not in word_counts:  # if not in vocab, add to vocab
                    word_counts[word] = 1
                else:  # if already in vocab, increment count
                    word_counts[word] += 1
    # add the pad, sos, eos, and unk special tokens
    word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
    for word, count in word_counts.items():
        if count >= min_word_count:
            word2idx[word] = len(idx2word)
            idx2word[len(idx2word)] = word
    return word2idx, idx2word


def encode_caption(caption, word2idx, max_len):
    # split the caption into words
    words = caption.split(' ')
    # add <sos>, <eos> and <pad> tokens
    caption = words[:max_len - 1] + ['<eos>']
    while len(caption) < max_len:
        caption.append('<pad>')
    unk_idx = word2idx['<unk>']  # idx of unk token
    # convert words to indices
    return [word2idx[word] if word in word2idx else unk_idx for word in caption]


def decode_caption(caption, idx2word):
    if len(caption) == 0:
        return ''
    if type(caption) == torch.Tensor:
        caption = caption.tolist()
    # convert indices to words
    caption = [idx2word[idx] for idx in caption]
    # find the index of (first) <eos> token (if exists)
    if '<eos>' in caption:
        eos_idx = caption.index('<eos>')
    else:
        eos_idx = len(caption)
    # remove tokens <eos> onwards
    caption = caption[:eos_idx]
    return ' '.join(caption)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_attn(image, words, attn_weights):
    words = words[:24]
    str_len = len(words)
    mapping = {
         1: (1, 1),  2: (2, 1),  3: (3, 1),  4: (4, 1),  5: (3, 2),  6: (3, 2),
         7: (4, 2),  8: (4, 2),  9: (4, 3), 10: (4, 3), 11: (4, 3), 12: (4, 3),
        13: (5, 3), 14: (5, 3), 15: (5, 3), 16: (6, 3), 17: (6, 3), 18: (6, 3), 
        19: (6, 4), 20: (6, 4), 21: (6, 4), 22: (6, 4), 23: (6, 4), 24: (6, 4)}
    cols, rows = mapping[str_len]
    fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=(12, 8))
    for i in range(rows):
        for j in range(cols):
            word_idx = i * cols + j
            ax[i, j].axis('off')
            if word_idx >= len(words):
                image = np.ones_like(image)
                continue
            ax[i, j].imshow(image)
            ax[i, j].set_title(words[word_idx])
            weights = attn_weights[:, word_idx].view(14, 14).cpu().numpy()
            weights = (weights - weights.min())/(weights.max() - weights.min())
            weights = np.asarray(weights * 255, dtype=np.uint8)
            weights = Image.fromarray(weights).resize((224, 224))
            weights = weights.filter(ImageFilter.GaussianBlur(radius=15))
            weights = np.array(weights)
            ax[i, j].imshow(weights, alpha=0.4, extent=(0, 224, 224, 0))
    plt.savefig('images/image_exp.png', bbox_inches='tight')
    plt.close()
