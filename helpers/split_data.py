import pickle
import numpy as np

from helpers.utils import extract_image_path_and_captions

np.random.seed(2024)


val_size = 500
with open('../data/flickr8k/captions.txt', 'r') as f:
    lines = f.readlines()
head = lines[0]
lines = lines[1:]

captions = {}
for line in lines:
    image_path, caption = extract_image_path_and_captions(line)
    if image_path not in captions:
        captions[image_path] = []
    captions[image_path].append(caption)

idxs = np.random.permutation(len(captions))
val_idxs = idxs[:val_size]
train_idxs = idxs[val_size:]

train_captions = {}
val_captions = {}
for idx, image_path in enumerate(captions):
    if idx in train_idxs:
        train_captions[image_path] = captions[image_path]
    else:
        val_captions[image_path] = captions[image_path]

with open('../data/flickr8k/train_captions.pkl', 'wb') as f:
    pickle.dump(train_captions, f)

with open('../data/flickr8k/val_captions.pkl', 'wb') as f:
    pickle.dump(val_captions, f)