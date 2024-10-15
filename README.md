# Image Captioning

Pytorch code for Image captioning using ViT Encoder and the following Decoders:  (i) Vanilla LSTM, (ii) Attention LSTM, (iii) Bahdanau Attention LSTM, and (iv) Transformers. (ii) and (iii) are used for explainablity

---

## Python Environment Setup

Ensure the following is installed in your python environment:

- Pytorch - required for standard nn llibraries and automatic backpropagation
- Torchvision - required for libraries related to computer vision
- Matploitlib - required for plotting
- Transformers (Huggingface): Required for Pretrained Vision Transformers
- NLTK - required for calculating BLEU score

## Dataset and preparation

After cloning, ensure to create the folders `data`, `models`, and `temp` by running:
```
$ mkdir data models temp
```
Then download the and extract the dataset [Kaggle - Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and paste the folder `flickr8k` inside the `data` folder.

## Vanilla LSTM Captioning


## Plots
![A plot showing how the model fits on the validaition set in the BLEU metric](images/plot.png "Val BLUE scores")
