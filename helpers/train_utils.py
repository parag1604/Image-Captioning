import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from nltk.translate import bleu_score

from helpers.dataloader import get_dataloader
from helpers.utils import encode_caption,decode_caption, plot_attn


def train(model, optimizer, trainset, valset, device, word2idx, idx2word, args):
    encoder, decoder = model
    smoothening_function = bleu_score.SmoothingFunction().method2
    
    scores = [0.0008]
    valloader = get_dataloader(
        valset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    for epoch in range(args.epochs):
        trainloader = get_dataloader(
            trainset, args.batch_size, num_workers=args.num_workers)
        
        # train
        encoder.train()
        decoder.train()
        for itr, (images, gt_captions) in enumerate(trainloader):
            idxs = np.random.randint(0, 5, size=len(images))
            gt_captions = [gt_captions[idxs[i]][i] for i in range(len(images))]
            gt_captions = [
                encode_caption(caption, word2idx, args.max_len)\
                    for caption in gt_captions]
            gt_captions = torch.tensor(gt_captions).long().to(device)
            images = images.to(device)
            gt_captions = gt_captions.to(device)
            masks = (gt_captions != 0).float().view(-1)

            image_features = encoder(images)
            outputs = decoder(image_features, gt_captions)
            if len(outputs) == 2:
                log_probs = outputs[0]
            else:
                log_probs = outputs

            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                gt_captions.view(-1),
                reduction='none',
            )

            loss = (loss * masks).sum() / masks.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.max_norm)
            optimizer.step()

            if (itr+1) % 10 == 0:
                print(f'Epoch: {epoch+1}, Itr: {itr+1}, Loss: {loss.item()}')

        # validate
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_score, total_counts = 0, 0
            for images, gt_captions in valloader:
                images = images.to(device)

                image_features = encoder(images)
                outputs = decoder(image_features)
                if len(outputs) == 2:
                    log_probs = outputs[0]
                else:
                    log_probs = outputs
                
                for i in range(len(log_probs)):
                    gts = [
                        gt_captions[0][i],
                        gt_captions[1][i],
                        gt_captions[2][i],
                        gt_captions[3][i],
                        gt_captions[4][i]
                    ]
                    gts = [gt.split(' ') for gt in gts]
                    pred = log_probs[i].argmax(dim=-1).detach().cpu().tolist()
                    pred = decode_caption(pred, idx2word).split(' ')
                    score = bleu_score.sentence_bleu(
                        references=gts, hypothesis=pred,
                        smoothing_function=smoothening_function)
                    val_score += score
                    total_counts += 1
                    
            val_score /= total_counts
            scores.append(val_score)
            print(f'Epoch: {epoch+1}, Val BLEU Score: {round(val_score, 4)}')
        
        if args.save_models:
            if val_score >= max(scores):
                # save model
                if args.save_encoder:
                    torch.save(
                        encoder.encoder.pooler.state_dict(),
                        f'models/{args.model_name}_encoder.pth')
                torch.save(
                    decoder.state_dict(),
                    f'models/{args.model_name}_decoder.pth')
            
            # save scores
            np.save(f'temp/{args.model_name}_val_scores.npy', scores)


def infer(model, testset, device, word2idx, idx2word, args):
    encoder, decoder = model
    
    if args.save_encoder:
        encoder.encoder.pooler.load_state_dict(
            torch.load(f'models/{args.model_name}_encoder.pth'))
    decoder.load_state_dict(
        torch.load(f'models/{args.model_name}_decoder.pth'))
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        testloader = get_dataloader(
            testset, args.batch_size,
            num_workers=args.num_workers, shuffle=False)

        for images, gt_captions in testloader:
            gt_captions = [
                encode_caption(caption, word2idx, args.max_len)\
                    for caption in gt_captions[0]]
            gt_captions = torch.tensor(gt_captions)
            images = images.to(device)
            image_features = encoder(images)
            outputs = decoder(image_features)
            if len(outputs) == 2:
                log_probs = outputs[0]
                attn_weights = outputs[1]
            else:
                log_probs = outputs
                attn_weights = None
            token_ids = log_probs.argmax(dim=-1).detach().cpu()
            
            for i in range(len(images)):
                image = images[i]
                pred_caption = token_ids[i]
                gt_caption = gt_captions[i]
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                image = np.clip(
                    (image * np.array([0.229, 0.224, 0.225])) +\
                    np.array([0.485, 0.456, 0.406]), 0, 1)
                plt.imsave('images/image.png', np.clip(image, 0, 1))
                if attn_weights is not None:
                    words = decode_caption(pred_caption, idx2word).split(' ')
                    plot_attn(image, words, attn_weights[i])
                print("GT Caption:", decode_caption(gt_caption, idx2word))
                print("Pred Caption:", decode_caption(pred_caption, idx2word))
                input()
            break
