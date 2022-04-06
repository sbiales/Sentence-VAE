import os
import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate, tokenize


def main(args):
    with open(args.data_dir + '/' + args.corpus + '.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    '''samples, z = model.inference(n=args.num_samples)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')'''

    # Sample from the examples given by Table 7 in the paper
    texts = ['we looked out at the setting sun.', 'i went to the kitchen.', 'how are you doing?']
    input_sequence = {
        'input': [],
        'length': []
    }

    # Arrange the text tokenizations and lengths
    for text in texts:
        tokenized = tokenize(text, w2i, args.max_sequence_length)
        input_sequence['input'].append(tokenized)
        input_sequence['length'].append(len(tokenized))

    input_sequence['input'] = to_var(torch.tensor(input_sequence['input']))
    input_sequence['length'] = to_var(torch.tensor(input_sequence['length']))

    mean, std = model.encode(input_sequence['input'], input_sequence['length'])

    mean_z = to_var(torch.zeros([len(texts), args.latent_size]))
    mean_z = mean_z * std + mean
    mean_samples, _ = model.inference(n=1, z=mean_z)

    print('----------SAMPLES----------')
    for i in range(len(texts)):
        # Get samples based on the mean/std of the input text
        z = to_var(torch.randn([args.num_samples, args.latent_size]))
        z = z * std[i, :] + mean[i, :]
        samples, _ = model.inference(n=args.num_samples, z=z)

        print('Input:', texts[i])
        print('Mean: ', *idx2word(mean_samples[i, :].unsqueeze(0), i2w=i2w, pad_idx=w2i['<pad>']))
        print('Samples:', *idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        print()

    # Interpolate between sentences given by Table 8 in the paper
    texts = ['he was silent for a long moment.', 'it was my turn.']
    input_sequence = {
        'input': [],
        'length': []
    }

    # Arrange the text tokenizations and lengths
    for text in texts:
        tokenized = tokenize(text, w2i, args.max_sequence_length)
        input_sequence['input'].append(tokenized)
        input_sequence['length'].append(len(tokenized))

    input_sequence['input'] = to_var(torch.tensor(input_sequence['input']))
    input_sequence['length'] = to_var(torch.tensor(input_sequence['length']))

    mean, std = model.encode(input_sequence['input'], input_sequence['length'])

    mean_z = to_var(torch.zeros([len(texts), args.latent_size]))
    mean_z = mean_z * std + mean
    z1 = mean_z[0, :].cpu().detach().numpy() 
    z2 = mean_z[1, :].cpu().detach().numpy() 
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=4)).float())
    samples, _ = model.inference(z=z)

    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-cp', '--corpus', type=str, default='ptb')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.corpus in ['ptb', 'books']
    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
