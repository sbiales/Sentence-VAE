import os
import io
import json
import torch
from datetime import datetime
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from datasets import load_dataset
from tqdm import tqdm

from utils import OrderedCounter

class Books(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'books.' + split + '.txt')
        self.data_file = 'books.' + split + '.json'
        self.vocab_file = 'books.vocab.json'

        if create_data:
            print("Creating new %s books data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):
        print("Loading %s JSON data file"%(self.split.upper()))

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)

        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_raw_files(self):
        print('Did not find raw corpus files. Creating...')
        corpus = load_dataset('bookcorpus')['train'].train_test_split(test_size=10000)
        sp = corpus['train'].train_test_split(test_size=10000)
        corpus['train'] = sp['train']
        corpus['valid'] = sp['test']

        splits = ['train', 'valid', 'test']
        for s in splits:
            print('Creating', s, 'split...')
            raw_data_path = 'books.' + s + '.txt'
            corpus[s].to_csv(os.path.join(self.data_dir, raw_data_path), header=False, index=False)
            # with io.open(os.path.join(self.data_dir, raw_data_path), 'wb') as data_file:
            #         split = json.dumps(corpus[s], ensure_ascii=False)
            #         data_file.write(split.encode('utf8', 'replace'))

    def _read_raw_file(self, file_handler):
        tokenizer = TweetTokenizer(preserve_case=False)
        for i, line in enumerate(file_handler):
            words = tokenizer.tokenize(line)

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            item_dict = {'input': input, 'target': target, 'length': length}
            yield i, item_dict

    def _create_data(self):

        if self.split == 'train' and not os.path.exists(os.path.join(self.data_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        data = defaultdict(dict)

        if not os.path.exists(self.raw_data_path):
            self._create_raw_files()

        with open(self.raw_data_path, 'r') as file:
            with io.open(os.path.join(self.data_dir, self.data_file), 'w') as data_file:
                gen = self._read_raw_file(file)
                print('Creating JSON file')
                data_file.write('{'.rstrip('\n'))
                fid, fline = next(gen)
                data = '"' + str(fid) + '": ' + json.dumps(fline, ensure_ascii=False)
                data_file.write(data.rstrip('\n'))
                for id, line in tqdm(gen):
                    data = ', "' + str(id) + '": ' + json.dumps(line, ensure_ascii=False)
                    data_file.write(data.rstrip('\n'))                
                data_file.write('}')

        # with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
        #     data = json.dumps(data, ensure_ascii=False)
        #     data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocabulary can only be created for training file."

        print("Creating vocabulary file")

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        if not os.path.exists(self.raw_data_path):
            self._create_raw_files()
        
        with open(self.raw_data_path, 'r') as file:

            for i, line in tqdm(enumerate(file)):
                words = tokenizer.tokenize(line)
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." %len(w2i))
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
