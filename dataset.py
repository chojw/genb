from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import pickle as cPickle

import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from functools import partial
import pickle

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-',
                   ' ').replace('.','').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img_idx, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'img_idx'       : img_idx,
        'question'    : question['question'],
        'answer'      : answer
    }
    return entry


def _load_dataset(dataroot, name, img_id2val, dataset):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if dataset=='cpv2':
      answer_path = os.path.join(dataroot, 'cp-cache', '%s_target.pkl' % name)
      name = "train" if name == "train" else "test"
      question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)
    elif dataset=='cpv1':
      answer_path = os.path.join(dataroot, 'cp-v1-cache', '%s_target.pkl' % name)
      name = "train" if name == "train" else "test"
      question_path = os.path.join(dataroot, 'vqacp_v1_%s_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)
    elif dataset=='v2':
      answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
      question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)["questions"]

    with open(answer_path, 'rb') as f:
      answers = cPickle.load(f)

    questions.sort(key=lambda x: x['question_id'])
    answers.sort(key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    print(len(questions))
    for question, answer in tqdm(zip(questions, answers), ncols=100, desc="load-dataset"):
        if answer["labels"] is None:
            raise ValueError()
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        img_idx = None
        if img_id2val:
            if img_id in img_id2val['train']:
                img_idx = img_id2val['train'][img_id]
            else:
                img_idx = img_id2val['val'][img_id]

        entries.append(_create_entry(img_idx, question, answer))
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', dataset='cpv2',
                 use_hdf5=True, cache_image_features=False):
        super(VQAFeatureDataset, self).__init__()
        self.name=name

        assert name in ['train', 'val']

        if dataset=='cpv2':
            ans2label_path = os.path.join(dataroot, 'cp-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cp-cache', 'trainval_label2ans.pkl')
        elif dataset=='cpv1':
            ans2label_path = os.path.join(dataroot, 'cp-v1-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cp-v1-cache', 'trainval_label2ans.pkl')
        elif dataset=='v2':
            ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.use_hdf5 = use_hdf5

        if use_hdf5:
            self.features = {}
            self.spatial = {}
            self.image_id2ix = {}
            h5_path_train = os.path.join(dataroot, 'detection_features','train36.hdf5')
            self.hf_train = h5py.File(h5_path_train, 'r')
            self.features['train'] = self.hf_train.get('image_features')
            self.spatial['train'] = self.hf_train.get('spatial_features')
            h5_path_test = os.path.join(dataroot, 'detection_features','val36.hdf5')
            self.hf_test = h5py.File(h5_path_test, 'r')
            self.features['val'] = self.hf_test.get('image_features')
            self.spatial['val'] = self.hf_test.get('spatial_features')
            self.image_id2ix['train'] = pickle.load(open(os.path.join(dataroot, 'detection_features', f'train36_imgid2img.pkl'), 'rb'))
            self.image_id2ix['val'] = pickle.load(open(os.path.join(dataroot, 'detection_features', f'val36_imgid2img.pkl'), 'rb'))
        else:
            self.image_id2ix = None

        self.entries = _load_dataset(dataroot, name, self.image_id2ix, dataset=dataset)
        if cache_image_features:
            image_to_fe = {}
            for entry in tqdm(self.entries, ncols=100, desc="caching-features"):
                img_id = entry["image_id"]
                if img_id not in image_to_fe:
                    if use_hdf5:
                        if img_id in self.image_id2ix['train']:
                            fe = np.array(self.features['train'][self.image_id2ix['train'][img_id]])
                        else:
                            fe = np.array(self.features['val'][self.image_id2ix['val'][img_id]])
                        fe = torch.from_numpy(fe)
                    if use_hdf5:
                        self.hf_train.close()
                        self.hf_test.close()
                    else:
                        fe=torch.load('data/rcnn_feature/'+str(img_id)+'.pth')['image_feature']
                    image_to_fe[img_id]=fe
            self.image_to_fe = image_to_fe
        else:
            self.image_to_fe = None

        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in tqdm(self.entries, ncols=100, desc="tokenize"):
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                padding_mask=[self.dictionary.padding_idx-1] * (max_length - len(tokens))
                tokens_mask = padding_mask + tokens
                tokens = padding + tokens

            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            entry['q_token_mask']=tokens_mask
                
    def tensorize(self):
        for entry in tqdm(self.entries, ncols=100, desc="tensorize"):
            question = torch.from_numpy(np.array(entry['q_token']))
            question_mask = torch.from_numpy(np.array(entry['q_token_mask']))

            entry['q_token'] = question
            entry['q_token_mask']=question_mask

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        img_id = entry['image_id']
        
        if self.image_to_fe is not None:
            features = self.image_to_fe[entry["image_id"]]
        elif self.use_hdf5:
            if img_id in self.image_id2ix['train']:
                split = 'train'
            else:
                split = 'val'
            features = np.array(self.features[split][entry['img_idx']])
            features = torch.from_numpy(features)
        else:
            features = torch.load('data/rcnn_feature/' + str(entry["image_id"]) + '.pth')['image_feature']

        q_id=entry['question_id']
        ques = entry['q_token']
        ques_mask=entry['q_token_mask']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, ques, target, q_id

    def __len__(self):
        return len(self.entries)

