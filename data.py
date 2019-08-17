# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
from torch.utils.data import Dataset, DataLoader
import options as opt

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}  #定义一个字典
        self.word2count = {}  
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 #including SOS and EOS

    def addSentence(self, sentence):
        #for word in sentence.split(' '):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class MyDataset(Dataset):

    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    vidpath = opt.trn_vid_path
    videos_path = glob.glob(os.path.join(vidpath, "*", "*", "*", "*"))
    videos = list(filter(lambda dir: len(os.listdir(dir)) == 75, videos_path))

    anno_path = opt.anno_path
    data = []
    for vid in videos:
        items = vid.split(os.path.sep)
        #print(items)
        data.append((vid, items[-4], items[-1])) 

    name2 = 'output_lang'
    output_lang = Lang(name2)
    k = 0
    for each in data:
        (vid, spk, name) = each
        name1 = os.path.join(anno_path, spk, 'align', name + '.align')
    
        with open(name1, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
            
            output_lang.addSentence(txt)
            
            #if k == 2:
            #    print('k=2: ', output_lang.index2word)

        k += 1
    
    vocabulary = output_lang.index2word
    indexs = output_lang.word2index

    print(vocabulary)
    print(vocabulary[4])

    def __init__(self, vid_path, anno_path, vid_pad, txt_pad, phase):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        self.videos = glob.glob(os.path.join(vid_path, "*", "*", "*", "*"))
        self.videos = list(filter(lambda dir: len(os.listdir(dir)) == 75, self.videos))
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)            
            self.data.append((vid, items[-4], items[-1]))        

    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        #print(self.data[idx])
        vid = self._load_vid(vid)
        anno, txt = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align'))
        #print('anno: ', anno)
        #print("text: ", txt)
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        vid = ColorNormalize(vid)
        
        return {'encoder_tensor': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
                'decoder_tensor': torch.LongTensor(anno)}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        #files = sorted(os.listdir(p))
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))        
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (100, 50)) for im in array]
        array = np.stack(array, axis=0)
        return array

    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
            
        #return MyDataset.txt2arr(' '.join(txt).upper(), 1), txt
        return MyDataset.word2key(txt), txt

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def word2key(txt):
        arr = []
        tensor = [0]
        for word in txt:
            tensor.append(MyDataset.indexs[word])
        tensor.append(1)
        return np.array(tensor)

    @staticmethod
    def key2word(arr):
        result = []
        n = arr.size(0)
        T = arr.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = arr[i, t]
                if (c >= 2):
                    c = c.cpu()
                    c = int(c)
                    text.append(MyDataset.vocabulary[c])
            text = ' '.join(text)
            result.append(text)   
        return result

    @staticmethod
    def txt2arr(txt, SOS=False):
        # SOS: 1, EOS: 2, P: 0, OTH: 3+x
        arr = []
        if(SOS):            
            tensor = [1]
        else:
            tensor = []
        for c in list(txt):
            tensor.append(3 + MyDataset.letters.index(c))
        tensor.append(2)
        return np.array(tensor)
        
    @staticmethod
    def arr2txt(arr):       
        # (B, T)
        result = []
        n = arr.size(0)
        T = arr.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = arr[i,t]
                if(c >= 3):
                    text.append(MyDataset.letters[c - 3])
            text = ''.join(text)
            result.append(text)
        return result
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt)
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()        


'''
def data_from_opt(vid_path, phase):
    dataset = MyDataset(vid_path, 
        opt.anno_path,
        opt.vid_pad,
        opt.txt_pad,
        phase=phase)
    print('vid_path:{},num_data:{}'.format(vid_path, len(dataset.data)))
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)

(train_dataset, train_loader) = data_from_opt(opt.trn_vid_path, 'train')

for (i, batch) in enumerate(train_loader):
    (encoder_tensor, decoder_tensor) = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
    if i == 2:
        print(encoder_tensor, decoder_tensor)
'''
