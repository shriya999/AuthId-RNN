#for article level case

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import file2dict as fdt
import read_minibatch as rmb
import read_minibatch_avemask as rmba
import os
import pickle
import numpy as np
import json
from data_util import load_embeddings

batch_size = 16
data_path = '/content/C50/C50train'

with open('/content/auth_id/tokenToIndex', 'r') as f:
    try:
        wordToIndex = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        wordToIndex = {}

glove_path = "/content/glove.6B.50d.txt"
glove_vector = load_embeddings(glove_path, 50)  # load the glove vectors

#auth_sent_num = fdt.file2auth_sent_num(data_path)  # read in the training data
#auth_sentbundle_num = fdt.file2auth_sentbundle_num(data_path, 3)[1:1000]

auth_news_num = fdt.file2auth_news_num(data_path)

ind = np.arange(len(auth_news_num))
np.random.shuffle(ind)
index = ind
raw_data = [auth_news_num[i] for i in index ]

batch_list = rmb.process_word2num(raw_data, wordToIndex, glove_vector,24)

batch_list_bundle = rmb.pack_batch_list(batch_list, batch_size)

output = open('/content/auth_id/data_sentence.pkl', 'wb')
pickle.dump(batch_list_bundle, output, -1)
output.close()

print "Success!"

#print batch_list
