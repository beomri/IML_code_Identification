from classifier import  Classifier
from ngrams import get_grams
from nltk import ngrams
from random import randint,shuffle
# from numpy import argmax
from time import time
import numpy as np


codes = []
cls = Classifier(toTrain=True)
# cls.load()
labels = range(7)
for label in labels:
    lines = []
    with open(str(label)+'_val.txt') as f:
        for l in f:
            lines.append(l)
    num_lines = len(lines)
    lengths = [randint(1,5)]
    while sum(lengths) < num_lines:
        lengths.append(randint(1,5))
    lengths = lengths[:-1]
    lengths.append(num_lines - sum(lengths))
    last_line = 0

    for length in lengths:
        # import IPython
        # IPython.embed()
        codes.append((''.join(lines[last_line: last_line+length]),label))
        last_line += length
# np.savez_compressed('val_codes',a=codes)

samples = 5000
shuffle(codes)
total_sets = len(codes)

lines_set = [c[0] for c in codes[:samples]]
true_labels = [c[1] for c in codes[:samples]]
# lines_set = []
# true_labels = []
# for i in range(7):
#     count = 0
#     for c in codes:
#         if c[1] == i:
#             lines_set.append(c[0])
#             true_labels.append(c[1])
#             count += 1
#             if count >= samples//7:
#                 break




# import IPython
# IPython.embed()
start = time()
error_count  =0
errors = []
for ind, (lines, label) in enumerate(zip(lines_set,true_labels)):
    pred = cls.predict(lines)
    if(pred != label):

        errors.append((lines,label,pred))
        print('error index: ', error_count)
        print('expected:', label,' got:',pred)
        error_count += 1
        print()
# preds = cls.classify(lines_set)
elapsed = time() - start
print('time: '+ str(elapsed/60))
# error_count  =0
# errors = []
# for i in range(samples):
#     if preds[i] != true_labels[i]:
#         # print('error found')
#         error_count += 1
#         errors.append((lines_set[i],true_labels[i],preds[i]))

# erros=[]
# error_count  =0
# start = time()
# for i in range(1000):
#     c, label = codes[i]
#     if i%50 == 0:
#         print(str(i) + '/' + str(len(codes)))
#     if i%5000 == 0:
#         elapsed = time() - start
#         print('time: '+ str(elapsed/60))
#         start = time()
#     pred = cls.predict(c)
#     if pred != label:
#
#         erros.append((c,label,pred))
#         error_count += 1
#         print('ohoh: ',str(error_count/(i+1)))
#         # print(erros[-1])
print('error=',error_count/samples)
print()
import IPython
IPython.embed()