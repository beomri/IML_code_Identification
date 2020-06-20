from ngrams import get_grams
from nltk import ngrams
from random import randint,shuffle
# from numpy import argmax
from time import time
import numpy as np

# n=[6,10,20]
# l=[25,50,100]

def predict(lines,n, grams):
    grams_bytes = ngrams(''.join(lines.split()), n)
    grams_bytes = set(grams_bytes)
    intersections = []
    for g in grams:
        intersections.append(len(grams_bytes.intersection(g)))
    ind =  np.argmax(intersections)
    # if intersections.count(intersections[ind]) > 1:
    #     print(intersections)

    return ind






ns=[(14, 6)]
# ls=[10000]
labels = range(7)

codes = []

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

shuffle(codes)
total_sets = len(codes)
erros=[]
l=50000
for n1, n2 in ns:
    error_count = 0
    grams1 = []
    grams2 = []
    start = time()
    for g in get_grams(n1):
        t = []
        for bla in g.most_common(l):
            t.append(bla[0])
        # grams.append(g.most_common(l).keys())
        grams1.append(t)
    for g in get_grams(n2):
        t = []
        for bla in g.most_common(l):
            t.append(bla[0])
        # grams.append(g.most_common(l).keys())
        grams2.append(t)
    np.savez_compressed('bla',a=grams1,b=grams2)
    # np.save('blb',grams2)
    print('learned! testing: ', str(len(codes)))
    elapsed = time() - start
    print('time: '+ str(elapsed/60))
    start = time()
    # for i, (c, label) in enumerate(codes):
    for i in range(1000):
        c, label = codes[i]
        if i%50 == 0:
            print(str(i) + '/' + str(len(codes)))
        if i%5000 == 0:
            elapsed = time() - start
            print('time: '+ str(elapsed/60))
            start = time()
        pred = predict(c,n,grams)
        if pred != label:

            erros.append((c,label,pred))
            error_count += 1
            print('ohoh: ',str(error_count/(i+1)))
            # print(erros[-1])
    print('n=',n,'error=',error_count/1000)

    import IPython
    IPython.embed()


# for n in ns:
#     for l in ls:
#         error_count = 0
#         grams = get_grams(n,l)
#         for c, label in codes:
#             pred = predict(c,n,grams)
#             if pred != label:
#                 erros.append((c,label,pred))
#                 error_count += 1
#     print('n=',n,'l=',l,'error=',error_count/total_sets)









