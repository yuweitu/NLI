import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import os
import time
from collections import OrderedDict
import spacy
import argparse
import json
from collections import Counter
nlp = spacy.load('en')


def process(text):
    
    processed = []
    for i in nlp(text):
        if i.pos_ in ['SPACE', 'PUNCT']:
            continue
        elif i.pos_ == 'PART':
            processed.append('_s')
        elif i.pos_ in ['NUM', 'SYM']:
            processed.append(i.pos_)
        else:
            processed.append(i.text)

    return processed

def main(args):

    vocab0 = OrderedDict()
    start = time.time()

    count = 0
    with open(args.saveto+'%s.txt'%args.save_label, 'w') as f:
        print("loading all files...")
            
        file = open(args.source+args.parse_file)
        count += 1
        for line in file:
            d = json.loads(line)
            label = Counter(d['annotator_labels']).most_common(1)[0][0]
            f.write(label+"\t")  

            s1 = process(d['sentence1'].lower())
            s2 = process(d['sentence2'].lower())

            for w in s1:
                f.write(w+" ")
                if w in vocab0:
                    vocab0[w] += 1
                else:
                    vocab0[w] = 1
            f.write("\t")

            for w in s2:
                f.write(w+" ")
                if w in vocab0:
                    vocab0[w] += 1
                else:
                    vocab0[w] = 1
            f.write("\n")

        if count % 1000 == 0:
            print("processed %s files" % count)
            print("%s seconds elapsed" % (time.time() - start))
    f.close()
    print(time.time() - start)

    tokens = list(vocab0.keys())
    
    freqs = list(vocab0.values())

    sidx = np.argsort(freqs)[::-1]

    # zero is reserve for padding
    vocab = OrderedDict([(tokens[s],i+1) for i, s in enumerate(sidx)])
    
    np.save(args.saveto+"vocab"+args.save_label+".npy", vocab)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-saveto', type=str, default="../intermediate/")
    parser.add_argument('-source', type=str, default="../snli_1.0/")
    parser.add_argument('-save_label', type=str, default='snli_tst')
    parser.add_argument('-parse_file', type=str, default='snli_1.0_test.jsonl')
    args = parser.parse_args()
    #parser.parse_file = 'snli_1.0_dev.jsonl'
    #parser.parse_file = 'snli_1.0_test.jsonl'
    main(args)






