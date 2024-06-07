import os
import json
from random import choice
from underthesea import sent_tokenize


def extract_data(path, name):
    fo = open(os.path.join("data", name + ".txt"), "w")
    with open(path) as f:
        data = json.load(f)
    for sample in data:
        
        question = sample['question']
        title, sent_index = sample['supporting_facts'][0]
        sentence_contain_answer = ""
        other_sentences = []
        for context in sample['context']:
            print(context)
            if context[0] == title:
                sents = context[1]
                sentence_contain_answer = sents[sent_index]
                if len(other_sentences) > 2: break
            else:
                other_sentences += context[1]
        if len(sample['supporting_facts']) == 1 and sentence_contain_answer != "":
            fo.write(f"{question}\t{sentence_contain_answer}\t1\n")
        fo.write(f"{question}\t{choice(other_sentences)}\t0\n")

    fo.close()



dir = "VIMQA_Dataset"
paths = [os.path.join(dir, "vimqa_dev.json"), os.path.join(dir, "vimqa_test.json"), os.path.join(dir, "vimqa_train.json")]
for path in paths:
    name = path.split('/')[-1].split('.')[0]
    extract_data(path, name)