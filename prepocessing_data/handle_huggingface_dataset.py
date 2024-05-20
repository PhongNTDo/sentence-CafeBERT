import os
import random
from datasets import load_dataset
from nltk.tokenize import sent_tokenize


def get_sentence(context, answer_start):
    list_sentence = sent_tokenize(context)[::-1]
    for index, sentence in enumerate(list_sentence):
        start_sent = context.find(sentence)
        if start_sent <= answer_start:
            index_sentence_contain_answer = index
            index_other = -1
            retry = 0
            while (index_other < 0) or (index_other == index_sentence_contain_answer):
                index_other = random.randint(0, len(list_sentence) - 1)
                retry += 1
                if retry > 10:
                    return "", ""
            break
    return list_sentence[index_sentence_contain_answer], list_sentence[index_other]


def extract_data_squad(out_dir):
    squad_dataset = load_dataset("rajpurkar/squad")
    fo = open(os.path.join(out_dir, "SQuAD.txt"), "w")
    for sample in squad_dataset['train']:
        context = sample['context']
        question = sample['question']
        answer_start = sample['answers']['answer_start'][0]

        sentence_contain_answer, other_sentence = get_sentence(context, answer_start)

        if (sentence_contain_answer != "") and (other_sentence != ""):
            fo.write(f"{question}\t{sentence_contain_answer}\t1\n")
            fo.write(f"{question}\t{other_sentence}\t0\n")

    fo.close()


def extract_data_quora(out_dir):
    quora_dataset = load_dataset("quora")
    fo = open(os.path.join(out_dir, "Quora.txt"), "w")
    cnt = 0
    for sample in quora_dataset['train']:
        question1, question2 = sample['questions']['text']
        is_duplicate = sample['is_duplicate']

        if is_duplicate:
            label = 1
        else:
            label = 0

        fo.write(f"{question1}\t{question2}\t{label}\n")

        cnt += 1
        if cnt > 100000:
            break
    
    fo.close()


out_dir = "./data"
# extract_data_squad(out_dir)
extract_data_quora(out_dir)