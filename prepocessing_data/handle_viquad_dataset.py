import os
import json
import random
from underthesea import sent_tokenize


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
            


def extract_data(file, out_file, org_lang="vi"):
    print("Handle file: ", file)
    fo = open(out_file, "w")
    count = 0
    with open(file) as f:
        data = json.load(f)
        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qas in paragraph['qas']:
                    question = qas['question']
                    answer_start = qas['answers'][0]['answer_start']

                    sentence_contain_answer, other_sentence = get_sentence(context, answer_start)

                    if (sentence_contain_answer != "") and (other_sentence != ""):
                        fo.write(f"{question}\t{sentence_contain_answer}\t1\n")
                        fo.write(f"{question}\t{other_sentence}\t0\n")
    fo.close()



dir = "./ViNewsQA"
list_file_dataset = ["train_ViNewsQA.json", "dev_ViNewsQA.json", "test_ViNewsQA.json"]
out_dir = "./data"

for file in list_file_dataset:
    file_name = file.rsplit('.', 1)[0]
    file_path = os.path.join(dir, file)
    extract_data(file_path, os.path.join(out_dir, file_name + ".txt"), org_lang="vi")