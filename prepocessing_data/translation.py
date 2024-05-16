import os.path
import tqdm
from googletrans import Translator


class Translation:
    def __init__(self):
        self.translator = Translator(service_urls=['translate.google.com',
                                                   'translate.google.co.vi'
                                                   ])

    def translate(self, org_text, src='en', dest='vi'):
        re = self.translator.translate(org_text, dest=dest, src=src)
        return re.text

translator = Translation()
file = "data/snli_1.0_dev.txt"
fo = open(file.rsplit('.')[0] + "_vi.txt", "w")

with open(file) as f:
    for line in tqdm.tqdm(f.readlines()):
        line = line.strip()
        if line == "":
            continue

        sent1, sent2 = line.split('\t')[:2]
        label = line.split('\t')[-1]

        translated_sent1 = translator.translate(sent1)
        translated_sent2 = translator.translate(sent2)

        fo.write(f"{translated_sent1}\t{translated_sent2}\t{label}\n")
fo.close()
