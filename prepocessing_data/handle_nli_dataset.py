import json
import os.path
import tqdm


def extract_data(file, out_file, org_lang="vi"):
    print("Handle file: ", file)
    fo = open(out_file, "w")
    count = 0
    with open(file) as f:
        for line in tqdm.tqdm(f.readlines()):
            count += 1
            line = line.strip()
            elements = line.split("\t")

            if file.endswith("txt"):
                # format of SNLI dataset
                nli_label = elements[0]
                sent1 = elements[5]
                sent2 = elements[6]
            elif file.endswith("jsonl"):
                # format of ViNLI dataset
                elements = json.loads(line)
                nli_label = elements['label']
                sent1 = elements['premise']
                sent2 = elements['hypothesis']

            if (nli_label == "entailment") or (nli_label == "e"):
                label = 1
            elif (nli_label == "contradiction") or (nli_label == "c"):
                label = 0
            else:
                continue

            fo.write(f"{sent1}\t{sent2}\t{label}\n")
            if count >= 100000:
                break
    fo.close()


dir = "/home/phongdnt/Documents/Cafe/data/ViNLI/vinli"
list_file_dataset = os.listdir(dir)
list_file_dataset = [os.path.join(dir, file) for file in list_file_dataset]

out_dir = "./data"
for file in list_file_dataset:
    file_name = file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    extract_data(file, os.path.join(out_dir, "vnli_" + file_name + ".txt"), org_lang="vi")
