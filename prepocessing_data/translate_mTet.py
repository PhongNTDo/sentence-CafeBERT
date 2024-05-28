from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tqdm


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()

inputs = [
    "vi: VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam.",
    "vi: Theo báo cáo mới nhất của Linkedin về danh sách việc làm triển vọng với mức lương hấp dẫn năm 2020, các chức danh công việc liên quan đến AI như Chuyên gia AI (Artificial Intelligence Specialist), Kỹ sư ML (Machine Learning Engineer) đều xếp thứ hạng cao.",
    "en: Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing our research and tools to fuel progress in the field.",
    "en: We're on a journey to advance and democratize artificial intelligence through open source and open science."
    ]


def translation(inputs):
    outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translated = []
    for re in out:
        translated.append(re[3:].strip())
    return translated


file = "data/squad.txt"
fo = open(file.rsplit('.', 1)[0] + "_vi.txt", "w")

count = 0
with open(file) as f:
    for line in tqdm.tqdm(f.readlines()):
        line = line.strip()
        if line == "":
            continue

        if len(line.split('\t')) != 3:
            continue
        sent1, sent2 = line.split('\t')[:2]
        label = line.split('\t')[-1]

        inps = [f"en: {sent1}", f"en: {sent2}"]

        translated = translation(inps)

        fo.write(f"{translated[0]}\t{translated[1]}\t{label}\n")
        count += 1
        if count > 100000:
            break
fo.close()