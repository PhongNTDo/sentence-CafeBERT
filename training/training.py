"""
This scripts demonstrates how to train a sentence embedding model for question pair classification
with cosine-similarity and a simple threshold.

As dataset, we use Quora Duplicates Questions, where we have labeled pairs of questions being either duplicates (label 1) or non-duplicate (label 0).

As loss function, we use OnlineConstrativeLoss. It reduces the distance between positive pairs, i.e., it pulls the embeddings of positive pairs closer together. For negative pairs, it pushes them further apart.

An issue with constrative loss is, that it might push sentences away that are already well positioned in vector space.
"""

import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, models
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout


# As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
# model = SentenceTransformer("stsb-distilbert-base")
word_embedding_model = models.Transformer('uitnlp/CafeBERT', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
model = model.cuda()
num_epochs = 2
train_batch_size = 8

# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# Negative pairs should have a distance of at least 0.5
margin = 0.5

dataset_path = "/data/training_text_mini.txt"
model_save_path = "output/training_OnlineConstrativeLoss-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

# Check if the dataset exists. If not, download and extract
if not os.path.exists(dataset_path):
    logger.info("Dataset not found.")

######### Read train data  ##########
# Read train data
train_samples = []
with open(dataset_path, encoding="utf8") as fIn:
    for line in fIn.readlines():
        text1, text2, label = line.strip().split('\t')
        sample = InputExample(texts=[text1, text2], label=int(label))
        train_samples.append(sample)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
with open("/data/valid_text_mini.txt", encoding="utf8") as fIn:
    for line in fIn.readlines():
        text1, text2, label = line.strip().split("\t")
        dev_sentences1.append(text1)
        dev_sentences2.append(text2)
        dev_labels.append(int(label))

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=seq_evaluator,
    epochs=num_epochs,
    warmup_steps=1000,
    output_path=model_save_path,
)

model.push_to_hub(repo_id="ThuanPhong/sentence_CafeBERT",
                  token="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",
                  commit_message="demo")
