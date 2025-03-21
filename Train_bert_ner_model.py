"""
@author: Dr Yen Fred WOGUEM 

"""

import json
import torch
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
from datetime import datetime

start_time = datetime.now()

# Loading annotated data from a JSON file
def load_annotated_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data  

# Transform data into a format suitable for fine-tuning
def preprocess_data(data, tokenizer, label_to_id):
    texts = []
    #labels = []
    
    for entry in data:
        text = entry["text"]
        entities = entry["entities"]

        # Text tokenization
        tokenized_input = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        #tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
        input_ids = tokenized_input["input_ids"][0]
        attention_mask = tokenized_input["attention_mask"][0]

        # Create labels for each token
        label_ids = [0] * len(input_ids)  # 0 for “O” (no entity)
        

        for entity in entities:
            start, end, label = entity["start"], entity["end"], entity["label"]
            label_id = label_to_id[label]
            
            # Assign label to corresponding tokens
            for i in range(len(tokenized_input.tokens())):
                if tokenized_input.token_to_chars(i) is not None:
                    char_start, char_end = tokenized_input.token_to_chars(i)
                    if char_start >= start and char_end <= end:
                        label_ids[i] = label_id

        texts.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(label_ids)})

    return Dataset.from_list(texts)


# Load a pre-trained model
def load_model(model_name, num_labels):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    return model, tokenizer



def freeze_bert_and_old_classes(model) :#, label_list): #, frozen_labels):
    """
    Freezes the weights of the BERT class and the former NER class.

    Args :
        model: The BERT model with a classification layer.
        label_list : Complete list of classes.
        frozen_labels: List of classes whose weights should not be updated.
    """
    # Freeze all layers of BERT
    for param in model.bert.parameters():
        param.requires_grad = False  

    # Freeze old class weights in the classification layer
    #classifier_weights = model.classifier.weight
    #classifier_bias = model.classifier.bias

    #frozen_indices = [label_list.index(label) for label in frozen_labels]

   # with torch.no_grad():
    #    for idx in frozen_indices:
    #        classifier_weights[idx].requires_grad = False  #  Old frozen classes
    #        classifier_bias[idx].requires_grad = False

    #print(f" Fixed weights for : BERT + {frozen_labels}")

# Class definition 
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
              "B-MISC", "I-MISC", "B-MONEY", "I-MONEY", "B-DATE", "I-DATE"]

label_to_id = {label: i for i, label in enumerate(label_list)}

# Classes NOT to be trained (Anciennes classes + MISC)
#frozen_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
               #  "B-MISC", "I-MISC"]

# Load BERT pre-trained model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model, tokenizer = load_model(model_name, num_labels=len(label_list))

# Apply gel to BERT and old classrooms
freeze_bert_and_old_classes(model) #, label_list, frozen_labels)

print(f" Taille du classifier : {model.classifier.out_features}")

# Load annotated data
all_data = load_annotated_data("train_data.json")


# Split data into train and validation (90% train, 10% validation)
train_data, val_data = train_test_split(all_data, test_size=0.1)


# Data pre-processing
train_dataset = preprocess_data(train_data, tokenizer, label_to_id)
val_dataset = preprocess_data(val_data, tokenizer, label_to_id)


# Load metric
metric = evaluate.load("accuracy")

# Metric calculation function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)  
    true_predictions = [[p for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    true_labels = [[l for l in label if l != -100] for label in labels]

    true_predictions = true_predictions[0]
    true_labels = true_labels[0]

    return metric.compute(predictions=true_predictions, references=true_labels)




# Fine-tuning 
def fine_tune_model(model, train_dataset, val_dataset):

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        eval_steps=1,
        save_total_limit=1,
        save_steps=30,
        logging_steps=30,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()


model.config.num_labels = len(label_list)  
model.config.id2label = {i: label for i, label in enumerate(label_list)}
model.config.label2id = {label: i for i, label in enumerate(label_list)}

# Model Fine-tuning 
fine_tune_model(model, train_dataset, val_dataset)

# Save model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Modify config.json to update `_name_or_path`
config_path = f"{"./fine_tuned_model"}/config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Change model name in config.json
config["_name_or_path"] = "dbmdz/bert-large-cased-finetuned-conll03-english-finetuned-money-date-by-Woguem"  
config["_num_labels"] = 13 

# Save modifications in config.json
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)

print("End of Fine-tuning and save model.")



end_time = datetime.now()  
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")
