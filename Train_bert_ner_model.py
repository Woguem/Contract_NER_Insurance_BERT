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

# Charger les données annotées depuis un fichier JSON
def load_annotated_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data  # Retourne la liste des exemples annotés

# Transformer les données en format adapté au fine-tuning
def preprocess_data(data, tokenizer, label_to_id):
    texts = []
    labels = []
    
    for entry in data:
        text = entry["text"]
        entities = entry["entities"]

        # Tokenization du texte
        tokenized_input = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        #tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
        input_ids = tokenized_input["input_ids"][0]
        attention_mask = tokenized_input["attention_mask"][0]

        # Création des labels pour chaque token
        label_ids = [0] * len(input_ids)  # 0 pour "O" (aucune entité)
        

        for entity in entities:
            start, end, label = entity["start"], entity["end"], entity["label"]
            label_id = label_to_id[label]
            
            # Assigner le label aux tokens correspondants
            for i in range(len(tokenized_input.tokens())):
                if tokenized_input.token_to_chars(i) is not None:
                    char_start, char_end = tokenized_input.token_to_chars(i)
                    if char_start >= start and char_end <= end:
                        label_ids[i] = label_id

        texts.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(label_ids)})

    return Dataset.from_list(texts)


# Charger un modèle pré-entraîné
def load_model(model_name, num_labels):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    # Charger la configuration
    #config = model.config

    # Mettre à jour le nombre de labels dans la configuration
    #config.num_labels = len(label_list)
    # config.label2id = label_to_id
    #config.id2label = {i: label for i, label in enumerate(label_list)}

    # Assigner la nouvelle configuration au modèle
    # model.config = config

    return model, tokenizer

# Définition des labels
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MONEY", "I-MONEY", "B-DATE", "I-DATE"]
label_to_id = {label: i for i, label in enumerate(label_list)}

# Charger les données annotées
all_data = load_annotated_data("train_data.json")


# Séparer les données en train et validation (90% train, 10% validation)
train_data, val_data = train_test_split(all_data, test_size=0.1)

# Charger le modèle pré-entraîné BERT et le tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  
model, tokenizer = load_model(model_name, num_labels=len(label_list))

# Prétraitement des données
train_dataset = preprocess_data(train_data, tokenizer, label_to_id)
val_dataset = preprocess_data(val_data, tokenizer, label_to_id)


# Charger la métrique NER
metric = evaluate.load("accuracy")

# Fonction pour calculer les métriques
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)  # Si vos prédictions sont des probabilités, prenez l'index max
    true_predictions = [[p for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    true_labels = [[l for l in label if l != -100] for label in labels]

    true_predictions = true_predictions[0]
    true_labels = true_labels[0]

    # Transformer en format attendu
    return metric.compute(predictions=true_predictions, references=true_labels)




# Fine-tuning du modèle
def fine_tune_model(model, train_dataset, val_dataset):

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
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

# Fine-tuning du modèle
fine_tune_model(model, train_dataset, val_dataset)

# Sauvegarde du modèle fine-tuné
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Modifier le fichier config.json pour mettre à jour `_name_or_path`
config_path = f"{"./fine_tuned_model"}/config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Changer le nom du modèle dans config.json
config["_name_or_path"] = "dbmdz/bert-large-cased-finetuned-conll03-english-finetuned-money-date-by-Woguem"  
config["_num_labels"] = 11 

# Sauvegarder les modifications dans config.json
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)

print("Fine-tuning terminé et modèle sauvegardé.")



end_time = datetime.now()  # Fin du chronomètre
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")
