"""
@author: Dr Yen Fred WOGUEM 

"""


import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import pipeline
from datetime import datetime

start_time = datetime.now()





# Function for load model 
def load_model(model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    return model, tokenizer

# Function for extract nammed entities
def extract_entities(text, model, tokenizer):
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp_ner(text)

# Pre-trainned BERT model for NER 
#model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model_path = "./fine_tuned_model"
model, tokenizer = load_model(model_path)

# Read Contractual Insurance
with open(r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\Portfiolo\Named_Entity_Recognition_Insurance\Contractual_Insurance.txt", "r") as file:
    text = file.read()



# Entities extraction 
entities = extract_entities(text, model, tokenizer)

# Display results in Results_NER file
with open(r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\Portfiolo\Named_Entity_Recognition_Insurance\Results_NER.txt", "w", encoding="utf-8") as output_file:
    for entity in entities:
        output_file.write(f"Entity: {entity['word']}, Type: {entity['entity']}, Confidence: {entity['score']}\n")
   


print("The extracted entities have been saved in Results_NER.txt")





end_time = datetime.now()  # Fin du chronomètre
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")



