import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random
import re
from torch.optim import AdamW
import time
from seqeval.metrics import classification_report

label_list = ["O", "B-AP", "I-AP", "B-PER", "I-PER", "B-SV", "I-SV", "B-MISC", "I-MISC"]
label_map = {label: idx for idx, label in enumerate(label_list)}
num_labels = len(label_list)

question_templates_ap = [
    "{person} has reported an issue with AP{ap_number} in production.",
    "AP{ap_number} was updated by {person} yesterday.",
    "{person} is requesting help with AP{ap_number} which is not responding.",
    "According to {person}, AP{ap_number} requires urgent maintenance.",
    "AP{ap_number} has been validated by {person} for deployment.",
    "{person} noticed performance issues in AP{ap_number}.",
    "Security audit by {person} found vulnerabilities in AP{ap_number}.",
    "AP{ap_number} deployment was reviewed by {person}.",
    "Database connection for AP{ap_number} was fixed by {person}.",
    "Critical alert: {person} detected failures in AP{ap_number}."
]

question_templates_sv = [
    "{person} initiated server SV{sv_number} for {misc_item}.",
    "The server SV{sv_number} managed by {person} is experiencing {misc_item} issues.",
    "{person} updated configurations on SV{sv_number} to address {misc_item}.",
    "AP{ap_number} and SV{sv_number} require attention due to {misc_item}.",
    "{person} is monitoring SV{sv_number} for {misc_item} problems."
]

def generate_synthetic_data(n_samples=5000):
    data = []
    
    first_names = ["Jean", "Marie", "Pierre", "Sophie", "Thomas", "Julie", "David", "Emma", "Lucas", "Sarah",
                   "Fabrice", "Quentin"]
    last_names = ["Martin", "Bernard", "Dubois", "Robert", "Richard", "Petit", "Durand", "Leroy", "Moreau", "Simon",
                 "le Deit", "FEVER"]
    
    misc_items = ["urgent", "performance", "security", "database", "optimization"]
    
    for _ in range(n_samples):
        entity_type = random.choice(['AP', 'SV'])
        
        if entity_type == 'AP':
            ap_number = str(random.randint(10000, 99999))
            ap_code = f"AP{ap_number}"
            
            person_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            main_content = random.choice(question_templates_ap).format(ap_number=ap_number, person=person_name)
            
            words = main_content.split()
            labels = ["O"] * len(words)
            
            person_words = person_name.split()
            for i, word in enumerate(words):
                if ap_code in word:
                    labels[i] = "B-AP"
                elif word in person_words:
                    if word == person_words[0]:
                        labels[i] = "B-PER"
                    else:
                        labels[i] = "I-PER"
                        
        elif entity_type == 'SV':
            sv_number = str(random.randint(1000, 9999))
            sv_code = f"SV{sv_number}"
            
            misc_item = random.choice(misc_items)
            
            person_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            main_content = random.choice(question_templates_sv).format(
                sv_number=sv_number, 
                person=person_name, 
                misc_item=misc_item, 
                ap_number='00000'
            )
            
            words = main_content.split()
            labels = ["O"] * len(words)
            
            person_words = person_name.split()
            for i, word in enumerate(words):
                if sv_code in word:
                    labels[i] = "B-SV"
                elif word in person_words:
                    if word == person_words[0]:
                        labels[i] = "B-PER"
                    else:
                        labels[i] = "I-PER"
                elif word == misc_item:
                    labels[i] = "B-MISC"
        
        data.append({
            "text": main_content,
            "labels": labels,
            "ap_code": ap_code if entity_type == 'AP' else None,
            "sv_code": sv_code if entity_type == 'SV' else None,
            "person": person_name,
            "misc_item": misc_item if entity_type == 'SV' else None
        })
    
    return data

class APCodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {
            "O": 0, 
            "B-AP": 1, 
            "I-AP": 2,
            "B-PER": 3,
            "I-PER": 4,
            "B-SV": 5,
            "I-SV": 6,
            "B-MISC": 7,
            "I-MISC": 8
        }
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        words = text.split()
        
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label_ids = [-100] * self.max_len
        
        word_ids = encoding.word_ids()
        
        print("\nDebug tokenization:")
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        print("Words:", words)
        print("Labels:", labels)
        print("Tokens:", tokens)
        print("Word IDs:", word_ids)
        
        previous_word_idx = None
        previous_token_label = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
                
            token = tokens[i]
            current_word = words[word_idx]
            current_label = labels[word_idx]
            
            is_subword = (word_idx == previous_word_idx)
            
            if is_subword:
                if current_label.startswith(('B-', 'I-')):
                    label_ids[i] = self.label_map[f"I-{current_label.split('-')[1]}"]
                else:
                    label_ids[i] = self.label_map["O"]
            else:
                label_ids[i] = self.label_map[current_label]
            
            print(f"Token: {token}, Word: {current_word}, Label: {label_ids[i]}, Is Subword: {is_subword}")
            
            previous_word_idx = word_idx
            previous_token_label = label_ids[i]
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label_ids)
        }

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_model():
    print("Starting model training...")
    print("Generating synthetic dataset...")
    data = generate_synthetic_data(5000)
    
    print("Splitting data into train/validation sets...")
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print("Initializing model and tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels
    )
    
    device = get_device()
    print(f"Using device: {device}")
    
    print("Creating datasets...")
    train_dataset = APCodeDataset(
        [d['text'] for d in train_data],
        [d['labels'] for d in train_data],
        tokenizer
    )
    val_dataset = APCodeDataset(
        [d['text'] for d in val_data],
        [d['labels'] for d in val_data],
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_epochs = 2
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{batch_count}, Current loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
        
        model.eval()
        val_loss = 0
        print("Running validation...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
    print("\nTraining completed! Saving model...")
    model.save_pretrained('ap_code_model')
    tokenizer.save_pretrained('ap_code_model')
    print("Model saved successfully!")

def extract_entities(text, model, tokenizer):
    device = get_device()
    model.to(device)
    model.eval()
    
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print("\n=== Debug Token Analysis ===")
        print(f"Input text: {text}")
        print("\nToken-level analysis:")
        print(f"{'Token':<15} {'Prediction':<12} {'Entity Type':<12}")
        print("-" * 40)
        
        current_ap = []
        current_person = []
        current_sv = []
        current_misc = []
        entities = {"AP_CODE": None, "PERSON": None, "SV_CODE": None, "MISC": None}
        
        previous_entity = None

        for token, pred in zip(tokens, predictions[0]):
            pred = pred.item()
            
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            if token.startswith('Ġ'):
                clean_token = token[1:]
            else:
                clean_token = token
                
            entity_type = "O"
            if pred == 1:
                entity_type = "B-AP"
            elif pred == 2:
                entity_type = "I-AP"
            elif pred == 3:
                entity_type = "B-PER"
            elif pred == 4:
                entity_type = "I-PER"
            elif pred == 5:
                entity_type = "B-SV"
            elif pred == 6:
                entity_type = "I-SV"
            elif pred == 7:
                entity_type = "B-MISC"
            elif pred == 8:
                entity_type = "I-MISC"
            
            print(f"{token:<15} {pred:<12} {entity_type:<12}")
            
            if entity_type == "B-PER":
                if previous_entity in ["B-PER", "I-PER"]:
                    entity_type = "I-PER"
            elif entity_type == "B-SV":
                if previous_entity in ["B-SV", "I-SV"]:
                    entity_type = "I-SV"
            elif entity_type == "B-MISC":
                if previous_entity in ["B-MISC", "I-MISC"]:
                    entity_type = "I-MISC"
            
            if pred == 1:
                current_ap = [clean_token]
            elif pred == 2:
                if current_ap:
                    current_ap.append(clean_token)
            elif entity_type == "B-PER":
                if token.startswith('Ġ'):
                    clean_token = ' ' + clean_token
                current_person = [clean_token]
            elif entity_type == "I-PER":
                if token.startswith('Ġ'):
                    clean_token = ' ' + clean_token
                current_person.append(clean_token)
            elif entity_type == "B-SV":
                if token.startswith('Ġ'):
                    clean_token = ' ' + clean_token
                current_sv = [clean_token]
            elif entity_type == "I-SV":
                if token.startswith('Ġ'):
                    clean_token = ' ' + clean_token
                if 'current_sv' in locals():
                    current_sv.append(clean_token)
            elif entity_type == "B-MISC":
                if token.startswith('Ġ'):
                    clean_token = ' ' + clean_token
                current_misc = [clean_token]
            elif entity_type == "I-MISC":
                if token.startswith('Ġ'):
                    clean_token = ' ' + clean_token
                if 'current_misc' in locals():
                    current_misc.append(clean_token)
            elif pred == 0:
                if current_ap:
                    ap_code = ''.join(current_ap).replace(' ', '')
                    if re.match(r'AP\d{5}', ap_code):
                        entities["AP_CODE"] = ap_code
                    current_ap = []
                if current_person:
                    person = ''.join(current_person).strip()
                    if entities["PERSON"] is None:
                        entities["PERSON"] = person
                    current_person = []
                if 'current_sv' in locals():
                    sv_code = ''.join(current_sv).replace(' ', '')
                    if re.match(r'SV\d{4}', sv_code):
                        entities["SV_CODE"] = sv_code
                    del current_sv
                if 'current_misc' in locals():
                    misc = ''.join(current_misc).strip()
                    if entities.get("MISC") is None:
                        entities["MISC"] = misc
                    del current_misc

            previous_entity = entity_type

        if current_ap:
            ap_code = ''.join(current_ap).replace(' ', '')
            if re.match(r'AP\d{5}', ap_code):
                entities["AP_CODE"] = ap_code
        if current_person:
            person = ''.join(current_person).strip()
            if entities["PERSON"] is None:
                entities["PERSON"] = person
        if 'current_sv' in locals():
            sv_code = ''.join(current_sv).replace(' ', '')
            if re.match(r'SV\d{4}', sv_code):
                entities["SV_CODE"] = sv_code
        if 'current_misc' in locals():
            misc = ''.join(current_misc).strip()
            if entities.get("MISC") is None:
                entities["MISC"] = misc

        print("\nExtracted Entities:", entities)
        print("=" * 40)
        return entities

if __name__ == "__main__":
    print("Starting fresh training...")
    train_model()
    
    print("\nWaiting a moment for files to be saved...")
    
    try:
        print("\nLoading saved model...")
        model = RobertaForTokenClassification.from_pretrained('./ap_code_model')
        tokenizer = RobertaTokenizerFast.from_pretrained('./ap_code_model', add_prefix_space=True)
        
        test_texts = [
            "Fabrice le Deit has reported an issue with SV1234 in production.",
            "Marie Louise is requesting help with SV5678 which is not responding.",
            "AP98765 was updated by Thomas Bernard yesterday.",
            "Sophie Dubois is working on optimizing AP55555.",
            "Quentin FEVER bosse sur AP24563",
            "Jean Martin initiated server SV4321 for urgent tasks.",
            "Marie Louise updated configurations on SV8765 to address performance issues.",
            "The server SV1234 managed by Pierre Dubois is experiencing security issues.",
            "Critical alert: SV5678 detected failures due to database problems."
        ]
        
        print("\nTesting with multiple examples:")
        for text in test_texts:
            print(f"\nInput: {text}")
            result = extract_entities(text, model, tokenizer)
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
