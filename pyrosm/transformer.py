from datasets import Dataset, Features, Sequence, ClassLabel, Value

# Define the unique labels from your OSM extraction
label_list = ['O', 'B-HOUSE', 'I-HOUSE', 'B-STREET', 'I-STREET', 'B-CITY', 'I-CITY', 'B-STATE', 'I-STATE']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Convert your 'formatted_data' list into a Hugging Face Dataset
features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=label_list))
})

# Convert string tags to their integer IDs
for item in formatted_data:
    item["ner_tags"] = [label2id[tag] for tag in item["ner_tags"]]

ds = Dataset.from_list(formatted_data, features=features)

from transformers import AutoTokenizer

model_checkpoint = "bert-base-multilingual-cased" # Good for Nigerian local context
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Special ID for padding/ignored tokens
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)

from transformers import pipeline

# Load your fine-tuned model and its tokenizer
model_path = "./nigeria-address-ner"
address_extractor = pipeline(
    "ner", 
    model=model_path, 
    tokenizer=model_path,
    aggregation_strategy="simple" # Combines sub-tokens back into full words
)


# A typical "messy" user input for a delivery
user_input = "Please deliver to 15 Admiralty Way near the pizza place in Lekki Lagos"

# Perform extraction
extracted_entities = address_extractor(user_input)

# Display results
for entity in extracted_entities:
    print(f"Entity: {entity['word']} | Label: {entity['entity_group']} | Confidence: {entity['score']:.2f}")


structured_address = {ent['entity_group']: ent['word'] for ent in extracted_entities}
# Result: {'HOUSE': '15', 'STREET': 'Admiralty Way', 'CITY': 'Lekki', 'STATE': 'Lagos'}
