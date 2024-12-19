import os
import xml.etree.ElementTree as ET
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

# Step 1: Parse XML Files
def parse_xml_to_qa_pairs(xml_folder):
    qa_data = []
    for file_name in os.listdir(xml_folder):
        if file_name.endswith(".xml"):
            file_path = os.path.join(xml_folder, file_name)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for qa_pair in root.findall(".//QAPair"):
                question = qa_pair.find("Question").text
                answer = qa_pair.find("Answer").text
                if question and answer:
                    qa_data.append({"question": question.strip(), "answer": answer.strip()})
    return qa_data

xml_folder = "data/5_NIDDK_QA"
qa_pairs = parse_xml_to_qa_pairs(xml_folder)

# Step 2: Prepare the Dataset for Fine-Tuning
def prepare_fine_tuning_data(qa_pairs):
    fine_tuning_data = []
    for qa in qa_pairs:
        prompt = f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        fine_tuning_data.append({"text": prompt})
    return fine_tuning_data

fine_tuning_data = prepare_fine_tuning_data(qa_pairs)
dataset = Dataset.from_dict({"text": [d["text"] for d in fine_tuning_data]})

# Step 3: Tokenize the Data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Load Pretrained Model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    eval_strategy="no",  # Update to eval_strategy
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    warmup_steps=500,
    fp16=True,  # Enable FP16 if using a compatible GPU
)

# Step 6: Prepare Data Collator (for padding)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 7: Fine-Tune the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
    data_collator=data_collator,  # Include data collator
)

trainer.train()

# Step 8: Save the Fine-Tuned Model
trainer.save_model("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("Fine-tuning complete! The model is saved in './fine_tuned_gpt2'")
