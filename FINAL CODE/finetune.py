import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

#Define tokenizer and model
#chose 355M gpt-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

#Adjust pad token for GPT2
tokenizer.pad_token = tokenizer.eos_token

#Read data from csv's we created
#concatenate both together
pickup_lines_dfcs = pd.read_csv("pickupdataall (1).csv")
pickup_lines_dfmath = pd.read_csv("pickupdatamathcsv.csv")
pickup_lines_df = pd.concat([pickup_lines_dfcs, pickup_lines_dfmath], ignore_index=True)

#Split data
train_data, test_data = train_test_split(pickup_lines_df, test_size=0.2, random_state=42)

#Tokenize data
train_tokenized = tokenizer(train_data["pickup_line"].tolist(), truncation=True, padding=True)
test_tokenized = tokenizer(test_data["pickup_line"].tolist(), truncation=True, padding=True)

#Convert to Dataset
train_dataset = Dataset.from_dict({"input_ids": train_tokenized.input_ids, "attention_mask": train_tokenized.attention_mask})
test_dataset = Dataset.from_dict({"input_ids": test_tokenized.input_ids, "attention_mask": test_tokenized.attention_mask})

#Define training arguments
training_args = TrainingArguments(
    output_dir="./project-pickup-tuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    save_strategy="epoch",  #Save model every epoch in case something happens
)

#Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

#Train the model
trainer.train()

#Save the model
model.save_pretrained("./project-pickup-tuned")
