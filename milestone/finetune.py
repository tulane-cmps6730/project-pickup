# Mackenzie Bookamer and Thalia Koutsougeras
# Project Pickup


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# We found this website really helpful!  https://huggingface.co/docs/transformers/en/training 

# We want to load in the pre-trained GPT-2 model and tokenizer
# Using the Hugging Face Transformers library
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") # Load pre-trained weights and configuration from hugging face hub
tokenizer.pad_token = tokenizer.eos_token  # Our terminal raised an error when we didn't have this line and suggested to use the end of sequence token
# The pad token makes sure the sequences are the same length for consistency
model = GPT2LMHeadModel.from_pretrained("gpt2-medium") # Instantiates a GPT model from hugging face

pickup_lines_df = pd.read_csv("pickupdataall.csv") # Load in our data for CS pickup lines and jokes

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(pickup_lines_df, test_size=0.2, random_state=42) # 

# The first tokenizer was for the pre-trained model, but now we need to tokenize our dataset
def tokenize_function(examples):
    return tokenizer(examples["pickup_line"], padding="max_length", max_length=50) # Go through and tokenize each line
# We put a maxlength padding to match the same padding we did in the gpt2 tokenizer - normally max_length it's 20, but we thought we needed more

# We found the Dataset library from Hugging Face, which is specifically design for NLP
# We also saw that people in discussion posts were converting to it so we decided to play with this library
train_dataset = Dataset.from_pandas(train_data) # Convert pandas to Dataset
test_dataset = Dataset.from_pandas(test_data)

# Applying tokenizer to each example
# Saw that "batched" improves efficiency, especially for large datasets
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True) # was pandas apply
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./pickup_lines_fine_tuned_round4", # We use this when we run our other file to generate text
    overwrite_output_dir=True,
    num_train_epochs=3, # Will eventually probably run more
    # maybe add compute_metrics?
)

# Here's where we fine-tune, following the structure on Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
)

# This function is way faster than the manual epoch loop were doing before
model.train()

# Evaluate the fine tuned model
results = trainer.evaluate()

# Save the fine-tuned model to be used later
model.save_pretrained("./pickup_lines_fine_tuned_model_round4")







