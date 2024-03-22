# Mackenzie Bookamer and Thalia Koutsougeras
# Project Pickup


from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Load in our fine-tuned model produced by finetune.py
model_path = "./pickup_lines_fine_tuned_model_round4"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") # Same as before
model = GPT2LMHeadModel.from_pretrained(model_path) # Same as before

# Set the model to evaluation mode - you can set it in train mode or eval mode and it apparently just makes the process more efficient
model.eval()


# Generation
prompt = "Do you have a map? Beacause I ...  " # We set this each time
input_ids = tokenizer.encode(prompt, return_tensors="pt") # pt stands for pytorch 
output = model.generate(input_ids, max_length=100,  
                        num_return_sequences=1,
                        temperature=0.7,  # Choices explained in report
                        top_k=50,         
                        top_p=0.9)

# Here we decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True) # output[0] gets first sequence, skip things like [PAD]
print("Generated text:")
print(generated_text)
