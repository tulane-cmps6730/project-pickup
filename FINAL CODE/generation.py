from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("./project-pickup-tuned")  # Load your fine-tuned model here

# Define a prompt
prompt = "Are you GitHub? Because"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text using top-K sampling
output = model.generate(input_ids, 
                        max_length=50, 
                        do_sample=True,  # Enable sampling-based decoding
                        temperature=0.7, 
                        top_k=50,  # Adjust the top-K value as needed
                        num_return_sequences=5)

# Decode and print the generated text
for i, sample_output in enumerate(output):
    print(f"\n=== Sample {i+1} ===\n")
    print(tokenizer.decode(sample_output, skip_special_tokens=True))

