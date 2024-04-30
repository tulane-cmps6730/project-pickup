from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

#Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")  #Load your fine-tuned model here, change model path as needed

#Define a prompt
#we manually changed the prompt
prompt = "My love for you is like a loop"

#tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

#Generate text using top-K sampling
output = model.generate(input_ids, 
                        max_length=50, 
                        do_sample=True,  #true so we can use temp
                        temperature=0.7, 
                        top_k=50,  
                        num_return_sequences=3)


'''
def generate_pickup_lines(model, tokenizer, start_text, num_lines=2, max_length=50, temperature=0.7, top_k=50):
    #tokenize the input text
    input_ids = tokenizer.encode(start_text, return_tensors='pt')
    #added eos since it kept giving warnings
    model.config.pad_token_id = model.config.eos_token_id

    #Generate pickup lines
    #we don't care about the gradients
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids,
                                  max_length=max_length,
                                  do_sample=True,
                                  temperature=temperature,
                                  top_k=top_k,
                                  num_return_sequences=num_lines)
    
    #Decode the generated lines
    pickup_lines = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return pickup_lines
'''

#Decode and print the generated text
for i, sample_output in enumerate(output):
    print(f"\n=== Sample {i+1} ===\n")
    print(tokenizer.decode(sample_output, skip_special_tokens=True))

