from transformers import GPT2Config

# Load the existing model's configuration
model_config = GPT2Config.from_pretrained("./project-pickup-tuned")

# Save the configuration to the same directory as your fine-tuned model
model_config.save_pretrained("./project-pickup-tuned")
