import os
import json

model_directory = "./project-pickup-tuned"
config_file = os.path.join(model_directory, "config.json")

if not os.path.exists(config_file):
    with open(config_file, "w") as f:
        config = {
            "model_name": "gpt2-medium",
            "model_type": "Transformer",
            "model_path": "~/project-pickup/gpt-2/project-pickup-tuned.pth",
            # Add any other necessary configuration options here
        }
        json.dump(config, f)
    print("Config file created.")
else:
    print("Config file already exists.")