# CMPS 6730 Project Pickup
# Mackenzie Bookamer and Thalia Koutsougeras

**Are you a graph? 
Because you’ve got my heart all connected.**

That's the power of the *PickupMaster 9000*. This CS/math-themed pickup line generator is here to meet your needs, whether it's to complete the end of your pickup line or to practice honing your skills with you. By fine-tuning the pre-trained language model (PLM) GPT-2 from OpenAI with a wide variety of math and computer science pickup lines, we were able to generate coherent and subject-relevant pickup lines. We pull the PLM from the HuggingFace model hub and use the GPT-2 Tokenizer as well as Transformers Trainer class to process and train our data, then subsequently feed user input back through our fine-tuned model to generate a response. To facilitate user interaction with our model, we created a Flask app that takes in user input and generates 2 pickup lines from our fine tuned model - better than the PLM without fine-tuning! 

Below is a look at our Flask app in action!


### Contents
- [FINAL CODE](FINAL CODE): where all our final code lies
- [ourflask](ourflask): our flask app files

- [docs](docs): template to create slides for project presentations
- [nlp](nlp): Python project code
- [notebooks](notebooks): Jupyter notebooks for project development and experimentation
- [report](report): LaTeX report
- [tests](tests): unit tests for project code
