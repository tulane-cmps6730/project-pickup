# CMPS 6730 Project Pickup
# Mackenzie Bookamer and Thalia Koutsougeras

***Are you a graph? 
Because you’ve got my heart all connected.***

That's the power of the *PickupMaster 9000*. This CS/math-themed pickup line generator is here to meet your needs, whether it's to complete the end of your pickup line or to practice honing your skills with you. By fine-tuning the pre-trained language model (PLM) GPT-2 from OpenAI with a wide variety of math and computer science pickup lines, we were able to generate coherent and subject-relevant pickup lines. We pull the PLM from the HuggingFace model hub and use the GPT-2 Tokenizer as well as Transformers Trainer class to process and train our data, then subsequently feed user input back through our fine-tuned model to generate a response. To facilitate user interaction with our model, we created a Flask app that takes in user input and generates 2 pickup lines from our fine tuned model - better than the PLM without fine-tuning! 

Below is a look at our Flask app in action!
<img width="687" alt="flask pg1" src="https://github.com/tulane-cmps6730/project-pickup/assets/100322984/19f5af54-e2f9-497d-9033-2f4e76359c45">
<img width="947" alt="flask pg2" src="https://github.com/tulane-cmps6730/project-pickup/assets/100322984/4dc969cb-23bc-4ab8-9045-89ad9347e42a">


### Contents
- [FINAL CODE](FINAL CODE): where all our final code lies
- [ourflask](ourflask): our flask app files

- [docs](docs): template to create slides for project presentations
- [nlp](nlp): Python project code
- [notebooks](notebooks): Jupyter notebooks for project development and experimentation
- [report](report): LaTeX report
- [tests](tests): unit tests for project code
