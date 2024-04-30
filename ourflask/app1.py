import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request
from generation import generate_pickup_lines

#used this website tutorial: https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3
#used this wesbite tutorial: https://www.geeksforgeeks.org/flask-tutorial/


#can run the app on your computer by changing the model_path to where the fine tuned model is located in your computer
model_path = './project-pickup-tuned'
model = GPT2LMHeadModel.from_pretrained(model_path)


#Load the tokenizer from GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

#Set the model to evaluation mode
model.eval()


#very much line for line from the tutorials we found
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    start_text = request.form['start_text']
    pickup_lines = generate_pickup_lines(model, tokenizer, start_text)
    return render_template('results.html', start_text=start_text, pickup_lines=pickup_lines)

if __name__ == '__main__':
    #kept saying port 5000 was in use, so had to specify 5001
    app.run(debug=True, port=5001)


