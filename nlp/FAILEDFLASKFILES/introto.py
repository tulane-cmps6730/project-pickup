"""
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run()
"""

from flask import Flask, redirect, url_for, request, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')


@app.route('/success/<name>')
def success(name): #name is the entered text
	# Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    saved_tuned_path = r"C:\Users\Thalia\Desktop\NLP\project\hashtag_cloned\project-pickup\nlp\saved_tuned"
    model = GPT2LMHeadModel.from_pretrained(saved_tuned_path)  # Load your fine-tuned model here

    # Define a prompt - this is taken in from input in flask
    prompt = name

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using top-K sampling
    output = model.generate(input_ids, 
                            max_length=50, 
                            do_sample=True,  # Enable sampling-based decoding
                            temperature=0.7, 
                            top_k=50,  # Adjust the top-K value as needed
                            num_return_sequences=1)

    """
    # Decode and print the generated text
    for i, sample_output in enumerate(output):
        print(f"\n=== Sample {i+1} ===\n")
        print(tokenizer.decode(sample_output, skip_special_tokens=True))
    """
	
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    
    #return 'Generated Line: %s' % generated_text
    return render_template('output.html', generated_text=generated_text)


@app.route('/login', methods=['POST', 'GET'])
def login():
	if request.method == 'POST':
		user = request.form['nm']
		return redirect(url_for('success', name=user))
	else:
		user = request.args.get('nm')
		return redirect(url_for('success', name=user))


if __name__ == '__main__':
	app.run(debug=True)
