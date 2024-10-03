import torch
from flask import Flask, render_template, request, jsonify
import numpy as np
import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = Flask(__name__)

#file = r"C:\Users\cyril\PycharmProjects\verdict\T5-small_model"
file = r"C:\Users\cyril\PycharmProjects\verdict\Pegasus_model"
#tokenizer = AutoTokenizer.from_pretrained(file, model_max_length=1024)
#model = AutoModelForSeq2SeqLM.from_pretrained(file)

# Load the model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained(file)
tokenizer = PegasusTokenizer.from_pretrained(file, model_max_length=512)
#print(2)
# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = model.to(device)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/text_process', methods=['POST'])
def text_process():
    try:
        #print(device)
        input_prompt = request.form['case_description']
        input_ids = tokenizer.encode(input_prompt, truncation=True, max_length=350, return_tensors="pt")
        input_ids = input_ids.to(model.device)

        generated_verdict = model.generate(input_ids, max_length=350, num_return_sequences=1)[0]
        #print(generated_verdict)
        # Convert the tensor back to CPU if necessary
        judgment = tokenizer.decode(generated_verdict.cpu(), skip_special_tokens=True)
        # Print the generated verdict
        print(judgment)
        #judgment = "hello"


        #Render the detect template and pass the result variable
        return render_template('output.html', judgment=judgment)

    except Exception as e:
        print(f"Error in upload: {e}")
        return jsonify({'error': 'An error occurred'})


if __name__ == '__main__':
    app.run(debug=True)