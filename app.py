from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = './fine_tuned_flan_t5_round2'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Function to generate a response from the model
def generate_response(human_input, max_length=100):
    input_text = f"Human: {human_input}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# API route for generating responses
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('input_text', '')
    if not user_input.strip():
        return jsonify({'error': 'Input text is empty'}), 400
    
    try:
        response = generate_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
