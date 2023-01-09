from flask import Flask, request, jsonify

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

app = Flask('bert')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.get_json()
    # Tokenize the input text
    input_ids = torch.tensor([tokenizer.encode(input_text['input_text'], add_special_tokens=True)])
    # Convert input_ids to tensor and pass through the model
    output = model(input_ids)[0]
    # Get the index of the highest probability
    output_index = output.argmax().item()
    # Get the label corresponding to the index
    output_label = model.config.id2label[output_index]
    
    result = {
        'label': output_label
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)