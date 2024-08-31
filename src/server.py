# server.py
# copied from AIONER_Run.py and converted it to api endpoint
from flask import Flask, request, jsonify
from default_model_singleton import ModelSingleton

app = Flask(__name__)


@app.route('/annotate', methods=['POST'])
def annotate():
    content = request.json.get('content')
    input_format = request.json.get('input_format')
    if input_format not in ['text', 'BioC', 'PubTator']:
        return jsonify({"error": "Invalid input format"}), 400
    if not content:
        return jsonify({"error": "Missing required parameters"}), 400
    model_singleton = ModelSingleton()
    result = model_singleton.process_content(content, input_format=input_format)
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
