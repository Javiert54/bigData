from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import subprocess
import os
import geminiAPI


app = Flask(__name__)

with open("languages.json") as f:
    # ["Python", "JavaScript", "C++", "Go", "Ruby", "PHP", "Swift", "Kotlin"]
    LANGUAGE_REQUESTED = json.load(f)["languages"]

# Ruta para servir la página principal
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_languages', methods=['GET'])
def get_languages():
    """
    Endpoint to get the list of programming languages supported for translation.
    Returns:
        JSON response with the list of programming languages.
    """
    return jsonify(LANGUAGE_REQUESTED)

# Ruta para manejar el archivo generado (diagram.xmi) y ejecutar Traductor.py
@app.route('/process-diagram', methods=['POST'])
def process_diagram():
    try:
        # Verificar si el archivo existe
        diagram_path = os.path.join('process-diagram', 'diagram.xmi')
        if not os.path.exists(diagram_path):
            return jsonify({'error': 'Archivo diagram.xmi no encontrado'}), 400

        # Ejecutar Traductor.py
        # Capturar la salida estándar
        result = subprocess.run(['python', 'Traductor.py'], check=True, capture_output=True, text=True)
        output = result.stdout

        # Confirmar éxito
        return jsonify({'message': 'Archivo generado correctamente', "output":output}), 200

    except subprocess.CalledProcessError as e:
        app.logger.error(f'Error al ejecutar Traductor.py: {e}')
        return jsonify({'error': f'Error al ejecutar Traductor.py: {e}'}), 500
    except Exception as e:
        app.logger.error(f'Error inesperado: {e}')
        return jsonify({'error': f'Error inesperado: {e}'}), 500

# Ruta para manejar la generación de archivos
@app.route('/generated_files', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file.save(os.path.join('process-diagram', 'diagram.xmi'))
        return jsonify({'message': 'File uploaded successfully'}), 200

# Ruta para servir el archivo output.clp
@app.route('/generated_files/output.clp', methods=['GET'])
def get_output_clp():
    try:
        return send_from_directory('generated_files', 'output.clp')
    except FileNotFoundError:
        return jsonify({'error': 'Archivo output.clp no encontrado'}), 404


@app.route('/translateFromJava', methods=['POST'])
def translateFromJava():
    if os.path.exists('geminiAPI-key.txt'):
        try:
            data = request.get_json()
            print("Received JSON data:", data)  # Debugging line
            if not data:
                return jsonify({'error': 'No JSON data received'}), 400
            
            response_str = geminiAPI.geminiResponse(data['javaCode'], data['selectedLanguage'])
            data = json.loads(response_str)
            
            
            return jsonify({'message': 'JSON received successfully', 'data': data}), 200
        except Exception as e:
            app.logger.error(f'Error processing JSON: {e}')
            return jsonify({'error': f'Error processing JSON: {e}'}), 500
    else:
        return jsonify({'error': 'geminiAPI-key.txt not found'}), 500


if __name__ == '__main__':
    app.run(debug=True)
    
