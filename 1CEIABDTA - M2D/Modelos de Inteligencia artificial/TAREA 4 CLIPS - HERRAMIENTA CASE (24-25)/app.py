from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import os

app = Flask(__name__)

# Ruta para servir la página principal
@app.route('/')
def home():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)