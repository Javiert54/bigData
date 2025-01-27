from flask import Flask, render_template, request, jsonify 
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
        diagram_path = os.path.join('generated_files', 'diagram.xmi') 
        if not os.path.exists(diagram_path): 
            return jsonify({'error': 'Archivo diagram.xmi no encontrado'}), 400 
 
        # Ejecutar Traductor.py 
        subprocess.run('ls', check=True)
        subprocess.run(['python', 'Traductor.py'], check=True) 

 
        # Confirmar éxito 
        return jsonify({'message': 'Archivo procesado correctamente'}), 200 
 
    except subprocess.CalledProcessError as e: 
        return jsonify({'error': f'Error al ejecutar Traductor.py: {e}'}), 500 
    except Exception as e: 
        return jsonify({'error': f'Error inesperado: {e}'}), 500 
 
if __name__ == '__main__': 
    app.run(debug=True)