# Flask Image Upload Application

This project is a simple Flask application that allows users to upload images to the server. The uploaded images are stored in a designated directory on the server.

## Project Structure

```
flask-image-upload-app
├── app.py                # Main entry point of the Flask application
├── requirements.txt      # Lists the dependencies required for the project
├── static
│   └── uploads           # Directory to store uploaded images
├── templates
│   └── upload.html       # HTML form for uploading images
└── README.md             # Documentation for the project
```

## Requirements

To run this application, you need to have Python installed on your machine. You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory:

   ```
   cd flask-image-upload-app
   ```

2. Run the application:

   ```
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:5000` to access the image upload interface.

## Usage

- Use the provided HTML form to select an image file from your computer.
- Click the "Upload" button to submit the image.
- The uploaded image will be saved in the `static/uploads` directory.

## License

This project is open-source and available under the MIT License.