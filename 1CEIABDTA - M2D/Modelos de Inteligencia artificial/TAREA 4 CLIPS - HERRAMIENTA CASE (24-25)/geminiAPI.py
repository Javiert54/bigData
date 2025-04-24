from google import genai
from pydantic import BaseModel
import json

# GET the API key: https://aistudio.google.com/app/apikey?hl=es-419


# Define the model to be used for translation
MODEL = "gemini-2.0-flash"

# List of programming languages supported for translation
with open("languages.json") as f:
    # Load the list of languages from the JSON file
    LANGUAGE_REQUESTED = json.load(f)["languages"]
print(LANGUAGE_REQUESTED)

# Define a Pydantic model for the request body
# Define a Pydantic model for error items in the JSON schema
class ErrorItem(BaseModel):
    errorType: str  # Type of the error (e.g., syntax, runtime, etc.)
    errorMessage: str  # Description of the error

# Define a Pydantic model for the JSON schema used in the API response
class JsonSchema(BaseModel):
    syntaxCorrect: bool  # Indicates if the Java code is syntactically correct
    codeInJava: str  # The original Java code
    errors: list[ErrorItem]  # List of errors if the Java code is not valid
    languageRequested: str  # The requested programming language for translation
    codeTranslated: str  # The translated code in the requested language

# Configuration for the API response
CONFIG = {'response_mime_type': 'application/json',
          'response_schema': JsonSchema}

# Read the API key from the geminiAPI-key.txt file
try:
    with open("geminiAPI-key.txt") as f:
        api_key = f.read().strip()
        client = genai.Client(api_key=api_key)
except FileNotFoundError:
    # Handle the case where the API key file is missing
    print("Error: geminiAPI-key.txt not found. Please create the file or set the GEMINI_API_KEY environment variable.")
    api_key = None
    client = None


def geminiResponse(java_code, language_requested=0, client=client, model=MODEL, config=CONFIG):
    """
    Translates Java code to the requested programming language using the Gemini API.
    Args:
        java_code (str): The Java code to be translated.
        language_requested (int): The index of the programming language in the LANGUAGE_REQUESTED list.
        client (genai.Client): The Gemini API client.
        model (str): The model to use for the translation.
        config (dict): The configuration for the translation.
    Returns:
        str: The translated code in the requested programming language in JSON format.
    """
    
    # Construct the prompt for the Gemini API
    prompt = """Please, translate the code in java provided to the programming language requested using this JSON schema:
    {{
        "syntaxCorrect": "boolean",
        "codeInJava": "string",
        "errors": [
            {{
                "errorType": "string",
                "errorMessage": "string"
            }}
        ],
        "languageRequested": "string",
        "codeTranslated": "string",
    }}

    Here is a brief explanation of the JSON schema:
    - syntaxCorrect: true if the code is syntactically correct in java, false otherwise.
    - codeInJava: the code in java provided.
    - errors: an array of error objects if the code is not syntactically correct in java. Each error object contains:
        - errorType: the type of the error (e.g., syntax, runtime, etc.)
        - errorMessage: a description of the error.
    - languageRequested: the programming language requested for the translation.
    - codeTranslated: the code translated to the requested programming language.

    The code in Java is:
    ```java
    %s
    ```
    The programming language requested is: %s
    """ % (java_code, LANGUAGE_REQUESTED[language_requested])

    # Send the prompt to the Gemini API and get the response
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    try:
        # Return the response text
        return response.text
    except Exception as e:
        # Handle any errors that occur during the API call
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Run the geminiResponse function with the Java code and Python as the requested language
    JAVA_CODE = """
    // Java code for class pepe3f
    public class pepe3f {
    }
    """
    # 0 = Python, 1 = JavaScript, 2 = C++, 3 = Go, 4 = Ruby, 5 = PHP, 6 = Swift, 7 = Kotlin
    print(geminiResponse(JAVA_CODE, language_requested=0))