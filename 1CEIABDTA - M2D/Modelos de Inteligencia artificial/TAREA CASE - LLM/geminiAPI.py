from google import genai
MODEL = "gemini-2.0-flash"
# java_code = input("Enter the Java code to be translated > ")

with open("prompt.txt") as f:
    java_code = f.read().strip()
    
PROMPT = f"""Please, translate the code in java provided to the programming language requested using this JSON schema:
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
{java_code}
"""
# {input("Enter your prompt > ")}
try:
    with open("geminiAPI-key.txt") as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print("Error: geminiAPI-key.txt not found. Please create the file or set the GEMINI_API_KEY environment variable.")
    exit()

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model=MODEL,
    contents=PROMPT,
)
print(response.text)

