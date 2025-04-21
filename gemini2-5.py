from google import genai
MODEL = "gemini-2.0-flash"
PROMPT = input("Enter your prompt > ")

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

