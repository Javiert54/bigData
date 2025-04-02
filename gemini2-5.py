from google import genai
from google.genai import types
from IPython.display import Image, Markdown, Code, HTML, display


with open("geminiAPI-key.txt") as f:
    api_key = f.read().strip()
client = genai.Client(api_key=api_key)


MODEL_ID = "gemini-2.5-pro-exp-03-25"

system_instruction = """
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""

chat = client.models.generate_content(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            code_execution=types.ToolCodeExecution()
        )]
    )
)

def display_code_execution_result(response):
    if not response.candidates or not response.candidates[0].content.parts:
        print("No content found in the response.")
        return

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print("Rendering Markdown:", part.text)  # Debugging output
            display(Markdown(part.text))
        if part.executable_code is not None:
            code_html = f'<pre style="background-color: green;">{part.executable_code.code}</pre>'
            display(HTML(code_html))
        if part.code_execution_result is not None:
            print("Code execution result:", part.code_execution_result.output)  # Debugging output
            display(Markdown(part.code_execution_result.output))
        if part.inline_data is not None:
            display(Image(data=part.inline_data.data, width=800, format="png"))
        display(Markdown("---"))


while True:
    user_input = input("\nUser Input: ")
    if user_input.lower() == "exit":
        break

    response = chat.send_message(user_input)
    display_code_execution_result(response)