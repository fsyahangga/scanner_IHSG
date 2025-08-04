import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

with open("your_file.py") as f:  # ganti dengan file aktual
    code = f.read()

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": f"Review this Python code:\n\n{code}"}
    ]
)

print(response.choices[0].message.content)
