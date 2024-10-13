import openai
import os

openai.api_key = os.getenv('PERPLEXITY_API_KEY')
openai.base_url = 'https://api.perplexity.ai'

response = openai.chat.completions.create(
    model="llama-3.1-sonar-small-128k-online",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"}
    ]
)

print(response.choices[0].message.content)