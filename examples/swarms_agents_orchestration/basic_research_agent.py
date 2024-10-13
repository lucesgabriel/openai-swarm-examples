from swarm import Swarm, Agent
from swarm.types import Result
import openai
import os
from termcolor import colored

# Set up Perplexity API
openai.api_key = os.getenv('PERPLEXITY_API_KEY')
openai.base_url = 'https://api.perplexity.ai'

def search_web(query):
    print(colored(f"Searching the web for: {query}", "cyan"))
    try:
        response = openai.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that searches the web for information."},
                {"role": "user", "content": query}
            ]
        )
        result = response.choices[0].message.content
        print(colored("Search completed successfully", "green"))
        return result
    except Exception as e:
        print(colored(f"Error occurred during search: {str(e)}", "red"))
        return f"An error occurred: {str(e)}"

def research_agent_instructions(context_variables):
    return """You are a research assistant capable of searching the web for information.
    Use the search_web function to find answers to user queries.
    Always provide a summary of your findings."""

research_agent = Agent(
    name="Research Agent",
    instructions=research_agent_instructions,
    functions=[search_web],
)

client = Swarm()

def run_research_swarm():
    print(colored("Starting Research Swarm", "yellow"))
    messages = []
    
    while True:
        user_input = input(colored("Ask a question (or type 'exit' to quit): ", "cyan"))
        
        if user_input.lower() == 'exit':
            print(colored("Exiting Research Swarm", "yellow"))
            break
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.run(
            agent=research_agent,
            messages=messages,
        )
        
        print(colored("Research Agent Response:", "magenta"))
        print(response.messages[-1]["content"])
        print()
        
        # Add the agent's response to the message history
        messages.append(response.messages[-1])

if __name__ == "__main__":
    run_research_swarm()
