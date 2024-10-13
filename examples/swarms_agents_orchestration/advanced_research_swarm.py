import os
from swarm import Swarm, Agent
from swarm.types import Result
import openai
from termcolor import colored
import json
from datetime import datetime

# Set up Perplexity API
openai.api_key = os.getenv('PERPLEXITY_API_KEY')
openai.base_url = 'https://api.perplexity.ai'

print(colored("Initializing Swarm client...", "yellow"))
client = Swarm()

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

def perform_perplexity_search(query):
    """Perform a search using Perplexity API."""
    print(colored(f"Performing Perplexity search for: {query}", "cyan"))
    
    response = openai.chat.completions.create(
        model="llama-3.1-sonar-small-128k-online",
        messages=[
            {"role": "system", "content": f"You are a helpful research assistant. Provide a concise answer to the query. Today's date is {current_date}."},
            {"role": "user", "content": query}
        ]
    )
    print(colored("Perplexity search completed", "cyan"))
    return response.choices[0].message.content

def transfer_to_research_team(context_variables):
    """Transfer control to the research team."""
    print(colored("Transferring to research team...", "yellow"))
    return research_coordinator

def save_research_interaction(interaction):
    """Save research team interaction to a file."""
    print(colored("Saving research interaction to file...", "cyan"))
    with open("research_interactions.json", "a") as f:
        json.dump(interaction, f)
        f.write("\n")
    print(colored("Research interaction saved", "cyan"))

def load_agent_history(agent_name):
    """Load agent's message history from file."""
    filename = f"{agent_name}_history.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def save_agent_history(agent_name, history):
    """Save agent's message history to file."""
    filename = f"{agent_name}_history.json"
    with open(filename, "w") as f:
        json.dump(history, f)

print(colored("Creating Research Agent 1...", "yellow"))
research_agent_1 = Agent(
    name="Research Agent 1",
    instructions=f"You are a research agent collaborating with another agent to answer a query. Build upon the information provided by your partner and contribute new insights. Perform searches when necessary. Today's date is {current_date}.",
    functions=[perform_perplexity_search]
)

print(colored("Creating Research Agent 2...", "yellow"))
research_agent_2 = Agent(
    name="Research Agent 2",
    instructions=f"You are a research agent collaborating with another agent to answer a query. Build upon the information provided by your partner and contribute new insights. Perform searches when necessary. Today's date is {current_date}.",
    functions=[perform_perplexity_search]
)

def research_coordination(context_variables):
    """Coordinate the research between two agents."""
    query = context_variables.get("research_query", "")
    print(colored(f"Starting research coordination for query: {query}", "yellow"))
    
    research_messages = [{"role": "system", "content": f"Collaborate to answer this query: {query}. Today's date is {current_date}."}]
    agents = [research_agent_1, research_agent_2]
    
    # Load previous history for both agents
    agent_1_history = load_agent_history("Research Agent 1")
    agent_2_history = load_agent_history("Research Agent 2")
    
    for turn in range(1, 4):  # Three turns of conversation
        current_agent = agents[turn % 2]
        print(colored(f"Research turn {turn} starting with {current_agent.name}", "cyan"))
        
        # Add the query to the messages for the first turn
        if turn == 1:
            research_messages.append({"role": "user", "content": query})
        
        # Combine previous history with current research messages
        if current_agent == research_agent_1:
            combined_messages = agent_1_history + research_messages
        else:
            combined_messages = agent_2_history + research_messages
        
        response = client.run(agent=current_agent, messages=combined_messages)
        last_message = response.messages[-1]
        print(colored(f"{current_agent.name}: {last_message['content']}", "green"))
        
        research_messages.append(last_message)
        save_research_interaction({"agent": current_agent.name, "message": last_message['content']})
        
        # Update agent's history
        if current_agent == research_agent_1:
            agent_1_history.append(last_message)
        else:
            agent_2_history.append(last_message)
        
        # Prepare the next message for the other agent
        if turn < 3:
            research_messages.append({
                "role": "user",
                "content": f"Your research partner said: '{last_message['content']}'. Please continue the research based on this information. Remember, today's date is {current_date}."
            })
    
    # Save updated history for both agents
    save_agent_history("Research Agent 1", agent_1_history)
    save_agent_history("Research Agent 2", agent_2_history)
    
    final_answer = research_messages[-1]['content']
    print(colored("Research coordination completed", "yellow"))
    return Result(value=final_answer, context_variables={"research_result": final_answer})

print(colored("Creating Research Coordinator...", "yellow"))
research_coordinator = Agent(
    name="Research Coordinator",
    instructions=f"You are a research coordinator. Manage the research process between two agents. Today's date is {current_date}.",
    functions=[research_coordination]
)

print(colored("Creating Main Agent...", "yellow"))
main_agent = Agent(
    name="Main Agent",
    instructions=f"You are a helpful assistant. When the user asks for research, transfer to the research team. Today's date is {current_date}.",
    functions=[transfer_to_research_team]
)

def run_conversation():
    print(colored("Starting conversation...", "yellow"))
    messages = [{"role": "system", "content": f"You are a helpful assistant. Today's date is {current_date}."}]
    while True:
        user_input = input(colored("You: ", "blue"))
        if user_input.lower() == "exit":
            print(colored("Exiting conversation...", "yellow"))
            break
        
        print(colored("Processing user input...", "cyan"))
        messages.append({"role": "user", "content": user_input})
        response = client.run(
            agent=main_agent,
            messages=messages,
            context_variables={"research_query": user_input}
        )
        
        assistant_message = response.messages[-1]['content']
        print(colored(f"Assistant: {assistant_message}", "magenta"))
        messages.extend(response.messages)
        print(colored("Turn completed", "cyan"))

if __name__ == "__main__":
    print(colored("Starting advanced_swarm.py", "yellow"))
    run_conversation()
    print(colored("advanced_swarm.py execution completed", "yellow"))
