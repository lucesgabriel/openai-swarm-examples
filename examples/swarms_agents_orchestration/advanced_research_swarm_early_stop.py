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
    user_input = context_variables.get("user_input", "")
    conversation_history = context_variables.get("conversation_history", [])
    
    # Generate a proper research query based on the conversation context
    research_query = f"Based on the following conversation:\n\n{' '.join([msg['content'] for msg in conversation_history[-5:]])}\n\nGenerate a focused research query for: {user_input}"
    
    return Result(agent=research_coordinator, context_variables={"research_query": research_query})

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
    instructions=f"""You are a research agent collaborating with another agent to answer queries comprehensively. 
    Always consider the full context of the conversation, including any previous topics discussed.
    Build upon the information provided by your partner and contribute new insights. 
    Perform searches when necessary, ensuring they are relevant to the original query and conversation context.
    Today's date is {current_date}. 
    Strive for thorough and in-depth research. Only conclude the research by stating 'RESEARCH COMPLETE' if you are confident that:
    1. All aspects of the query have been addressed comprehensively.
    2. Multiple reliable sources have been consulted and cross-referenced.
    3. Any potential contradictions or nuances in the information have been explored.
    4. The answer provides a well-rounded view of the topic.
    If there's any doubt or potential for further valuable insights, continue the research process.""",
    functions=[perform_perplexity_search]
)

print(colored("Creating Research Agent 2...", "yellow"))
research_agent_2 = Agent(
    name="Research Agent 2",
    instructions=f"""You are a research agent collaborating with another agent to answer queries comprehensively. 
    Always consider the full context of the conversation, including any previous topics discussed.
    Build upon the information provided by your partner and contribute new insights. 
    Perform searches when necessary, ensuring they are relevant to the original query and conversation context.
    Today's date is {current_date}. 
    Strive for thorough and in-depth research. Only conclude the research by stating 'RESEARCH COMPLETE' if you are confident that:
    1. All aspects of the query have been addressed comprehensively.
    2. Multiple reliable sources have been consulted and cross-referenced.
    3. Any potential contradictions or nuances in the information have been explored.
    4. The answer provides a well-rounded view of the topic.
    If there's any doubt or potential for further valuable insights, continue the research process.""",
    functions=[perform_perplexity_search]
)

def research_coordination(context_variables):
    """Coordinate the research between two agents."""
    query = context_variables.get("research_query", "")
    print(colored(f"Starting research coordination for query: {query}", "yellow"))
    
    research_messages = [
        {"role": "system", "content": f"You are collaborating to answer a research query. Today's date is {current_date}."},
        {"role": "user", "content": f"Research query: {query}"}
    ]
    agents = [research_agent_1, research_agent_2]
    
    for turn in range(1, 11):  # Up to 10 turns of conversation
        current_agent = agents[turn % 2]
        print(colored(f"Research turn {turn} starting with {current_agent.name}", "cyan"))
        
        response = client.run(agent=current_agent, messages=research_messages)
        last_message = response.messages[-1]
        print(colored(f"{current_agent.name}: {last_message['content']}", "green"))
        
        research_messages.append(last_message)
        save_research_interaction({"agent": current_agent.name, "message": last_message['content']})
        
        if "RESEARCH COMPLETE" in last_message['content']:
            print(colored("Research completed early.", "yellow"))
            break
        
        if turn < 10:
            research_messages.append({
                "role": "user",
                "content": "Please continue the research based on this information."
            })
    
    final_answer = research_messages[-1]['content'].replace("RESEARCH COMPLETE", "").strip()
    print(colored("Research coordination completed", "yellow"))
    return Result(value=final_answer, context_variables={"research_result": final_answer})

print(colored("Creating Research Coordinator...", "yellow"))
research_coordinator = Agent(
    name="Research Coordinator",
    instructions=f"""You are a research coordinator managing the research process between two agents. 
    Ensure that the research stays focused on the original query and considers the full conversation context.
    If the research seems to be going off-track, redirect the agents to the original topic.
    Encourage thorough and comprehensive research. Only allow the research to conclude when:
    1. All aspects of the query have been addressed comprehensively.
    2. Multiple reliable sources have been consulted and cross-referenced.
    3. Any potential contradictions or nuances in the information have been explored.
    4. The answer provides a well-rounded view of the topic.
    If these criteria are not met, guide the agents to continue their research.
    Today's date is {current_date}.""",
    functions=[research_coordination]
)

print(colored("Creating Main Agent...", "yellow"))
main_agent = Agent(
    name="Main Agent",
    instructions=f"""You are a helpful assistant. When the user asks for research, transfer to the research team. 
    When transferring, make sure to provide the full context of the conversation, including any previous topics discussed.
    This will help the research team stay focused on the correct topic.
    Today's date is {current_date}.""",
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
            context_variables={
                "user_input": user_input,
                "conversation_history": messages[-5:]  # Pass the last 5 messages for context
            }
        )
        
        assistant_message = response.messages[-1]['content']
        print(colored(f"Assistant: {assistant_message}", "magenta"))
        messages.extend(response.messages)
        print(colored("Turn completed", "cyan"))

if __name__ == "__main__":
    print(colored("Starting advanced_swarm.py", "yellow"))
    run_conversation()
    print(colored("advanced_swarm.py execution completed", "yellow"))
