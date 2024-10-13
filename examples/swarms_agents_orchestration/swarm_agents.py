from swarm import Swarm, Agent
from swarm.types import Result

# Initialize the Swarm client
client = Swarm()

# Define agent-specific functions
def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    return "Escalated to human agent"

def transfer_to_sales_agent():
    """Use for anything sales or buying related."""
    return sales_agent

def transfer_to_issues_and_repairs():
    """Use for issues, repairs, or refunds."""
    return issues_and_repairs_agent

def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return triage_agent

def execute_order(product, price: int):
    """Price should be in USD."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."

def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id

def execute_refund(item_id, reason="not provided"):
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"

# Define agents
triage_agent = Agent(
    name="Triage Agent",
    model="gpt-4",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "But make your questions subtle and natural."
    ),
    functions=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)

sales_agent = Agent(
    name="Sales Agent",
    model="gpt-4",
    instructions=(
        "You are a sales agent for ACME Inc. "
        "Always answer in a sentence or less. "
        "Follow the following routine with the user: "
        "1. Ask them about any problems in their life related to catching roadrunners. "
        "2. Casually mention one of ACME's crazy made-up products can help. "
        " - Don't mention price. "
        "3. Once the user is bought in, drop a ridiculous price. "
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order."
    ),
    functions=[execute_order, transfer_back_to_triage],
)

issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    model="gpt-4",
    instructions=(
        "You are a customer support agent for ACME Inc. "
        "Always answer in a sentence or less. "
        "Follow the following routine with the user: "
        "1. First, ask probing questions and understand the user's problem deeper. "
        " - unless the user has already provided a reason. "
        "2. Propose a fix (make one up). "
        "3. ONLY if not satisfied, offer a refund. "
        "4. If accepted, search for the ID and then execute refund."
    ),
    functions=[execute_refund, look_up_item, transfer_back_to_triage],
)

# Main loop
def main():
    agent = triage_agent
    messages = []
    context_variables = {}

    while True:
        user_input = input("User: ")
        if not user_input.strip():  # Break the loop if user input is empty
            break
        
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables
        )

        agent = response.agent
        messages = response.messages
        context_variables = response.context_variables

        print(f"{agent.name}: {messages[-1]['content']}")

if __name__ == "__main__":
    main()
