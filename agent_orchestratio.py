from swarm import Agent, Swarm
from swarm.repl import run_demo_loop

# Define functions for the agents
def transfer_to_flight_modification():
    return flight_modification_agent

def transfer_to_flight_cancel():
    return flight_cancel_agent

def transfer_to_flight_change():
    return flight_change_agent

def transfer_to_lost_baggage():
    return lost_baggage_agent

def transfer_to_triage():
    return triage_agent

def escalate_to_agent(reason=None):
    return f"Escalating to agent: {reason}" if reason else "Escalating to agent"

def valid_to_change_flight():
    return "Customer is eligible to change flight"

def change_flight():
    return "Flight was successfully changed!"

def initiate_refund():
    return "Refund initiated"

def initiate_flight_credits():
    return "Successfully initiated flight credits"

def case_resolved():
    return "Case resolved. No further questions."

def initiate_baggage_search():
    return "Baggage was found!"

def triage_instructions(context_variables):
    customer_context = context_variables.get("customer_context", "")
    flight_context = context_variables.get("flight_context", "")
    return f"""You are to triage a user's request and provide relevant information or transfer to the appropriate agent.
    Use the following context information to assist the user:
    
    Customer Context:
    {customer_context}
    
    Flight Context:
    {flight_context}
    
    If the user asks about flight details that are available in the context, provide that information directly.
    If the user's request requires modification of the flight, transfer to the Flight Modification Agent.
    If the request is about lost baggage, transfer to the Lost Baggage Agent.
    When you need more information, ask a direct question without explaining why you're asking it.
    Do not share your thought process with the user or make unreasonable assumptions."""

def flight_modification_instructions(context_variables):
    customer_context = context_variables.get("customer_context", "")
    flight_context = context_variables.get("flight_context", "")
    return f"""You are a Flight Modification Agent for a customer service airlines company.
    Use the following context information to assist the user:
    
    Customer Context:
    {customer_context}
    
    Flight Context:
    {flight_context}
    
    You are an expert customer service agent deciding which sub-intent the user should be referred to.
    If the user wants to cancel their flight, transfer to the Flight Cancel Agent.
    If the user wants to change their flight, transfer to the Flight Change Agent.
    If the user asks about flight details that are available in the context, provide that information directly.
    Ask clarifying questions if needed, but don't explain why you're asking."""

# Define the agents
triage_agent = Agent(
    name="Triage Agent",
    instructions=triage_instructions,
    functions=[transfer_to_flight_modification, transfer_to_lost_baggage]
)

flight_modification_agent = Agent(
    name="Flight Modification Agent",
    instructions=flight_modification_instructions,
    functions=[transfer_to_flight_cancel, transfer_to_flight_change],
    parallel_tool_calls=False
)

flight_cancel_agent = Agent(
    name="Flight Cancel Agent",
    instructions="""You are a Flight Cancel Agent. Follow these steps:
    1. Confirm which flight the customer is asking to cancel.
    2. Confirm if the customer wants a refund or flight credits.
    3. Initiate the appropriate action (refund or flight credits).
    4. Inform the customer of the next steps.
    5. If the customer has no further questions, resolve the case.""",
    functions=[escalate_to_agent, initiate_refund, initiate_flight_credits, transfer_to_triage, case_resolved]
)

flight_change_agent = Agent(
    name="Flight Change Agent",
    instructions="""You are a Flight Change Agent. Follow these steps:
    1. Verify the flight details and the reason for the change request.
    2. Check if the customer is eligible to change their flight.
    3. If eligible, suggest alternative flights and check for availability.
    4. Inform the customer of any fare differences or additional charges.
    5. Process the flight change if the customer agrees.
    6. If the customer has no further questions, resolve the case.""",
    functions=[escalate_to_agent, change_flight, valid_to_change_flight, transfer_to_triage, case_resolved]
)

lost_baggage_agent = Agent(
    name="Lost Baggage Agent",
    instructions="""You are a Lost Baggage Agent. Follow these steps:
    1. Gather information about the lost baggage (description, last seen location, etc.).
    2. Initiate a baggage search.
    3. If the baggage is found, arrange for delivery to the customer's address.
    4. If the baggage is not found, escalate to a human agent.
    5. If the customer has no further questions, resolve the case.""",
    functions=[escalate_to_agent, initiate_baggage_search, transfer_to_triage, case_resolved]
)

# Set up the context variables
context_variables = {
    "customer_context": """Here is what you know about the customer's details:
    1. CUSTOMER_ID: customer_12345
    2. NAME: John Doe
    3. PHONE_NUMBER: (123) 456-7890
    4. EMAIL: johndoe@example.com
    5. STATUS: Premium
    6. ACCOUNT_STATUS: Active
    7. BALANCE: $0.00
    8. LOCATION: 1234 Main St, San Francisco, CA 94123, USA
    """,
    "flight_context": """The customer has an upcoming flight from LGA (Laguardia) in NYC to LAX in Los Angeles.
    The flight # is 1919. The flight departure date is 3pm ET, 5/21/2024."""
}

if __name__ == "__main__":
    run_demo_loop(triage_agent, context_variables=context_variables, debug=True)
