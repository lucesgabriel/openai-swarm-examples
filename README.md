# Airline Customer Service Swarm

This example demonstrates a multi-agent setup for handling different customer service requests in an airline context using the Swarm framework. The agents can triage requests, handle flight modifications, cancellations, and lost baggage cases.

## Key Features

- **Context-Aware Agents**: All agents have access to customer and flight context, allowing them to provide accurate information without unnecessary transfers.
- **Triage System**: The Triage Agent efficiently routes requests to specialized agents when necessary.
- **Flight Information**: Agents can directly provide flight details when available in the context.
- **Modification Handling**: Separate agents for flight cancellations and changes ensure specialized handling of these requests.

## Agents

1. **Triage Agent**: Determines the type of request, provides flight information when possible, and transfers to specialized agents when needed.
2. **Flight Modification Agent**: Handles requests related to flight modifications, providing flight information and further triaging to:
   - **Flight Cancel Agent**: Manages flight cancellation requests.
   - **Flight Change Agent**: Manages flight change requests.
3. **Lost Baggage Agent**: Handles lost baggage inquiries.

## Setup and Usage

1. Ensure you have installed all dependencies listed in `requirements.txt`.
2. Run the example using:
