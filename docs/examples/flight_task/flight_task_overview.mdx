# Flight Booking Task Bundle

This example demonstrates a Task Bundle for the RewardKit Agent Evaluation Framework. It includes everything needed to evaluate an agent's ability to book a flight through a series of tools.

## Components

- `reward.py`: Reward function that evaluates if the agent successfully completed the flight booking
- `tools.py`: Tool registry with flight search, booking, and payment tools
- `seed.sql`: Initial database state with flights, booking tables, and sample data
- `task.jsonl`: Task specifications for different flight booking scenarios

## Usage

To run the evaluation:

```bash
cd examples/
export MODEL_AGENT=openai/gpt-4o-mini  # Specify which model to evaluate
reward-kit agent-eval --dataset flight_task/task.jsonl
```

## Task Description

The agent is tasked with booking a flight based on user specifications. To complete the task successfully, the agent must:

1. Search for flights matching the user's criteria
2. Create a booking for an appropriate flight
3. Complete payment for the booking

The reward function evaluates both task completion (binary success/failure) and gives partial credit for completing intermediate steps.

## Tool Functionality

- `search_flights`: Search for available flights by origin, destination, and date
- `create_booking`: Reserve a seat on a specific flight
- `pay_booking`: Process payment for a reserved booking
- `get_booking`: Retrieve details about an existing booking

## Database Schema

The task uses a SQLite database with the following schema:

- `flights`: Available flights with seat information
- `bookings`: Flight reservations
- `payments`: Payment records
- `tool_calls`: Log of agent tool usage

## Expected Behavior

A successful agent should:

1. Parse the user's request to understand their needs
2. Use the search_flights tool to find available options
3. Select an appropriate flight based on user preferences
4. Create a booking with the correct passenger name
5. Complete the payment process
6. Confirm the booking details to the user