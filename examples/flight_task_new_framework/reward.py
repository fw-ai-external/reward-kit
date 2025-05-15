"""
Refactored flight booking reward function for the new agent evaluation framework.
"""

from typing import Dict, Any

def evaluate_flight_booking(
    task_achieved: bool,
    tool_usage_counts: Dict[str, int],
    **kwargs: Any # For any other potential future arguments
) -> Dict[str, Any]:
    """
    Evaluates the flight booking task based on goal achievement and tool usage.

    Args:
        task_achieved: Boolean indicating if the main goal (e.g., flight booked and paid) was met.
                       This is determined by the Orchestrator using 'evaluation_criteria'
                       from the task_definition.yaml.
        tool_usage_counts: Dictionary with counts of relevant tool calls,
                           e.g., {"search_flights": 2, "create_booking": 1, "pay_booking": 1}.
                           The Orchestrator is expected to populate this by querying
                           the 'tool_calls' table (or similar mechanism) on the EpisodeResource.
        **kwargs: Additional arguments that might be passed by the Orchestrator.

    Returns:
        A dictionary with "score", "reason", and "metrics",
        compatible with Reward Kit's evaluation result structure.
    """
    search_count = tool_usage_counts.get("search_flights", 0)
    booking_count = tool_usage_counts.get("create_booking", 0)
    payment_count = tool_usage_counts.get("pay_booking", 0)

    # Define metrics based on the provided counts and task achievement
    metrics_dict = {
        "task_goal_achieved": { # Clearer metric name
            "score": 1.0 if task_achieved else 0.0,
            "success": task_achieved,
            "reason": (
                "Agent successfully booked and paid for the flight."
                if task_achieved
                else "Agent did not complete the primary goal of booking and paying for the flight."
            ),
        },
        "search_flights_tool_usage": {
            "score": 1.0 if search_count > 0 else 0.0, # Binary: used or not
            "success": search_count > 0,
            "reason": f"search_flights tool was called {search_count} times.",
            "count": search_count,
        },
        "create_booking_tool_usage": {
            "score": 1.0 if booking_count > 0 else 0.0, # Binary: used or not
            "success": booking_count > 0,
            "reason": f"create_booking tool was called {booking_count} times.",
            "count": booking_count,
        },
        "pay_booking_tool_usage": {
            "score": 1.0 if payment_count > 0 else 0.0, # Binary: used or not
            "success": payment_count > 0,
            "reason": f"pay_booking tool was called {payment_count} times.",
            "count": payment_count,
        },
    }

    # Determine overall score and reason
    if task_achieved:
        score = 1.0
        reason = "Task completed successfully: flight booked and paid."
    else:
        # Simplified partial credit logic for PoC
        if payment_count > 0 and booking_count > 0 : # Assumes payment implies successful booking attempt
            score = 0.75 
            reason = "Task incomplete: Payment was attempted after booking, but primary goal not met."
        elif booking_count > 0:
            score = 0.5
            reason = "Task incomplete: Booking was attempted, but payment was not, or primary goal not met."
        elif search_count > 0:
            score = 0.25
            reason = "Task incomplete: Flights were searched, but no booking was attempted."
        else:
            score = 0.0
            reason = "Task incomplete: No significant flight booking actions (search, book, pay) were taken."
            
    return {"score": score, "reason": reason, "metrics": metrics_dict}
