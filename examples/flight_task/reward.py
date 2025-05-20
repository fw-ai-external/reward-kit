"""
Flight booking reward function for agent evaluation.

This reward function evaluates if an agent successfully booked a flight by checking the database
for a paid booking record for the specified passenger.
"""

from typing import Any, Dict, List

from sqlalchemy.engine.base import Connection  # Assuming db is an SQLAlchemy connection

from reward_kit import EvaluateResult, Message, MetricResult, reward_function


@reward_function
def evaluate(
    messages: List[Message],
    *,
    db: Connection,
    end_goal_sql: str = None,
    passenger: str = "Alice",
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate if a flight booking was successfully completed.

    Args:
        messages: List of conversation messages (currently unused by this function but part of standard signature)
        db: Database connection object.
        end_goal_sql: Optional SQL query to check the end goal (overrides default).
        passenger: Name of the passenger to check for booking.
        **kwargs: Additional arguments.

    Returns:
        EvaluateResult with score and metrics.
    """
    # Default SQL query if not provided
    if not end_goal_sql:
        end_goal_sql = f"SELECT COUNT(*) FROM bookings WHERE passenger='{passenger}' AND status='paid'"

    # Execute the query
    from sqlalchemy import text

    result = db.execute(text(end_goal_sql)).scalar()
    success = bool(result) and result > 0

    # Check if there was at least one flight search
    search_count = db.execute(
        text("SELECT COUNT(*) FROM tool_calls WHERE tool_name='search_flights'")
    ).scalar()

    # Check how many booking attempts were made
    booking_count = db.execute(
        text("SELECT COUNT(*) FROM tool_calls WHERE tool_name='create_booking'")
    ).scalar()

    # Check how many payment attempts were made
    payment_count = db.execute(
        text("SELECT COUNT(*) FROM tool_calls WHERE tool_name='pay_booking'")
    ).scalar()

    # Create metrics dictionary compatible with EvaluateResult
    metrics_dict: Dict[str, MetricResult] = {
        "task_complete": MetricResult(
            score=1.0 if success else 0.0,
            is_score_valid=success,
            reason=(
                "Successfully booked and paid for flight"
                if success
                else "Failed to complete booking"
            ),
        ),
        "search_flights": MetricResult(
            score=min(1.0, search_count / 1.0),
            is_score_valid=search_count > 0,
            reason=f"Agent searched for flights {search_count} times",
        ),
        "create_booking": MetricResult(
            score=min(1.0, booking_count / 1.0),
            is_score_valid=booking_count > 0,
            reason=f"Agent created {booking_count} bookings",
        ),
        "pay_booking": MetricResult(
            score=min(1.0, payment_count / 1.0),
            is_score_valid=payment_count > 0,
            reason=f"Agent made {payment_count} payment attempts",
        ),
    }

    # Final score is 1.0 if successful, otherwise partial credit for steps completed
    if success:
        score = 1.0
        reason = "Task completed successfully: flight booked and paid"
    else:
        # Calculate partial credit based on steps completed
        progress_score = (
            0.2 * min(1.0, search_count / 1.0)  # 20% for searching
            + 0.3 * min(1.0, booking_count / 1.0)  # 30% for booking
            + 0.5 * min(1.0, payment_count / 1.0)  # 50% for payment
        )
        score = progress_score
        reason = f"Task incomplete: {progress_score:.2f} progress score"

    # Return as an EvaluateResult object
    return EvaluateResult(
        score=score, reason=reason, metrics=metrics_dict, is_score_valid=success
    )
