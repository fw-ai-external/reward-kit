"""
Refactored flight booking tools for the new agent evaluation framework.
These tools interact with an SQLResource instance.
"""

import json
from typing import Dict, Any, List, Optional

# Assuming SQLResource is in reward_kit.agent and provides the step method
# For type hinting, we can use SQLResource if it's importable, or ForkableResource
from reward_kit.agent import SQLResource # Or use ForkableResource for more generic type hint

# Helper to log tool calls
async def _log_tool_call(db_resource: SQLResource, tool_name: str, tool_parameters: Dict[str, Any]):
    log_query = "INSERT INTO tool_calls (tool_name, parameters) VALUES (:tool_name, :parameters)"
    try:
        # Serialize parameters to JSON string for logging
        parameters_str = json.dumps(tool_parameters)
    except TypeError:
        parameters_str = str(tool_parameters) # Fallback if not JSON serializable

    log_query_params = {"tool_name": tool_name, "parameters": parameters_str}
    
    log_result = await db_resource.step(
        action_name="execute_sql", 
        action_params={"query": log_query, "parameters": log_query_params}
    )
    if log_result.get("status") != "success":
        print(f"Warning: Failed to log tool call for {tool_name}. Error: {log_result.get('message')}")


async def search_flights(
    origin: str, 
    dest: str, 
    date: str, 
    db_resource: SQLResource
) -> List[Dict[str, Any]]:
    """
    Search for available flights matching the criteria using SQLResource.
    """
    tool_name = "search_flights"
    await _log_tool_call(db_resource, tool_name, {"origin": origin, "dest": dest, "date": date})

    query = """
        SELECT id, airline, origin, dest, depart, arrive, price, seats_available
        FROM flights
        WHERE origin=:origin AND dest=:dest AND date(depart)=:date AND seats_available>0
        ORDER BY price ASC
    """
    params = {"origin": origin, "dest": dest, "date": date}
    
    result = await db_resource.step(action_name="fetch_all_sql", action_params={"query": query, "parameters": params})
    
    if result.get("status") == "success":
        return result.get("result", [])
    else:
        print(f"Error in {tool_name}: {result.get('message')}")
        # Return an empty list or a dict with error, consistent with agent expectations
        return [] # Or: return {"error": result.get("message", "Failed to search flights")}


async def create_booking(
    flight_id: int, 
    passenger: str, 
    db_resource: SQLResource
) -> Dict[str, Any]:
    """
    Create a new flight booking using SQLResource.
    """
    tool_name = "create_booking"
    await _log_tool_call(db_resource, tool_name, {"flight_id": flight_id, "passenger": passenger})

    # Check flight availability
    flight_check_query = "SELECT id, seats_available, price FROM flights WHERE id=:id AND seats_available>0"
    flight_check_params = {"id": flight_id}
    flight_result = await db_resource.step(action_name="fetch_one_sql", action_params={"query": flight_check_query, "parameters": flight_check_params})

    if flight_result.get("status") != "success" or not flight_result.get("result"):
        return {"error": "Flight not found or no seats available", "details": flight_result.get("message")}
    
    flight_data = flight_result["result"]
    flight_price = flight_data["price"]

    # Generate booking ID (using SQLResource to execute SQL for randomness)
    # SQLite's lower(hex(randomblob(4))) is a good way to get a short random hex string
    random_id_query = "SELECT lower(hex(randomblob(4)))"
    booking_id_result = await db_resource.step(action_name="fetch_val_sql", action_params={"query": random_id_query})

    if booking_id_result.get("status") != "success" or not booking_id_result.get("result"):
        return {"error": "Failed to generate booking ID", "details": booking_id_result.get("message")}
    booking_id = booking_id_result["result"]

    # Create booking
    insert_booking_query = """
        INSERT INTO bookings(id, flight_id, passenger, status, price)
        VALUES(:id, :flight_id, :passenger, 'reserved', :price)
    """
    insert_booking_params = {
        "id": booking_id, "flight_id": flight_id, "passenger": passenger, "price": flight_price
    }
    insert_result = await db_resource.step(action_name="execute_sql", action_params={"query": insert_booking_query, "parameters": insert_booking_params})
    if insert_result.get("status") != "success":
        return {"error": "Failed to create booking record", "details": insert_result.get("message")}

    # Update flight seats
    update_seats_query = "UPDATE flights SET seats_available = seats_available - 1 WHERE id = :id"
    update_seats_params = {"id": flight_id}
    update_seats_result = await db_resource.step(action_name="execute_sql", action_params={"query": update_seats_query, "parameters": update_seats_params})
    if update_seats_result.get("status") != "success":
        # This is problematic; booking created but seats not updated. May need rollback logic in a real system.
        print(f"Warning: Booking {booking_id} created, but failed to update flight seat count. Error: {update_seats_result.get('message')}")
        # Proceed to return booking details, but with a warning or partial success indicator if desired.

    # Fetch and return booking details
    final_booking_query = """
        SELECT b.id as booking_id, b.status, b.price, f.airline, f.origin, f.dest, f.depart, f.arrive
        FROM bookings b JOIN flights f ON b.flight_id = f.id
        WHERE b.id = :id
    """
    final_booking_params = {"id": booking_id}
    final_booking_details = await db_resource.step(action_name="fetch_one_sql", action_params={"query": final_booking_query, "parameters": final_booking_params})

    if final_booking_details.get("status") == "success":
        return final_booking_details.get("result", {"error": "Failed to retrieve booking details after creation"})
    else:
        return {"error": "Booking created, but failed to retrieve final details", "booking_id": booking_id, "details": final_booking_details.get("message")}


async def pay_booking(
    booking_id: str, 
    payment_method: str, 
    db_resource: SQLResource
) -> Dict[str, Any]:
    """
    Pay for a reserved booking using SQLResource.
    """
    tool_name = "pay_booking"
    await _log_tool_call(db_resource, tool_name, {"booking_id": booking_id, "payment_method": payment_method})

    # Check booking status
    booking_check_query = "SELECT id, status, price FROM bookings WHERE id=:id AND status='reserved'"
    booking_check_params = {"id": booking_id}
    booking_result = await db_resource.step(action_name="fetch_one_sql", action_params={"query": booking_check_query, "parameters": booking_check_params})

    if booking_result.get("status") != "success" or not booking_result.get("result"):
        return {"error": "Booking not found or not in 'reserved' status", "details": booking_result.get("message")}
    
    booking_data = booking_result["result"]
    booking_price = booking_data["price"]

    # Update booking status
    update_booking_query = "UPDATE bookings SET status = 'paid', payment_method = :payment_method WHERE id = :id"
    update_booking_params = {"payment_method": payment_method, "id": booking_id}
    update_status_result = await db_resource.step(action_name="execute_sql", action_params={"query": update_booking_query, "parameters": update_booking_params})
    if update_status_result.get("status") != "success":
        return {"error": "Failed to update booking status to paid", "details": update_status_result.get("message")}

    # Record payment
    random_id_query = "SELECT lower(hex(randomblob(4)))" # For payment_id
    payment_id_result = await db_resource.step(action_name="fetch_val_sql", action_params={"query": random_id_query})
    if payment_id_result.get("status") != "success" or not payment_id_result.get("result"):
        return {"error": "Failed to generate payment ID", "details": payment_id_result.get("message")}
    payment_id = payment_id_result["result"]
    
    insert_payment_query = "INSERT INTO payments(id, booking_id, amount, method) VALUES(:id, :booking_id, :amount, :method)"
    insert_payment_params = {"id": payment_id, "booking_id": booking_id, "amount": booking_price, "method": payment_method}
    insert_payment_result = await db_resource.step(action_name="execute_sql", action_params={"query": insert_payment_query, "parameters": insert_payment_params})
    if insert_payment_result.get("status") != "success":
        # Booking status might be 'paid' but payment record failed. This is an inconsistency.
        print(f"Warning: Booking {booking_id} status updated to paid, but failed to record payment. Error: {insert_payment_result.get('message')}")
        return {"error": "Booking status updated, but failed to record payment entry.", "booking_id": booking_id, "status": "paid", "payment_record_error": insert_payment_result.get("message")}

    return {
        "success": True, "booking_id": booking_id, "payment_id": payment_id,
        "amount": booking_price, "status": "paid", "payment_method": payment_method
    }


async def get_booking_details(
    booking_id: str, 
    db_resource: SQLResource
) -> Dict[str, Any]:
    """
    Get details of a specific booking using SQLResource.
    """
    tool_name = "get_booking_details" # Renamed from get_booking for clarity
    await _log_tool_call(db_resource, tool_name, {"booking_id": booking_id})

    query = """
        SELECT b.id as booking_id, b.passenger, b.status, b.price, b.payment_method,
               f.airline, f.origin, f.dest, f.depart, f.arrive
        FROM bookings b JOIN flights f ON b.flight_id = f.id
        WHERE b.id = :id
    """
    params = {"id": booking_id}
    
    result = await db_resource.step(action_name="fetch_one_sql", action_params={"query": query, "parameters": params})
    
    if result.get("status") == "success":
        if result.get("result"):
            return result.get("result")
        else:
            return {"error": "Booking not found"}
    else:
        print(f"Error in {tool_name}: {result.get('message')}")
        return {"error": result.get("message", "Failed to get booking details")}
