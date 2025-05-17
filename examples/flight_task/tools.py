"""
Flight booking tools for agent evaluation.

This module provides tools for searching flights, creating bookings, and processing payments.
"""

from reward_kit.agent import ToolRegistry

# Create the tool registry
R = ToolRegistry("flight_tools", "Tools for flight booking tasks")


@R.tool(
    description="Search for available flights",
    parameters={"origin": str, "dest": str, "date": str},
)
async def search_flights(origin, dest, date, db):
    """
    Search for available flights matching the criteria.

    Args:
        origin: Origin airport code (e.g., "SFO")
        dest: Destination airport code (e.g., "JFK")
        date: Flight date in YYYY-MM-DD format
        db: Database connection

    Returns:
        List of matching flights
    """
    # Log the tool call
    await db.execute(
        "INSERT INTO tool_calls (tool_name, parameters) VALUES (:tool_name, :parameters)",
        {
            "tool_name": "search_flights",
            "parameters": f"{origin}, {dest}, {date}",
        },
    )

    # Execute the search query
    flights = await db.fetch_all(
        """
        SELECT id, airline, origin, dest, depart, arrive, price, seats_available
        FROM flights
        WHERE origin=:origin AND dest=:dest AND date(depart)=:date AND seats_available>0
        ORDER BY price ASC
        """,
        {"origin": origin, "dest": dest, "date": date},
    )

    return flights


@R.tool(
    description="Create a flight booking",
    parameters={"flight_id": int, "passenger": str},
)
async def create_booking(flight_id, passenger, db):
    """
    Create a new flight booking.

    Args:
        flight_id: ID of the flight to book
        passenger: Name of the passenger
        db: Database connection

    Returns:
        Booking details including booking_id
    """
    # Log the tool call
    await db.execute(
        "INSERT INTO tool_calls (tool_name, parameters) VALUES (:tool_name, :parameters)",
        {
            "tool_name": "create_booking",
            "parameters": f"{flight_id}, {passenger}",
        },
    )

    # Check if the flight exists and has available seats
    flight = await db.fetch_one(
        """
        SELECT id, seats_available, price
        FROM flights
        WHERE id=:id AND seats_available>0
        """,
        {"id": flight_id},
    )

    if not flight:
        return {"error": "Flight not found or no seats available"}

    # Generate a booking ID
    booking_id = await db.fetch_val("SELECT lower(hex(randomblob(4)))")

    # Create the booking
    await db.execute(
        """
        INSERT INTO bookings(id, flight_id, passenger, status, price)
        VALUES(:id, :flight_id, :passenger, 'reserved', :price)
        """,
        {
            "id": booking_id,
            "flight_id": flight_id,
            "passenger": passenger,
            "price": flight["price"],
        },
    )

    # Reduce available seats
    await db.execute(
        """
        UPDATE flights
        SET seats_available = seats_available - 1
        WHERE id = :id
        """,
        {"id": flight_id},
    )

    # Return booking details
    booking = await db.fetch_one(
        """
        SELECT b.id as booking_id, b.status, b.price,
               f.airline, f.origin, f.dest, f.depart, f.arrive
        FROM bookings b
        JOIN flights f ON b.flight_id = f.id
        WHERE b.id = :id
        """,
        {"id": booking_id},
    )

    return booking


@R.tool(
    description="Pay for a reserved booking",
    parameters={"booking_id": str, "payment_method": str},
)
async def pay_booking(booking_id, payment_method, db):
    """
    Pay for a reserved booking.

    Args:
        booking_id: ID of the booking to pay for
        payment_method: Payment method (e.g., "credit", "debit")
        db: Database connection

    Returns:
        Payment confirmation
    """
    # Log the tool call
    await db.execute(
        "INSERT INTO tool_calls (tool_name, parameters) VALUES (:tool_name, :parameters)",
        {
            "tool_name": "pay_booking",
            "parameters": f"{booking_id}, {payment_method}",
        },
    )

    # Check if the booking exists and is in reserved status
    booking = await db.fetch_one(
        """
        SELECT id, status, price
        FROM bookings
        WHERE id=:id AND status='reserved'
        """,
        {"id": booking_id},
    )

    if not booking:
        return {"error": "Booking not found or not in 'reserved' status"}

    # Update the booking status to paid
    await db.execute(
        """
        UPDATE bookings
        SET status = 'paid', payment_method = :payment_method
        WHERE id = :id
        """,
        {"payment_method": payment_method, "id": booking_id},
    )

    # Record the payment
    payment_id = await db.fetch_val("SELECT lower(hex(randomblob(4)))")

    await db.execute(
        """
        INSERT INTO payments(id, booking_id, amount, method)
        VALUES(:id, :booking_id, :amount, :method)
        """,
        {
            "id": payment_id,
            "booking_id": booking_id,
            "amount": booking["price"],
            "method": payment_method,
        },
    )

    # Return payment confirmation
    return {
        "success": True,
        "booking_id": booking_id,
        "payment_id": payment_id,
        "amount": booking["price"],
        "status": "paid",
        "payment_method": payment_method,
    }


@R.tool(description="Get booking details", parameters={"booking_id": str})
async def get_booking(booking_id, db):
    """
    Get details of a booking.

    Args:
        booking_id: ID of the booking
        db: Database connection

    Returns:
        Booking details
    """
    # Log the tool call
    await db.execute(
        "INSERT INTO tool_calls (tool_name, parameters) VALUES (:tool_name, :parameters)",
        {"tool_name": "get_booking", "parameters": booking_id},
    )

    # Get the booking details
    booking = await db.fetch_one(
        """
        SELECT b.id as booking_id, b.passenger, b.status, b.price, b.payment_method,
               f.airline, f.origin, f.dest, f.depart, f.arrive
        FROM bookings b
        JOIN flights f ON b.flight_id = f.id
        WHERE b.id = :id
        """,
        {"id": booking_id},
    )

    if not booking:
        return {"error": "Booking not found"}

    return booking


# Create a FastAPI app for debugging and local testing
app = R.create_fastapi_app()
