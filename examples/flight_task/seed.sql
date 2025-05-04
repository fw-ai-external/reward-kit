-- Initial database setup for flight booking task

-- Create tables
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY,
    airline TEXT NOT NULL,
    origin TEXT NOT NULL,
    dest TEXT NOT NULL,
    depart TEXT NOT NULL,  -- ISO8601 datetime
    arrive TEXT NOT NULL,  -- ISO8601 datetime
    price REAL NOT NULL,
    seats_available INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    id TEXT PRIMARY KEY,
    flight_id INTEGER NOT NULL,
    passenger TEXT NOT NULL,
    status TEXT NOT NULL,  -- reserved, paid, cancelled
    price REAL NOT NULL,
    payment_method TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (flight_id) REFERENCES flights(id)
);

CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    booking_id TEXT NOT NULL,
    amount REAL NOT NULL,
    method TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (booking_id) REFERENCES bookings(id)
);

-- Create a table to log tool calls for tracking agent behavior
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    parameters TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample flights
-- SFO to JFK
INSERT INTO flights (airline, origin, dest, depart, arrive, price, seats_available)
VALUES 
    ('United', 'SFO', 'JFK', '2023-09-15T08:00:00', '2023-09-15T16:30:00', 299.99, 5),
    ('Delta', 'SFO', 'JFK', '2023-09-15T10:15:00', '2023-09-15T18:45:00', 349.99, 3),
    ('American', 'SFO', 'JFK', '2023-09-15T13:30:00', '2023-09-15T22:00:00', 279.99, 7),
    ('JetBlue', 'SFO', 'JFK', '2023-09-15T16:45:00', '2023-09-16T01:15:00', 259.99, 2);

-- SFO to LAX
INSERT INTO flights (airline, origin, dest, depart, arrive, price, seats_available)
VALUES 
    ('Southwest', 'SFO', 'LAX', '2023-09-15T07:30:00', '2023-09-15T09:15:00', 129.99, 12),
    ('United', 'SFO', 'LAX', '2023-09-15T11:00:00', '2023-09-15T12:45:00', 149.99, 8),
    ('Delta', 'SFO', 'LAX', '2023-09-15T14:30:00', '2023-09-15T16:15:00', 139.99, 5),
    ('American', 'SFO', 'LAX', '2023-09-15T18:00:00', '2023-09-15T19:45:00', 159.99, 10);

-- JFK to SFO
INSERT INTO flights (airline, origin, dest, depart, arrive, price, seats_available)
VALUES 
    ('United', 'JFK', 'SFO', '2023-09-16T09:00:00', '2023-09-16T12:30:00', 319.99, 4),
    ('Delta', 'JFK', 'SFO', '2023-09-16T13:15:00', '2023-09-16T16:45:00', 329.99, 6),
    ('American', 'JFK', 'SFO', '2023-09-16T17:30:00', '2023-09-16T21:00:00', 299.99, 9);

-- Next day flights: SFO to JFK for tomorrow
INSERT INTO flights (airline, origin, dest, depart, arrive, price, seats_available)
VALUES 
    ('United', 'SFO', 'JFK', '2023-09-16T07:00:00', '2023-09-16T15:30:00', 319.99, 8),
    ('Delta', 'SFO', 'JFK', '2023-09-16T09:15:00', '2023-09-16T17:45:00', 369.99, 5),
    ('American', 'SFO', 'JFK', '2023-09-16T12:30:00', '2023-09-16T21:00:00', 299.99, 10),
    ('JetBlue', 'SFO', 'JFK', '2023-09-16T15:45:00', '2023-09-17T00:15:00', 279.99, 4);