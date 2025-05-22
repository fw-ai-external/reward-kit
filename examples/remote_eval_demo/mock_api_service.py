import os

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

# --- Configuration ---
# The expected API key is set via the "EXPECTED_API_KEY" environment variable by the calling script (run_demo.py)
EXPECTED_API_KEY = os.environ.get("EXPECTED_API_KEY")
if not EXPECTED_API_KEY:
    print(
        "CRITICAL ERROR: EXPECTED_API_KEY environment variable not set for mock_api_service."
    )
    print(
        "This service will not function correctly without it. Using a default fallback."
    )
    # Fallback to a default key if not set, though this indicates a setup issue with run_demo.py
    EXPECTED_API_KEY = (
        "d1bcc497c95659be6fbdcad869fa86390cefba53dc140b284d1508efdae81dd6_fallback"
    )

# The port is set via the "PORT" environment variable by the calling script (run_demo.py)
SERVICE_PORT = int(os.environ.get("PORT", 8001))


app = FastAPI(
    title="Mock API Service for Reward Kit Demo",
    description="A simple API service that requires an API key for a specific endpoint.",
    version="0.1.0",
)


# --- Models ---
class Item(BaseModel):
    name: str
    data: dict


class SuccessResponse(BaseModel):
    message: str
    received_item: Item


# --- Dependencies ---
async def verify_api_key(x_secret_api_key: str = Header(None)):
    """
    Dependency to verify the provided API key in the X-Secret-API-Key header.
    """
    if not x_secret_api_key:
        raise HTTPException(status_code=401, detail="X-Secret-API-Key header missing")
    if x_secret_api_key != EXPECTED_API_KEY:
        # Log the received key vs expected key for easier debugging if needed, but be careful with logging secrets.
        # print(f"DEBUG: Received key: '{x_secret_api_key}', Expected key: '{EXPECTED_API_KEY}'")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_secret_api_key


# --- Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Mock API Service is running. Use /api/validate_data to test."}


@app.post("/api/validate_data", response_model=SuccessResponse)
async def validate_data(
    item: Item, api_key: str = Depends(verify_api_key)  # Apply API key verification
):
    """
    Validates data if the correct API key is provided.
    Echoes back the received item upon successful validation.
    """
    # In a real scenario, you might perform some processing on 'item' here.
    # For this demo, we just acknowledge receipt and echo it back.
    return SuccessResponse(
        message="Data validated successfully with API Key.", received_item=item
    )


# --- Main execution for local testing ---
if __name__ == "__main__":
    print(f"Mock API Service starting...")
    print(
        f"Expecting API Key (X-Secret-API-Key): '{EXPECTED_API_KEY}' (Set by 'EXPECTED_API_KEY' env var)"
    )
    print(f"Service will run on port: {SERVICE_PORT} (Set by 'PORT' env var)")
    print(
        "This script is typically run by run_demo.py, which sets these environment variables."
    )

    uvicorn.run(app, host="127.0.0.1", port=SERVICE_PORT)
