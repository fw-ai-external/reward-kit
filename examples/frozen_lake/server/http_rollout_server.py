"""
HTTP rollout server for Frozen Lake game.

This server implements the standard HTTP rollout protocol using the reward-kit
library's standardized types for consistent client/server communication.
"""

import uuid
import sys
import os
from typing import Dict
from fastapi import FastAPI, HTTPException

# Add parent directory to path to import frozen_lake_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frozen_lake_server import FrozenLakeGame

# Import standardized HTTP rollout protocol types from reward-kit
from reward_kit.agent.resources import (
    StartEpisodeResponse,
    StepRequest,
    StepResponse,
    EndEpisodeRequest,
    EndEpisodeResponse,
    HealthResponse,
)

# Additional model for initial prompt
from pydantic import BaseModel

class InitialPromptResponse(BaseModel):
    """Response containing the initial prompt and game state for the agent."""
    content: str
    visual_state: str
    game_rules: str


# FastAPI app
app = FastAPI(title="Frozen Lake HTTP Rollout Server")

# Store active episodes
episodes: Dict[str, FrozenLakeGame] = {}


@app.get("/initial_prompt", response_model=InitialPromptResponse)
async def get_initial_prompt() -> InitialPromptResponse:
    """Get the initial prompt and game setup for the agent."""
    content = """🎮 FROZEN LAKE GAME - AUTONOMOUS PLAY MODE

🎯 OBJECTIVE: Navigate from S to G without hitting H

📋 GAME RULES: S=start, F=safe, H=hole(death), G=goal(win)

🤖 AUTONOMOUS MODE INSTRUCTIONS:
- You are playing this game AUTONOMOUSLY until completion
- KEEP MAKING MOVES using the step tool until you reach G or hit H
- DO NOT ask for user input or wait for confirmation
- DO NOT stop after one move - continue until the game ends
- Each move should be followed immediately by another move
- Game only ends when you reach G (win) or hit H (lose)

🎮 ACTION: Use step tool with: "left", "right", "up", or "down"

⚡ START NOW - Make your first move and continue until the game is complete!"""
    
    visual_state = """[S] F  F  F 
 F  H  F  H 
 F  F  F  H 
 H  F  F  G """
    
    game_rules = """Game Rules:
- S = Start position
- F = Frozen (safe to step on)
- H = Hole (game over if you step here)
- G = Goal (reach this to win)
- [X] = Your current position"""
    
    return InitialPromptResponse(
        content=content,
        visual_state=visual_state,
        game_rules=game_rules
    )


@app.post("/start_episode", response_model=StartEpisodeResponse)
async def start_episode() -> StartEpisodeResponse:
    """Start a new episode of the Frozen Lake game."""
    episode_id = str(uuid.uuid4())
    game = FrozenLakeGame()
    observation = game.reset()
    episodes[episode_id] = game
    
    return StartEpisodeResponse(
        episode_id=episode_id,
        observation=observation
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    """Take a step in the specified episode."""
    if req.episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    game = episodes[req.episode_id]
    observation, is_done = game.step(req.action)
    
    return StepResponse(
        observation=observation,
        is_done=is_done
    )


@app.post("/end_episode", response_model=EndEpisodeResponse)
async def end_episode(req: EndEpisodeRequest) -> EndEpisodeResponse:
    """End the specified episode."""
    if req.episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    del episodes[req.episode_id]
    return EndEpisodeResponse(message=f"Episode {req.episode_id} ended successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", 
        game="frozen_lake"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)