"""
Frozen Lake game server implementation.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class FrozenLakeGame:
    """
    Simple Frozen Lake game implementation.
    
    The game is played on a 4x4 grid where:
    - S: Starting position
    - F: Frozen surface (safe to walk on)
    - H: Hole (game over if you fall in)
    - G: Goal (reach this to win)
    
    Actions:
    - 0: Left
    - 1: Down  
    - 2: Right
    - 3: Up
    """
    
    def __init__(self, map_name: str = "4x4"):
        """Initialize the Frozen Lake game."""
        if map_name == "4x4":
            self.desc = np.asarray([
                "SFFF",
                "FHFH", 
                "FFFH",
                "HFFG"
            ], dtype='c')
        else:
            raise ValueError(f"Unknown map name: {map_name}")
            
        self.nrow, self.ncol = len(self.desc), len(self.desc[0])
        self.nS = self.nrow * self.ncol
        self.nA = 4
        
        # Find start and goal positions
        self.start_pos = None
        self.goal_pos = None
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i, j] == b'S':
                    self.start_pos = (i, j)
                elif self.desc[i, j] == b'G':
                    self.goal_pos = (i, j)
        
        self.reset()
    
    def _to_s(self, row: int, col: int) -> int:
        """Convert (row, col) to state number."""
        return row * self.ncol + col
    
    def _from_s(self, s: int) -> Tuple[int, int]:
        """Convert state number to (row, col)."""
        return s // self.ncol, s % self.ncol
    
    def reset(self) -> Dict:
        """Reset the game to the starting position."""
        self.current_pos = self.start_pos
        self.done = False
        self.won = False
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict, bool]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=left, 1=down, 2=right, 3=up)
            
        Returns:
            Tuple of (observation, done)
        """
        if self.done:
            return self._get_observation(), True
        
        row, col = self.current_pos
        
        # Apply action
        if action == 0:  # Left
            col = max(col - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # Right
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # Up
            row = max(row - 1, 0)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        self.current_pos = (row, col)
        
        # Check what's at the new position
        cell = self.desc[row, col]
        
        if cell == b'H':  # Hole
            self.done = True
            self.won = False
        elif cell == b'G':  # Goal
            self.done = True
            self.won = True
        
        return self._get_observation(), self.done
    
    def _get_observation(self) -> Dict:
        """Get the current observation."""
        row, col = self.current_pos
        cell = self.desc[row, col].decode('utf-8')
        
        # Create a visual representation
        visual = []
        for i in range(self.nrow):
            row_str = ""
            for j in range(self.ncol):
                if (i, j) == self.current_pos:
                    row_str += "[" + self.desc[i, j].decode('utf-8') + "]"
                else:
                    row_str += " " + self.desc[i, j].decode('utf-8') + " "
            visual.append(row_str)
        
        obs = {
            "position": self.current_pos,
            "current_cell": cell,
            "done": self.done,
            "won": self.won,
            "visual": "\n".join(visual),
            "message": self._get_message()
        }
        
        return obs
    
    def _get_message(self) -> str:
        """Get a descriptive message about the current state."""
        if self.done:
            if self.won:
                return "Congratulations! You reached the goal! You win!"
            else:
                return "Oh no! You fell into a hole. Game over."
        else:
            row, col = self.current_pos
            cell = self.desc[row, col].decode('utf-8')
            return f"You are at position ({row}, {col}) on a {cell} cell. Choose your next move carefully."