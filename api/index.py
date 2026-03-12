from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import tempfile
import torch

# Ensure the parent directory is in the path to import markov
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markov.gridworld import MazeMDP, RobbingBanksMDP, value_iteration, q_learning, sarsa
from markov.mountain_car import train_mountain_car
from markov.lunar_lander import train_dqn, train_ppo

app = FastAPI(title="Markov RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

class TrainRequest(BaseModel):
    episodes: int = 10
    max_steps: int = 200

@app.post("/api/train/gridworld/maze")
async def train_maze():
    maze_grid = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    mdp = MazeMDP(maze_grid, minotaur_start=(0, 4), player_start=(0, 0))
    V, policy = value_iteration(mdp)

    # Format policy for frontend visualization
    policy_list = [{"state": str(s), "value": V[s], "action": int(policy[s])} for s in policy.keys()]

    return {"status": "success", "policy": policy_list[:100]} # return subset to avoid huge response

@app.post("/api/train/mountain_car")
async def api_train_mountain_car(req: TrainRequest):
    agent, rewards = train_mountain_car(episodes=req.episodes, max_steps=req.max_steps)
    return {
        "status": "success",
        "rewards": rewards
    }

@app.post("/api/train/lunar_lander/dqn")
async def api_train_dqn(req: TrainRequest):
    agent, rewards = train_dqn(episodes=req.episodes, max_steps=req.max_steps, batch_size=32)
    return {
        "status": "success",
        "rewards": rewards,
        "algorithm": "DQN"
    }

@app.post("/api/train/lunar_lander/ppo")
async def api_train_ppo(req: TrainRequest):
    agent, rewards = train_ppo(episodes=req.episodes, max_steps=req.max_steps)
    return {
        "status": "success",
        "rewards": rewards,
        "algorithm": "PPO"
    }

@app.get("/api/models/{model_name}")
async def get_model(model_name: str):
    # Dummy endpoint to simulate serving a pre-trained model
    if model_name not in ["dqn.pth", "ddpg.pth", "ppo.pth"]:
        return {"error": "Model not found"}, 404

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    # Write a dummy tensor
    torch.save({"dummy": torch.tensor([1, 2, 3])}, temp_file.name)

    return FileResponse(temp_file.name, media_type="application/octet-stream", filename=model_name)
