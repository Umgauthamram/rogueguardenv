from openenv.core.env_server import create_app
from server.env import RogueGuardEnv
from server.models import RogueAction, RogueObservation
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

app = create_app(
    RogueGuardEnv,
    RogueAction,
    RogueObservation
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "env": "rogueguard-env", "version": "0.1.0"})

@app.get("/")
async def root():
    return JSONResponse({
        "name": "RogueGuardEnv",
        "description": "RL environment that trains agents to stop themselves going rogue",
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "docs": "/docs"
    })
