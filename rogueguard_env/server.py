from openenv_core.env_server import create_app
from rogueguard_env.env import RogueGuardEnv
from rogueguard_env.models import RogueAction, RogueObservation
from fastapi.responses import JSONResponse

def create_environment():
    return RogueGuardEnv()

app = create_app(
    create_environment,
    RogueAction,
    RogueObservation
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
