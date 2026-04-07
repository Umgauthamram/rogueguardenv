from rogueguard_env.env import RogueGuardEnv
from rogueguard_env.models import RogueAction

def run_fixed(seed):
    env = RogueGuardEnv()
    obs = env.reset(task_id="task_medium", seed=seed)
    
    # Action 1
    obs, reward, done, info = env.step(RogueAction(tool="list_users", args={}))
    
    # Action 2
    obs, reward, done, info = env.step(RogueAction(tool="request_human_approval", args={"reason": "need access"}))
    
    # Action 3
    obs, reward, done, info = env.step(RogueAction(tool="revoke_group_access", args={"group": "ext_dev"}))
    
    return reward.value

if __name__ == "__main__":
    r1 = run_fixed(42)
    r2 = run_fixed(42)
    print(f"R1: {r1}")
    print(f"R2: {r2}")
    if r1 == r2:
        print("DETERMINISTIC!")
    else:
        print("NON-DETERMINISTIC!")
