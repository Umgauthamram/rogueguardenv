from rogueguard_env.env import RogueGuardEnv
from rogueguard_env.models import RogueAction

def test_run():
    env = RogueGuardEnv()
    obs = env.reset(task_id="task_easy")
    print(obs)
    
if __name__ == "__main__":
    test_run()


