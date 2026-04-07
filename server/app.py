import uvicorn
import os
from rogueguard_env.server import app

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
