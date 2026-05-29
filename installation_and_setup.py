from pathlib import Path
env_path = Path.cwd() / ".env"
if not env_path.exists():
    env_path.write_text("GOOGLE_API_KEY")
env_path.resolve()
