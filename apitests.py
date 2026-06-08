from dotenv import load_dotenv

load_dotenv()

import os

print(os.getenv("GOOGLE_API_KEY"))

import os

print("API Key exists:", bool(os.environ.get("GOOGLE_API_KEY")))