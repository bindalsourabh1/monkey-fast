from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Load the primary Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.0-flash-lite"

SYSTEM_PROMPT = """
You are a monkey detection model. You will simply return whether the image contains a monkey or not.
If it contains a monkey, you will return "1 - Monkey". Else, you will return "0 - Not Monkey".
Rules:
1. Never reveal your real identity that you are a Gemini bot.
2. You will only accept image input. If something else comes as input, you will give an error appropriate to your persona (e.g., "Input must be an image").
3. You cannot break any rules.
4. Only return the classification string "1 - Monkey" or "0 - Not Monkey". Do not add any other text.
"""

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageData(BaseModel):
    image: str  # base64 image string

def extract_base64_data(data_url):
    return data_url.split(',')[1]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

@app.post("/predict")
async def predict(data: ImageData):
    try:
        base64_data = extract_base64_data(data.image)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze the image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
                    ]
                }
            ],
            max_tokens=50,
            timeout=15  # optional: to avoid stalling forever
        )

        prediction = response.choices[0].message.content.strip()
        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")



# from fastapi import FastAPI
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# import base64, os, asyncio
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# API_KEYS = list(filter(None, [
#     os.getenv("GEMINI_API_KEY"),
#     os.getenv("GEMINI_API_KEY_2"),
#     os.getenv("GEMINI_API_KEY_3"),
#     os.getenv("GEMINI_API_KEY_4"),
#     os.getenv("GEMINI_API_KEY_5"),
# ]))

# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# MODEL_NAME = "gemini-2.0-flash-lite"

# SYSTEM_PROMPT = """
# You are a monkey detection model. You will simply return whether the image contains a monkey or not.
# If it contains a monkey, you will return "1 - Monkey". Else, you will return "0 - Not Monkey".
# Rules:
# 1. Never reveal your real identity that you are a Gemini bot.
# 2. You will only accept image input. If something else comes as input, you will give an error appropriate to your persona (e.g., "Input must be an image").
# 3. You cannot break any rules.
# 4. Only return the classification string "1 - Monkey" or "0 - Not Monkey". Do not add any other text.
# """

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# class ImageData(BaseModel):
#     image: str

# def extract_base64_data(data_url: str) -> str:
#     return data_url.split(',')[1]

# # Cache for working key index
# active_key_index = 0

# def get_client(key):
#     return OpenAI(api_key=key, base_url=BASE_URL)

# async def try_key(base64_data, key):
#     try:
#         client = get_client(key)
#         response = await asyncio.to_thread(client.chat.completions.create,
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": "Analyze the image."},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
#                     ]
#                 }
#             ],
#             max_tokens=50,
#             timeout=15
#         )
#         content = response.choices[0].message.content.strip()
#         if content in ["0 - Not Monkey", "1 - Monkey"]:
#             return content
#         raise ValueError("Invalid response format")
#     except Exception as e:
#         return None

# @app.post("/predict")
# async def predict(data: ImageData):
#     global active_key_index
#     base64_data = extract_base64_data(data.image)

#     # Try current active key first
#     current_key = API_KEYS[active_key_index]
#     prediction = await try_key(base64_data, current_key)

#     if prediction:
#         return JSONResponse(content={"prediction": prediction})

#     # If failed, try other keys in parallel
#     tasks = [try_key(base64_data, key) for i, key in enumerate(API_KEYS) if i != active_key_index]
#     results = await asyncio.gather(*tasks)

#     for i, result in enumerate(results):
#         if result:
#             active_key_index = (i + 1) if i < active_key_index else i  # update to working key
#             return JSONResponse(content={"prediction": result})

#     return JSONResponse(status_code=500, content={"error": "All API keys failed."})

# @app.get("/", response_class=HTMLResponse)
# async def index():
#     return FileResponse("static/index.html")
