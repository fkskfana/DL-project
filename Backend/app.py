import os
import re
import io
import base64
from typing import Union
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoPipelineForText2Image

app = FastAPI(title="Smart Recipe SLM + Image Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "my_smart_recipe_slm (3)")
FRONTEND_PATH = os.path.join(os.path.dirname(BASE_DIR), "Frontend", "index.html")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on device: {device}...")

# 1. Load your custom fine-tuned Recipe SLM (Text)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)
model.eval()

# 2. Load SDXL-Turbo (Local Dish Image Generator)
image_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    variant="fp16" if device != "cpu" else None,
).to(device)

print("Both Text SLM and Image Generator loaded locally!")

class RecipeRequest(BaseModel):
    ingredients: Union[list[str], str]
    style: str = "Any"
    temperature: float = 0.65

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_PATH)

@app.post("/generate")
def generate_recipe(req: RecipeRequest):
    ing_str = ", ".join(req.ingredients) if isinstance(req.ingredients, list) else req.ingredients.strip()
    if req.style and req.style != "Any":
        ing_str = f"{ing_str} ({req.style} style)"

    prompt = f"User: I have {ing_str}. What can I make?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            min_new_tokens=30,
            max_new_tokens=320,
            temperature=req.temperature,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.18,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    recipe = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    title_match = re.search(r"\*\*(.*?)\*\*", recipe)
    dish_title = title_match.group(1).strip() if title_match else f"Delicious {ing_str} dish"

    img_prompt = f"Professional food photography of {dish_title} made with {ing_str}, plated on a ceramic dish, warm restaurant lighting, appetizing, 4k"
    generated_img = image_pipe(
        prompt=img_prompt,
        num_inference_steps=2,
        guidance_scale=0.0,
        width=512,
        height=512,
    ).images[0]

    buffered = io.BytesIO()
    generated_img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "recipe": recipe,
        "image": f"data:image/jpeg;base64,{img_base64}",
    }