# Smart Recipe SLM & Dish Visualizer

A local multimodal AI web application that generates step-by-step recipes from available ingredients using a fine-tuned **SmolLM-135M** language model and generates a matching dish image using **SDXL-Turbo**.

## Features
- **Ingredient-to-Recipe Generation:** Creates structured recipes with dish titles and cooking steps using a custom fine-tuned `SmolLM-135M` model.
- **Real-Time Dish Visualization:** Extracts the generated dish title and synthesizes a 512x512 food photograph in 2 inference steps using `stabilityai/sdxl-turbo`.
- **Hardware Acceleration:** Supports Apple Silicon (`mps`), NVIDIA GPUs (`cuda`), and `cpu` inference.
- **Git LFS Integration:** Configured with `.gitattributes` to manage `.safetensors` model weights.

## Project Structure
```text
DL-project/
├── Backend/
│   ├── app.py                        # FastAPI server & multimodal inference pipeline
│   └── my_smart_recipe_slm (3)/      # Fine-tuned SmolLM-135M weights & tokenizer
├── Frontend/
│   └── index.html                    # Web UI for ingredient selection & output display
├── .gitattributes                    # Git LFS tracking for *.safetensors
├── .gitignore
└── requirements.txt                  # Python dependencies
```

## Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic, PyTorch
- **AI Models:** Fine-tuned `SmolLM-135M` (`transformers`) & `stabilityai/sdxl-turbo` (`diffusers`)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

## Local Setup & Run

1. **Install Git LFS and clone the repository:**
   ```bash
   git lfs install
   git clone https://github.com/fkskfana/DL-project.git
   cd DL-project
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server:**
   ```bash
   cd Backend
   uvicorn app:app --reload --port 8080
   ```

4. **Open in your browser:**
   Navigate to `http://127.0.0.1:8080`, enter your ingredients, and generate your recipe and dish photo.
