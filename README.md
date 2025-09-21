# Autism Early Screening – Flask + TensorFlow

A professional landing page for early autism screening (binary image classification) with a Flask backend that loads a TensorFlow/Keras model and performs predictions on uploaded images.

## Features
- Clean, responsive landing page (Bootstrap-like look via custom CSS).
- Upload an image, run inference, and display the result (label + confidence) on a result page.
- Graceful error messages when the model is missing.
- Health endpoint for simple liveness checks.

## Project Structure
```
autism_flask_app/
├─ app.py
├─ requirements.txt
├─ models/
│  └─ convnext_tiny_faag_binary.keras        # <-- place your trained model here. Download Model: https://huggingface.co/spaces/darkzane007/autimsclass
├─ uploads/                  # runtime-uploaded files
├─ static/
│  ├─ styles.css
│  └─ logo.svg
└─ templates/
   ├─ base.html
   ├─ index.html
   └─ predict.html
```

## Getting Started

1. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model**
   - Export your trained Keras model as `autism_model.h5` and put it in `models/`.

4. **Run the app**
   ```bash
   export FLASK_SECRET_KEY="change-me"  # Windows PowerShell: $env:FLASK_SECRET_KEY="change-me"
   python app.py
   ```
   Visit: http://127.0.0.1:5000

## Model Assumptions
- Input size defaults to 224×224×3, scaled to [0,1].
- Output is either:
  - Sigmoid (1 unit): probability of **Autistic** (class 1).
  - Softmax (2 units): `[Non_Autistic, Autistic]` probabilities.

Edit `IMAGE_SIZE`, `CLASS_NAMES`, and `THRESHOLD` in `app.py` to match your model.

## Notes
- This project **does not** ship a trained model due to size and licensing.
- For production, consider:
  - Running behind a reverse proxy (Nginx).
  - Disabling `debug=True`.
  - File size and type validation (already present at basic level).
  - Request rate limiting and CSRF protection if you add forms elsewhere.
