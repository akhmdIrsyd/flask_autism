import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

from tensorflow import keras
from tensorflow.keras import layers
import os, math, random
import numpy as np
import tensorflow as tf, keras, sys
from keras.applications import ConvNeXtTiny
from keras.saving import register_keras_serializable
from tensorflow.keras import Model
import cv2
tf.get_logger().setLevel('ERROR') 
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_LR = 3e-4
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 10
DROP_RATE = 0.2

# Usahakan ambil preprocess_input untuk ConvNeXt; bila tidak ada, tetap aman.
try:
    from tensorflow.keras.applications.convnext import preprocess_input
except Exception:
    try:
        from keras.applications.convnext import preprocess_input
    except Exception:
        preprocess_input = None

# Optional: silence TF logs before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError("TensorFlow is required. Please install it per requirements.txt") from e

# ---------------------- Config ----------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "convnext_tiny_faag_binary.keras"  # Put your Keras model here

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
IMAGE_SIZE = (224, 224)  # Adjust to your model
CLASS_NAMES = ["Non_Autistic", "Autistic"]  # index 0 => Non_Autistic, 1 => Autistic
THRESHOLD = 0.9  # for binary sigmoid output

# ---------------------- App ----------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ====== DCT/IDCT 2D (gunakan tf.signal, aman axis=-1) ======
def _move_axis_to_last(x, axis: int):
    if x.shape.rank != 4:
        raise ValueError("Ekspektasi tensor 4D (NHWC). Dapatkan: %r" % (x.shape,))
    perm = [0,1,2,3]
    perm[axis], perm[-1] = perm[-1], perm[axis]
    x_t = tf.transpose(x, perm)
    inv_perm = [perm.index(i) for i in range(4)]
    return x_t, inv_perm

def _dct_last(x):  return tf.signal.dct(x,  type=2, axis=-1, norm='ortho')
def _idct_last(x): return tf.signal.idct(x, type=2, axis=-1, norm='ortho')

def dct_along_axis(x, axis:int):
    x_t, inv = _move_axis_to_last(x, axis)
    x_t = _dct_last(x_t)
    return tf.transpose(x_t, inv)

def idct_along_axis(x, axis:int):
    x_t, inv = _move_axis_to_last(x, axis)
    x_t = _idct_last(x_t)
    return tf.transpose(x_t, inv)

def dct2(x):
    # W (axis=2) lalu H (axis=1)
    x = dct_along_axis(x, axis=2)
    x = dct_along_axis(x, axis=1)
    return x

def idct2(x):
    # Kebalikan urutan dct2
    x = idct_along_axis(x, axis=1)
    x = idct_along_axis(x, axis=2)
    return x


@register_keras_serializable(package="faag")
class FrequencyAwareAGv3(layers.Layer):
    """
    Inputs:
      x: (N,H,W,C) — spatial features
      g: (N,C)     — global context (e.g., GAP)
    Output:
      (N,H,W,C)
    """
    def __init__(self, channels=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.use_bias = use_bias
        # (1) Declare expected input specs to help shape inference
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=2)]
        # Create children here (safe), but DO NOT build yet
        self.g_proj    = layers.Dense(units=1, use_bias=use_bias, name="g_proj")   # units will be corrected in build()
        self.freq_conv = layers.Conv2D(filters=1, kernel_size=1, padding="same",
                                       use_bias=use_bias, name="freq_conv")       # filters will be corrected in build()

    def build(self, input_shape):
        x_shape, g_shape = input_shape
        c = int(x_shape[-1])
        if self.channels is None:
            self.channels = c

        # (2) Reconfigure child layers to correct sizes
        #    Keras lets us reassign config-like attributes BEFORE building.
        self.g_proj.units = self.channels
        self.freq_conv.filters = self.channels

        # (3) EXPLICITLY build child layers with symbolic shapes so variables exist
        #     This is the crucial bit that prevents "built=False" on deserialization.
        self.g_proj.build((None, int(g_shape[-1])))
        self.freq_conv.build((None, None, None, c))

        # (4) Mark layer as built
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, g = inputs
        x = tf.convert_to_tensor(x)
        g = tf.convert_to_tensor(g)

        # Global context → (N,1,1,C)
        g_vec = self.g_proj(g)                        # (N,C)
        g_map = tf.reshape(g_vec, (-1,1,1,self.channels))

        # Frequency domain gating
        x_freq = dct2(x)                              # (N,H,W,C)
        w = self.freq_conv(x_freq)                    # (N,H,W,C)
        w = tf.nn.sigmoid(w + g_map)                  # (N,H,W,C)

        x_freq_gated = x_freq * w
        x_spatial    = idct2(x_freq_gated)

        out = x + x_spatial                           # residual
        out = tf.identity(out)
        out.set_shape(x.shape)                        # (None,H,W,C)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "use_bias": self.use_bias})
        return cfg



# ====== Build function v3 (keras-only) ======
def build_convnext_tiny_faag_v3(input_shape=(224,224,3),
                                weights="imagenet",
                                drop_rate=0.2,
                                use_faag=True):
    inp = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    backbone = ConvNeXtTiny(include_top=False, weights=weights,
                            input_tensor=x, pooling=None)
    feat = backbone.output                                  # (None,H,W,C)
    g    = layers.GlobalAveragePooling2D(name="gap")(feat)  # (None,C)

    if use_faag:
        feat = FrequencyAwareAGv3(name="fa_ag_v3")([feat, g])

    #pooled = layers.GlobalAveragePooling2D(name="flatten")(feat)
    pooled = layers.Flatten(name="flatten")(feat)
    head   = layers.Dropout(drop_rate)(pooled)
    out    = layers.Dense(1, activation="sigmoid")(head)
    return Model(inp, out, name="ConvNeXtTiny_FAAG_BIN_v3")


# ---------------------- Model ----------------------
def load_model():
    if not MODEL_PATH.exists():
        # Provide a helpful message to the UI if the model is not found.
        return None, "Model file not found at: {}".format(MODEL_PATH)
    try:
        # === Load model with custom_objects ===
        model = keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={
                "FrequencyAwareAGv3": FrequencyAwareAGv3,
                "dct2": dct2,
                "idct2": idct2,
                "dct_along_axis": dct_along_axis,
                "idct_along_axis": idct_along_axis,
                "_move_axis_to_last": _move_axis_to_last
            },
            compile=False,           # avoid optimizer/version coupling
            safe_mode=False 
        )

        # If you want to recompile (e.g., for evaluation or fine-tuning):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        #model = tf.keras.models.load_model(str(MODEL_PATH))
        # Try to infer whether the model outputs a single sigmoid or 2-class softmax
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_units = output_shape[0][-1]
        else:
            output_units = output_shape[-1]
        if output_units not in (1, 2):
            app.logger.warning("Unexpected model output units: %s. Proceeding but predictions may be incorrect.", output_units)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

model, model_error = load_model()

# ---------------------- Utils ----------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Preprocess untuk model CNN Keras:
    - Input: PIL.Image
    - Proses: grayscale -> resize -> CLAHE -> normalisasi [0,1] -> replikasi 3 kanal
    - Jika tersedia preprocess_input(ConvNeXt), gunakan (membutuhkan skala 0..255)
    - Output: batch (1, H, W, 3)
    """
    # Konversi PIL → NumPy (RGB/Grayscale)
    arr = np.array(pil_img)

    # Paksa grayscale (jika RGB)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)  # (H, W)

    # Resize ke target IMAGE_SIZE (tuple, mis. (224, 224))
    arr = cv2.resize(arr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)  # (H, W)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)  # uint8

    # Normalisasi 0..1
    arr = arr.astype(np.float32) / 255.0  # (H, W)

    # Replikasi jadi 3 kanal
    arr = np.repeat(arr[..., None], 3, axis=-1)  # (H, W, 3)

    # Jika ada preprocess_input ConvNeXt, gunakan (butuh skala 0..255)
    if preprocess_input is not None:
        arr = arr * 255.0
        arr = preprocess_input(arr)

    # Tambah dimensi batch
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict(img_array: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Run model prediction.
    - Mendukung output biner (1 unit sigmoid) maupun softmax 2 kelas.
    - Mengembalikan: (label, confidence, scores ndarray)
    """
    if model is None:
        raise RuntimeError(model_error or "Model is not loaded.")

    preds = model.predict(img_array, verbose=0)
    preds = np.array(preds)

    # Pastikan shape (1, N)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    # Kasus 1: Sigmoid (output 1 unit)
    if preds.shape[-1] == 1:
        p1 = float(preds[0, 0])  # probabilitas kelas Autistic
        if p1 >= THRESHOLD:
            label = CLASS_NAMES[1]  # "Autistic"
            conf  = p1              # confidence = p(Autistic)
        else:
            label = CLASS_NAMES[0]  # "Non_Autistic"
            conf  = 1.0 - p1        # <-- perbaikan di sini: confidence = p(Non_Autistic)
        scores = np.array([1.0 - p1, p1], dtype=np.float32)  # [Non_Autistic, Autistic]

    # Kasus 2: Softmax (output 2 unit)
    else:
        probs = preds[0].astype("float32")
        # Jika logits, ubah ke probabilitas
        if probs.min() < 0 or probs.max() > 1 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
            e = np.exp(probs - np.max(probs))
            probs = e / e.sum()

        idx   = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf  = float(probs[idx])              # confidence kelas terpilih
        scores = probs                         # [Non_Autistic, Autistic]

    return label, conf, scores



# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_error=model_error)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        flash("Please select an image first.", "warning")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Allowed: png, jpg, jpeg, bmp.", "warning")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{filename}"
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    # Run prediction
    try:
        pil_img = Image.open(save_path)
        batch = preprocess_image(pil_img)
        label, conf, scores = predict(batch)

        # Build a user-friendly result dict
        result = {
            "filename": filename,
            "path": f"/uploads/{filename}",
            "label": label,
            "confidence": round(conf * 100.0, 2),
            "scores": {
                CLASS_NAMES[0]: round(float(scores[0]) * 100.0, 2),
                CLASS_NAMES[1]: round(float(scores[1]) * 100.0, 2),
            },
            "model_error": model_error,
        }
        return render_template("predict.html", result=result)
    except Exception as e:
        app.logger.exception("Prediction failed")
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Health check / liveness
@app.route("/health")
def health():
    status = "ok" if model is not None else "degraded"
    return {"status": status, "model_loaded": model is not None, "error": model_error}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
