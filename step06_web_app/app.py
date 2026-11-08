"""
STAGE 6: INDUSTRIAL WEB INSPECTION APPLICATION WITH DEFECT LOCALIZATION
-----------------------------------------------------------------------
Flask-powered web interface for bearing defect classification:
1. Provides clean, modern UI featuring Google Font 'Poppins'.
2. Supports Dual Testing Modes:
   - File Upload (Drag & Drop or File Chooser)
   - Real-time Browser Webcam (Live snapshot capture)
   - One-click Sample Presets (from unseen Test set)
3. Connects directly to the Stage 5 Pure CNN Explainable AI (Grad-CAM) engine.
4. Generates side-by-side comparison:
   - Original Inspection Image
   - QC Red Dots Annotated Image (showing exact defect locations)
5. Returns real-time decision, confidence breakdown, and latency (ms).
"""

import io
import sys
import time
import base64
from pathlib import Path
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step05_inference.defect_localizer import DefectLocalizer

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static")
)

# Initialize localizer with best checkpoint
try:
    localizer = DefectLocalizer()
    print("[INFO] Explainable AI Localizer initialized successfully for Web Application.")
except Exception as e:
    localizer = None
    print(f"[WARN] Could not initialize localizer: {e}")


def pil_to_base64(pil_img, format="JPEG"):
    """
    Encodes a PIL image into a Base64 data URL string for web rendering.
    """
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/")
def index():
    """
    Renders main inspection dashboard.
    """
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts image file or base64 data URL, runs Pure CNN inference + Grad-CAM localization,
    and returns dual images (original + red dot annotated) along with QC telemetry.
    """
    if localizer is None:
        return jsonify({"error": "Model checkpoint is not loaded. Train the model first."}), 500

    pil_img = None

    # Handle standard multipart file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            try:
                pil_img = Image.open(file.stream).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"Invalid image file format: {str(e)}"}), 400

    # Handle base64 encoded image (e.g. from browser webcam capture)
    elif request.is_json and "image_base64" in request.json:
        data_url = request.json["image_base64"]
        try:
            if "," in data_url:
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            img_bytes = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Failed to decode base64 image: {str(e)}"}), 400

    if pil_img is None:
        return jsonify({"error": "No image provided. Please upload an image or use webcam."}), 400

    # Sensitivity configuration (lower = catches more fine/subtle defects)
    sensitivity = 0.25
    if "sensitivity" in request.form:
        try:
            sensitivity = float(request.form["sensitivity"])
        except ValueError:
            pass
    elif request.is_json and "sensitivity" in request.json:
        try:
            sensitivity = float(request.json["sensitivity"])
        except ValueError:
            pass

    # Run inference + defect localization
    start_time = time.perf_counter()
    result = localizer.inspect_and_mark(pil_img, sensitivity=sensitivity)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    orig_b64 = pil_to_base64(result["original_image"])
    annotated_b64 = pil_to_base64(result["annotated_image"])

    # Encode all multi-layer metrology views
    views_b64 = {
        name: pil_to_base64(img) for name, img in result.get("views", {}).items()
    }

    return jsonify({
        "status": "success",
        "decision": result["decision"],
        "is_defect": result["is_defect"],
        "confidence": round(result["confidence"], 2),
        "predicted_class": result["predicted_class"],
        "latency_ms": round(latency_ms, 2),
        "num_defect_markers": result["num_defect_markers"],
        "original_image": orig_b64,
        "annotated_image": annotated_b64,
        "views": views_b64,
        "metrics": result.get("metrics", {}),
        "probabilities": {
            k: round(v, 2) for k, v in result["probabilities"].items()
        }
    })


@app.route("/api/sample/<sample_type>")
def get_sample(sample_type):
    """
    Returns a sample image from the test set for quick testing.
    sample_type: 'defect' or 'normal'
    """
    test_dir = PROJECT_ROOT / "dataset_split" / "test"
    target_folder = "defect_ring" if sample_type == "defect" else "normal_ring"
    folder_path = test_dir / target_folder

    if not folder_path.exists():
        return jsonify({"error": "Sample folder not found"}), 404

    samples = list(folder_path.glob("*.jpg"))
    if not samples:
        return jsonify({"error": "No samples available"}), 404

    return send_file(str(samples[0]), mimetype="image/jpeg")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   BEARING DEFECT INSPECTION WEB APPLICATION RUNNING")
    print("=" * 60)
    print("   URL: http://127.0.0.1:5000")
    print("   Press CTRL+C in terminal to stop the server.")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
