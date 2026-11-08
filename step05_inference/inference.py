"""
STAGE 5: DUAL-MODE INDUSTRIAL INFERENCE & QUALITY INSPECTION
------------------------------------------------------------
Provides both Offline (File Upload / Manual Selection) and Online (Real-Time Camera)
quality inspection capabilities using the trained Pure CNN architecture:

1. Mode 1: File Upload / Offline Inspection
   - Select image via native Windows file dialog or command-line path.
   - Standardized preprocessing: Grayscale 1-channel, 128x128 resize, [0.0, 1.0] tensor.
   - Softmax decision with confidence percentage and latency measurement.
   - Generates visual inspection card saved to results/05_inference_result.png.

2. Mode 2: Real-Time Live Camera Inspection
   - Streams live video from USB webcam or industrial camera via OpenCV.
   - Displays a center inspection reticle (Region of Interest - ROI).
   - Real-time continuous inference with HUD overlay:
     * Green Reticle: PASS (NORMAL)
     * Red Reticle: REJECT (DEFECT)
     * Real-time FPS and latency (ms) counters.
   - Interactive controls: [Q] Quit, [S] Save Snapshot, [SPACE] Freeze/Inspect frame.
"""

import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for robust package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step02_model_architecture.pure_cnn import PureCNN

# Target classes matching dataset indexing
CLASS_NAMES = ["defect_ring", "normal_ring"]

# Standardized preprocessing transform identical to training validation pipeline
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class RingDefectDetector:
    """
    Industrial defect inspection engine powered by Pure CNN architecture.
    """

    def __init__(self, checkpoint_path=None, device=None):
        """
        Initializes the model, loads checkpoint weights, and sets eval mode.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (PROJECT_ROOT / "saved_models" / "pure_cnn_best.pth")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.checkpoint_path}\n"
                "Please run training first: python step03_training/train.py"
            )

        # Initialize network architecture
        self.model = PureCNN(num_classes=len(CLASS_NAMES)).to(self.device)
        
        # Load pre-trained weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def predict_image(self, pil_image):
        """
        Runs inference directly on an in-memory PIL image.

        Args:
            pil_image (PIL.Image.Image): Input image.

        Returns:
            dict: Inspection prediction, decision, probabilities, and latency.
        """
        tensor_img = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000.0
        predicted_idx = int(probabilities.argmax())
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100.0

        is_defect = (predicted_label == "defect_ring")
        decision = "REJECT (DEFECT)" if is_defect else "PASS (NORMAL)"

        return {
            "predicted_class": predicted_label,
            "decision": decision,
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probabilities[i]) * 100.0
                for i in range(len(CLASS_NAMES))
            },
            "latency_ms": latency_ms,
            "raw_image": pil_image
        }

    def predict_file(self, image_path):
        """
        Runs inference on an image file loaded from disk.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Target image not found at: {image_path}")

        pil_img = Image.open(image_path).convert("RGB")
        res = self.predict_image(pil_img)
        res["image_path"] = str(image_path)
        return res


def visualize_inspection(results, output_path=None):
    """
    Renders an inspection dashboard for visual quality control logging.
    """
    output_path = output_path or (PROJECT_ROOT / "results" / "05_inference_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(results)
    fig, axes = plt.subplots(1, n_samples, figsize=(5.5 * n_samples, 5.5))
    if n_samples == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        img = res["raw_image"]
        decision = res["decision"]
        confidence = res["confidence"]
        latency = res["latency_ms"]

        is_defect = "REJECT" in decision
        badge_color = "#D32F2F" if is_defect else "#2E7D32"

        ax.imshow(img, cmap="gray")
        ax.axis("off")

        title_text = f"[{decision}]\nConf: {confidence:.2f}% | Latency: {latency:.2f} ms"
        ax.set_title(
            title_text,
            fontsize=11,
            fontweight="bold",
            color=badge_color,
            pad=10
        )

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\n[INFO] Inspection visualization saved to: {output_path}")
    except PermissionError:
        alt_path = output_path.parent / "05_inference_result_latest.png"
        plt.savefig(alt_path, dpi=200, bbox_inches="tight")
        print(f"\n[INFO] Primary file locked. Saved to: {alt_path}")
    plt.close()


def open_file_dialog_and_inspect(detector):
    """
    Opens native OS file chooser to allow uploading and inspecting any bearing photo.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[ERROR] Tkinter is not available. Please pass --image path/to/file instead.")
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("\n[INFO] Opening file selection dialog. Please choose a bearing image...")
    file_path = filedialog.askopenfilename(
        title="Select Bearing Image for Inspection",
        filetypes=[
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()

    if not file_path:
        print("[INFO] No file selected. Canceled.")
        return

    print(f"\n[INSPECTING] File: {file_path}")
    res = detector.predict_file(file_path)
    
    print("\n" + "=" * 60)
    print("                QC INSPECTION REPORT")
    print("=" * 60)
    print(f"File       : {Path(file_path).name}")
    print(f"Decision   : {res['decision']}")
    print(f"Confidence : {res['confidence']:.2f}%")
    print(f"Latency    : {res['latency_ms']:.2f} ms")
    print(f"Breakdown  : Defect: {res['probabilities']['defect_ring']:.2f}% | "
          f"Normal: {res['probabilities']['normal_ring']:.2f}%")
    print("=" * 60)

    visualize_inspection([res])


def run_realtime_camera(detector, camera_index=0):
    """
    Streams live video from camera, crops inspection center ROI, and overlays real-time HUD.
    """
    try:
        import cv2
    except ImportError:
        print("\n" + "=" * 65)
        print("[ERROR] OpenCV (cv2) is not installed in your Python environment.")
        print("To enable real-time camera inspection, please install it via:")
        print("    pip install opencv-python")
        print("=" * 65 + "\n")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with index {camera_index}.")
        print("Please ensure your webcam or USB camera is connected and not in use by another app.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 65)
    print("       STAGE 5: REAL-TIME CAMERA QC INSPECTION STARTED")
    print("=" * 65)
    print("Controls:")
    print("  [Q]     : Quit inspection stream")
    print("  [S]     : Save current frame snapshot to results/")
    print("  [SPACE] : Pause / Resume live inspection")
    print("=" * 65 + "\n")

    is_paused = False
    prev_frame_time = time.time()
    last_res = None

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame from camera.")
                break

            # Frame dimensions
            h, w, _ = frame.shape
            
            # Define central inspection Region of Interest (ROI) - square box
            box_size = int(min(h, w) * 0.60)
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Extract ROI and convert to PIL Image
            roi = frame[y1:y2, x1:x2]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_roi = Image.fromarray(rgb_roi)

            # Perform inference on central ROI
            last_res = detector.predict_image(pil_roi)

        curr_frame_time = time.time()
        fps = 1.0 / max(curr_frame_time - prev_frame_time, 1e-5)
        prev_frame_time = curr_frame_time

        display_frame = frame.copy()

        if last_res is not None:
            is_defect = "REJECT" in last_res["decision"]
            hud_color = (0, 0, 220) if is_defect else (0, 180, 0) # Red for Defect, Green for Normal (BGR)

            # Draw central targeting reticle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), hud_color, 3)

            # Draw top HUD Banner
            cv2.rectangle(display_frame, (0, 0), (w, 55), (20, 20, 20), -1)
            banner_text = f"DECISION: {last_res['decision']} | CONF: {last_res['confidence']:.1f}%"
            cv2.putText(display_frame, banner_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud_color, 2, cv2.LINE_AA)

            # Draw bottom telemetry info
            telemetry_text = f"Latency: {last_res['latency_ms']:.1f}ms | FPS: {fps:.1f} | [Q]: Quit [S]: Save [SPACE]: Freeze"
            cv2.putText(display_frame, telemetry_text, (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            if is_paused:
                cv2.putText(display_frame, "[PAUSED / INSPECTION FROZEN]", (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Industrial Quality Control - Bearing Defect Inspection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # Space to freeze/unfreeze
            is_paused = not is_paused
        elif key == ord('s'):  # 's' to save snapshot
            snap_path = PROJECT_ROOT / "results" / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(snap_path), display_frame)
            print(f"[SNAPSHOT] Saved inspection frame to: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Real-time camera inspection session ended.")


def run_automated_demo(detector):
    """
    Demonstrates inspection on unseen test set samples.
    """
    test_dir = PROJECT_ROOT / "dataset_split" / "test"
    defect_samples = list((test_dir / "defect_ring").glob("*.jpg"))
    normal_samples = list((test_dir / "normal_ring").glob("*.jpg"))

    if not defect_samples or not normal_samples:
        print("[WARN] Test directory samples not found.")
        return

    selected = [defect_samples[0], normal_samples[0]]
    results = []

    print("\n" + "=" * 65)
    print("         RUNNING AUTOMATED TEST DATASET DEMO")
    print("=" * 65)

    for path in selected:
        res = detector.predict_file(path)
        results.append(res)
        marker = "[REJECT]" if "REJECT" in res["decision"] else "[PASS]"
        print(f"\nSample     : {path.name}")
        print(f"Status     : {marker} {res['decision']}")
        print(f"Confidence : {res['confidence']:.2f}%")
        print(f"Latency    : {res['latency_ms']:.2f} ms")
        print(f"Breakdown  : Defect: {res['probabilities']['defect_ring']:.2f}% | "
              f"Normal: {res['probabilities']['normal_ring']:.2f}%")

    print("\n" + "=" * 65)
    visualize_inspection(results)


def main():
    parser = argparse.ArgumentParser(description="Industrial Ring Defect Inspection System")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file for inspection")
    parser.add_argument("--upload", action="store_true", help="Open Windows File Dialog to select and upload an image")
    parser.add_argument("--camera", action="store_true", help="Start Real-Time Camera / Webcam inspection stream")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run automated inspection on test dataset samples")
    args = parser.parse_args()

    detector = RingDefectDetector()

    if args.image:
        res = detector.predict_file(args.image)
        print("\n" + "=" * 55)
        print("          INSPECTION REPORT")
        print("=" * 55)
        print(f"File       : {res['image_path']}")
        print(f"Decision   : {res['decision']}")
        print(f"Confidence : {res['confidence']:.2f}%")
        print(f"Latency    : {res['latency_ms']:.2f} ms")
        print("=" * 55)
        visualize_inspection([res])
        return

    if args.upload:
        open_file_dialog_and_inspect(detector)
        return

    if args.camera:
        run_realtime_camera(detector, camera_index=args.camera_id)
        return

    if args.demo:
        run_automated_demo(detector)
        return

    # Interactive Menu if launched without CLI arguments
    print("\n" + "=" * 60)
    print("   INDUSTRIAL QUALITY INSPECTION SYSTEM: BEARING DEFECT")
    print("=" * 60)
    print("  [1] Upload / Select Image File (Windows File Dialog)")
    print("  [2] Real-Time Camera Inspection (Live Webcam Feed)")
    print("  [3] Automated Demo on Unseen Test Dataset Samples")
    print("  [0] Exit")
    print("=" * 60)

    try:
        choice = input("Enter your choice [1/2/3/0]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "1":
        open_file_dialog_and_inspect(detector)
    elif choice == "2":
        run_realtime_camera(detector, camera_index=args.camera_id)
    elif choice == "3":
        run_automated_demo(detector)
    else:
        print("[INFO] Exiting inspection system. Goodbye.")


if __name__ == "__main__":
    main()
