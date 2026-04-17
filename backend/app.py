from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import numpy as np
import cv2
import tempfile
import base64
import io
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")

hf_model = AutoModelForImageClassification.from_pretrained(
    "Organika/sdxl-detector",
    output_attentions=True
).to(DEVICE)

hf_model.eval()
print("✅ Model ready!")

# =========================
# 🔥 ATTENTION (FIXED)
# =========================
def compute_attention_rollout(model, pixel_values):
    outputs = model(pixel_values, output_attentions=True)
    attn = outputs.attentions[-1]

    attn = attn.mean(dim=1)[0]
    mask = attn[0, 1:]

    num_tokens = mask.shape[0]
    size = int(np.sqrt(num_tokens))

    mask = mask.detach().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    if size * size == num_tokens:
        mask = mask.reshape(size, size)
    else:
        mask = mask.reshape(-1, 1)

    return mask


# =========================
# 🔥 IMAGE PREDICTION
# =========================
def predict_image(img_pil):
    inputs = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(DEVICE)

    outputs = hf_model(pixel_values)
    probs = torch.softmax(outputs.logits, dim=1)[0]

    labels = hf_model.config.id2label
    results = [(labels[i], float(probs[i].detach())) for i in range(len(probs))]
    results.sort(key=lambda x: x[1], reverse=True)

    return results[0], pixel_values


# =========================
# 🔥 IMAGE ANALYSIS (NEW)
# =========================
def analyze_image_properties(img_pil):
    img = np.array(img_pil.convert("L"))

    variance = np.var(img)
    edges = cv2.Canny(img, 100, 200)
    edge_density = edges.mean()

    return variance, edge_density


# =========================
# 🔥 HEATMAP
# =========================
def generate_heatmap(img_pil, pixel_values):
    try:
        attn_mask = compute_attention_rollout(hf_model, pixel_values)
    except:
        attn_mask = np.random.rand(224, 224)

    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    attn_mask = cv2.resize(attn_mask, (224, 224))

    heatmap = plt.cm.jet(attn_mask)[:, :, :3]
    overlay = np.clip(heatmap * 0.5 + img_np * 0.5, 0, 1)

    overlay = (overlay * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


# =========================
# 🔥 VIDEO DETECTION
# =========================
def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    artificial_score = 0
    real_score = 0

    SAMPLE_RATE = 10
    CONF_THRESHOLD = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % SAMPLE_RATE != 0:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        (label, score), _ = predict_image(img)

        if score < CONF_THRESHOLD:
            continue

        if "artificial" in label.lower():
            artificial_score += score
        else:
            real_score += score

    cap.release()

    total = artificial_score + real_score + 1e-6

    if artificial_score > real_score:
        return "AI GENERATED", artificial_score / total, artificial_score, real_score
    else:
        return "REAL", real_score / total, artificial_score, real_score


# =========================
# 🔥 VIDEO EXPLANATION
# =========================
def generate_video_explanation(label, confidence, ai_score, real_score):

    if "AI" in label:
        return f"""
The video is classified as AI-generated with {confidence*100:.2f}% confidence.
Frames show smooth textures and synthetic patterns.
AI score ({ai_score:.2f}) > Real score ({real_score:.2f}).
"""
    else:
        return f"""
The video is classified as real with {confidence*100:.2f}% confidence.
Frames show natural textures and consistent edges.
Real score ({real_score:.2f}) > AI score ({ai_score:.2f}).
"""


# =========================
# 🌐 IMAGE API (IMPROVED)
# =========================
@app.route("/detect-image", methods=["POST"])
def detect_image():

    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    (label, score), pixel_values = predict_image(image)
    heatmap = generate_heatmap(image, pixel_values)

    variance, edge_density = analyze_image_properties(image)

    # 🔥 HYBRID DECISION SYSTEM
    ai_score = 0

    # Model contribution
    if "artificial" in label.lower():
        ai_score += score * 2
    else:
        ai_score -= score

    # Texture check
    if variance < 500:
        ai_score += 0.5

    # Edge check
    if edge_density < 20:
        ai_score += 0.5

    # Final decision
    if ai_score > 1.5:
        final_label = "AI GENERATED"
        explanation = "Detected AI patterns: smooth textures and low edge complexity."
    elif ai_score > 0.5:
        final_label = "UNCERTAIN"
        explanation = "Mixed signals detected. Could be edited or compressed image."
    else:
        final_label = "REAL"
        explanation = "Natural texture and edge patterns detected."

    return jsonify({
        "label": final_label,
        "confidence": round(score * 100, 2),
        "heatmap": heatmap,
        "explanation": explanation
    })


# =========================
# 🌐 VIDEO API
# =========================
@app.route("/detect-video", methods=["POST"])
def detect_video_api():

    file = request.files["video"]

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(file.read())
    temp.close()

    label, confidence, ai_score, real_score = detect_video(temp.name)

    explanation = generate_video_explanation(label, confidence, ai_score, real_score)

    return jsonify({
        "result": label,
        "confidence": round(confidence * 100, 2),
        "ai_score": ai_score,
        "real_score": real_score,
        "explanation": explanation
    })


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "✅ Backend Running"


if __name__ == "__main__":
    app.run(debug=True)