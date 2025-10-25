from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import io
from PIL import Image
from ultralytics import YOLO

# ==========================
#  1. กำหนดค่า
# ==========================
app = Flask(__name__)

# กำหนด path ของโมเดล
YOLO_MODEL_PATH = "yolov8m.pt"
PREDICTION_MODEL_PATH = "saved_model/newsimple_model.h5"
CLASS_NAMES = ["d1", "d2", "d3"]

# โหลดโมเดล
yolo_model = YOLO(YOLO_MODEL_PATH)
prediction_model = load_model(PREDICTION_MODEL_PATH)
IMG_SIZE = (224, 224)


# ==========================
#  2. ฟังก์ชันตรวจจับช้าง (เช็คก่อน)
# ==========================
def detect_and_crop_image(image_data):
    """
    รับภาพ (bytes) → ตรวจจับช้างด้วย YOLO → ตัดช้าง → ปรับขนาดเป็น 224x224
    คืนค่า: ภาพที่ตัด (ถ้ามี), หรือ dict ที่มี "action"
    """
    try:
        # โหลดภาพ
        img = Image.open(io.BytesIO(image_data))

        # แปลงเป็น RGB เพื่อรองรับ PNG ที่มี transparency
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")

        img_array = np.array(img)

        # ใช้ YOLO ตรวจจับวัตถุ
        results = yolo_model(img)

        if len(results) == 0:
            return None, {"error": "ไม่พบวัตถุใดๆ ในภาพ", "action": "skip"}

        # ดึง bounding box และความน่าจะเป็น
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names

        best_idx = np.argmax(confs)
        x1, y1, x2, y2 = boxes[best_idx]
        label_idx = int(labels[best_idx])
        label_name = class_names[label_idx]

        if label_name != "elephant":
            return None, {"error": f"ตรวจพบวัตถุ: {label_name} ไม่ใช่ช้าง", "action": "skip"}

        if x1 < 0 or y1 < 0 or x2 > img.size[0] or y2 > img.size[1]:
            return None, {"error": "bounding box อยู่นอกภาพ", "action": "skip"}

        # ตัดภาพ
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img = cropped_img.resize(IMG_SIZE)

        # 🔍 แปลงเป็น RGB ทันทีก่อนแปลงเป็น array (สำคัญ!)
        if cropped_img.mode in ("RGBA", "LA", "P"):
            cropped_img = cropped_img.convert("RGB")

        cropped_array = img_to_array(cropped_img)
        cropped_array = np.expand_dims(cropped_array, axis=0)
        cropped_array = cropped_array / 255.0

        return cropped_array, {
            "detected": True,
            "class": "elephant",
            "confidence": float(confs[best_idx]),
            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "action": "detected",
        }

    except Exception as e:
        return None, {"error": str(e), "action": "error"}


# ==========================
#  3. ฟังก์ชันทำนายภาพ (กับการตรวจสอบความน่าจะเป็น)
# ==========================
def predict_image_from_crop(cropped_array):
    """
    ใช้โมเดลที่ฝึกแล้วเพื่อทำนายชื่อคลาส
    กลับค่าที่มี "ความน่าจะเป็น" และ "เตือนความไม่แน่ใจ"
    """
    try:
        predictions = prediction_model.predict(cropped_array)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])

        # ตรวจสอบความน่าจะเป็น
        if confidence < 0.999:
            uncertainty = "ไม่แน่ใจ"
        else:
            uncertainty = "แน่ใจ"

        # สร้างผลลัพธ์
        result = {
            "class": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "probabilities": dict(zip(CLASS_NAMES, predictions.tolist())),
            "uncertainty": uncertainty,
        }

        return result

    except Exception as e:
        return {"error": str(e), "action": "predict_error"}


# ==========================
#  4. เส้นทาง API
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "ไม่มีชื่อไฟล์ภาพ"}), 400

    image_data = file.read()

    # ขั้นตอน 1: เช็คก่อนด้วย YOLO
    cropped_array, detection_result = detect_and_crop_image(image_data)

    if detection_result is None:
        return jsonify(detection_result), 400

    if detection_result.get("action") == "skip":
        return jsonify(
            {"error": "ไม่พบช้างในภาพ", "message": "ภาพไม่มีช้าง ไม่สามารถใช้ในการทำนายได้"}
        ), 400

    if detection_result.get("action") == "error":
        return jsonify(
            {"error": detection_result["error"], "message": "เกิดข้อผิดพลาดในการตรวจจับภาพ"}
        ), 400

    # ขั้นตอน 2: ทำนาย
    prediction_result = predict_image_from_crop(cropped_array)

    # รวมผลลัพธ์
    final_result = {"detection": detection_result, "prediction": prediction_result}

    return jsonify(final_result)


# ==========================
# 🔹 5. เริ่มต้น API
# ==========================
if __name__ == "__main__":
    print("🚀 รีสตาร์ท API ที่ http://localhost:5000/predict")
    app.run(host="0.0.0.0", port=5800, debug=False)
