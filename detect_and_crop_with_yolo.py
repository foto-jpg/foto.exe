# detect_and_crop_with_yolo.py
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image

# กำหนด path
dataset_dir = "datasetnew"
crop_dir = "datasetnewcrop"
model_path = "yolov8m.pt"  # หรือใช้โมเดลที่คุณฝึกมา

# สร้างโฟลเดอร์
os.makedirs(crop_dir, exist_ok=True)

# โหลดโมเดล
model = YOLO(model_path)

class_names = sorted(os.listdir(dataset_dir))

for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        print(f"⚠️ โฟลเดอร์ไม่พบ: {class_path}")
        continue

    crop_path = os.path.join(crop_dir, class_name)
    os.makedirs(crop_path, exist_ok=True)

    for filename in os.listdir(class_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(class_path, filename)
            try:
                # ใช้ YOLO ตรวจจับช้าง
                results = model(img_path)

                # ดูว่ามีช้างหรือไม่
                if len(results) == 0:
                    print(f"❌ ไม่พบช้างในภาพ: {filename}")
                    continue

                # ดึง bounding box ที่พบ
                boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
                confs = results[0].boxes.conf.cpu().numpy()

                # ดึงภาพที่ตรวจพบ
                img = Image.open(img_path)
                img_array = np.array(img)

                # ตัดช้างจาก bounding box ที่มีความน่าจะเป็นสูงสุด
                best_idx = np.argmax(confs)
                x1, y1, x2, y2 = boxes[best_idx]

                # ตัดภาพ
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img = cropped_img.resize((224, 224))

                # บันทึก
                save_path = os.path.join(crop_path, filename)
                cropped_img.save(save_path)
                print(f"✅ ตรวจจับและตัดช้างแล้ว: {save_path}")

            except Exception as e:
                print(f"❌ ข้อผิดพลาดเมื่อจัดการภาพ {filename}: {e}")

print("🎉 สรุป: ภาพทั้งหมดถูกตรวจจับและตัดช้างแล้ว")
