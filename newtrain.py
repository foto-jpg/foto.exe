# newtrain.py (แก้ไขสำหรับโฟลเดอร์ d1, d2, d3)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==========================
# 🔹 1. กำหนด path และค่าพื้นฐาน
# ==========================
dataset_dir = "datasetnewcrop"  # โฟลเดอร์ของภาพที่คุณจับจากวิดีโอ
img_size = (224, 224)
batch_size = 64
epochs = 6

# ดึงจำนวนโฟลเดอร์ (เช่น d1, d2, d3 → 3 คลาส)
class_names = sorted(os.listdir(dataset_dir))
num_classes = len(class_names)  # เช่น 3 ถ้ามี d1, d2, d3
print(f"✅ พบโฟลเดอร์คลาส: {class_names} → จำนวนคลาส = {num_classes}")

# ==========================
# 🔹 2. โหลดข้อมูลภาพ
# ==========================
X, y = [], []

for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        print(f"⚠️ โฟลเดอร์ไม่พบ: {class_path}")
        continue

    for filename in os.listdir(class_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(class_path, filename)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            # แปลง d1 → 1 → 0, d2 → 2 → 1, d3 → 3 → 2
            y.append(int(class_name[1:]) - 1)

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=num_classes)  # ตอนนี้ y อยู่ในช่วง [0, num_classes-1]

# ==========================
# 🔹 3. แบ่งข้อมูล
# ==========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ==========================
# 🔹 4. สร้างโมเดล
# ==========================
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==========================
# 🔹 5. เรียนรู้ (train)
# ==========================
print("🚀 เริ่มการฝึกโมเดลจากศูนย์...")
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    verbose=1,
)

# ==========================
# 🔹 6. บันทึกโมเดล
# ==========================
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/newsimple_model.h5")
print("✅ บันทึกโมเดลแล้วที่ saved_model/newsimple_model.h5")
print(f"🎉 สร้างโมเดลสำเร็จสำหรับ {num_classes} คลาส (d1, d2, ..., d{num_classes})!")
