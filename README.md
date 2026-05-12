# Res2bChar ML API

FastAPI server สำหรับ predict biochar properties จาก 5 ML models

---

## โครงสร้างโฟลเดอร์

```
res2bchar-api/
├── main.py
├── requirements.txt
├── render.yaml
└── models/                        ← วางไฟล์ .joblib ทั้งหมดไว้ที่นี่
    ├── Yield_char__CatBoost.joblib
    ├── Yield_char__CatBoost__meta.json
    ├── C_char__HistGradientBoosting.joblib
    ├── C_char__HistGradientBoosting__meta.json
    ├── H_C_char__GradientBoosting.joblib
    ├── H_C_char__GradientBoosting__meta.json
    ├── O_C_char__CatBoost.joblib
    ├── O_C_char__CatBoost__meta.json
    ├── FC_char__GradientBoosting.joblib
    └── FC_char__GradientBoosting__meta.json
```

---

## วิธี Deploy บน Render

1. สร้าง GitHub repo ใหม่ แล้ว push โฟลเดอร์ทั้งหมดนี้ขึ้นไป  
   (รวมถึงโฟลเดอร์ `models/` ที่มีไฟล์ .joblib และ .json ครบ)

2. ไปที่ https://render.com → **New → Web Service**

3. เชื่อม GitHub repo ที่สร้างไว้

4. ตั้งค่า:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. กด **Deploy** รอสักครู่ แล้ว URL จะเป็น:  
   `https://res2bchar-ml-api.onrender.com`

---

## API Endpoints

### `POST /predict`  ← ใช้หลักตัวนี้

รับ input ทุกตัว → return ค่า predict ทั้ง 5 โมเดลพร้อมกัน

**Request body:**
```json
{
  "Temp": 500,
  "RT": 60,
  "HR": 10,
  "VM_bio": 70.0,
  "Ash_bio": 5.0,
  "FC_bio": 15.0,
  "C_bio": 45.0,
  "H_bio": 6.0,
  "N_bio": 1.0,
  "O_bio": 40.0
}
```

**Response:**
```json
{
  "predicted_yield": 32.15,
  "predicted_C":     68.42,
  "predicted_HC":    0.38,
  "predicted_OC":    0.12,
  "predicted_FC":    55.30,
  "errors": null,
  "input": { ... }
}
```

### `POST /predict/yield`

Return เฉพาะ `predicted_yield` — compatible กับ Apps Script เดิม 100%

### `GET /health`

ตรวจสอบว่า models โหลดครบหรือไม่

---

## แก้ Apps Script ให้รับค่าครบทุกโมเดล

เปลี่ยน `callMLApi_` ให้เรียก `/predict` แทน `/predict/yield`:

```javascript
function callMLApi_(payload) {
  // เปลี่ยนจาก /predict/yield → /predict เพื่อรับครบทุกค่า
  const url = 'https://res2bchar-ml-api.onrender.com/predict';
  // ... โค้ดที่เหลือเหมือนเดิม
}

function runMLPrediction(input) {
  const payload = { ... };  // เหมือนเดิม
  const result = callMLApi_(payload);
  // ตอนนี้ result มี:
  //   result.predicted_yield  → Yield_char (%)
  //   result.predicted_C      → C_char (%)
  //   result.predicted_HC     → H/C ratio
  //   result.predicted_OC     → O/C ratio
  //   result.predicted_FC     → FC_char (%)
  return result;
}
```

---

## Input Fields ที่แต่ละโมเดลต้องการ

| โมเดล | Features หลัก |
|---|---|
| Yield_char | Temp, RT, HR, VM_bio, Ash_bio, FC_bio, H/C_bio, O/C_bio, VM_to_Ash, FC_to_Ash, FC_bio_derived, O_bio_derived |
| C_char | Temp, RT, HR, C_bio, H_bio, N_bio, Ash_bio, O_bio, H/C_bio, O/C_bio, FC_bio_derived, O_bio_derived |
| H/C_char | เหมือน C_char |
| O/C_char | เหมือน C_char |
| FC_char | Temp, RT, HR, VM_bio, Ash_bio, FC_bio, C_bio, H/C_bio, O/C_bio, VM_to_Ash, FC_to_Ash, FC_bio_derived, O_bio_derived |

> ค่า H/C_bio, O/C_bio, VM_to_Ash, FC_to_Ash, FC_bio_derived, O_bio_derived  
> คำนวณอัตโนมัติใน `build_features()` ไม่ต้องส่งมาจาก frontend
