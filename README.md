# Biochar Prediction API

FastAPI service ทำนายคุณสมบัติ biochar จาก pyrolysis conditions + biomass composition

## โครงสร้างโปรเจกต์

```
biochar-api/
├── main.py
├── requirements.txt
├── render.yaml
├── README.md
└── models/                  ← วางไฟล์ .joblib ทั้งหมดตรงนี้
    ├── C_char__HistGradientBoosting.joblib
    ├── H_C_char__GradientBoosting.joblib
    ├── O_C_char__CatBoost.joblib
    ├── FC_char__GradientBoosting.joblib
    └── Yield_char__CatBoost.joblib
```

## วิธี Deploy บน Render

1. สร้าง folder `models/` แล้วใส่ไฟล์ `.joblib` ทั้ง 5 ไฟล์
2. Push ทั้งโปรเจกต์ขึ้น **GitHub** (repo ใหม่ก็ได้)
3. ไปที่ [render.com](https://render.com) → New → Web Service
4. เชื่อม GitHub repo
5. ตั้งค่าดังนี้:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variable:** `MODEL_DIR = models`
6. Deploy → รอ ~2 นาที

## Endpoints

| Method | URL | คำอธิบาย |
|--------|-----|----------|
| GET | `/` | ตรวจสอบสถานะและโมเดลที่โหลด |
| POST | `/predict` | ทำนายทุก target พร้อมกัน |
| POST | `/predict/{target}` | ทำนาย target เดียว |
| GET | `/docs` | Swagger UI (ทดสอบได้เลย) |

## ตัวอย่าง Input (JSON)

```json
{
  "Temp": 500,
  "RT": 60,
  "HR": 10,
  "C_bio": 45.0,
  "H_bio": 6.0,
  "N_bio": 1.0,
  "O_bio": 40.0,
  "Ash_bio": 5.0,
  "VM_bio": 75.0,
  "FC_bio": 15.0
}
```

## ตัวอย่าง Output

```json
{
  "predictions": {
    "C_char": 72.3,
    "H_C_char": 0.42,
    "O_C_char": 0.11,
    "FC_char": 68.5,
    "Yield": 35.2
  },
  "units": {
    "C_char": "%",
    "H_C_char": "molar ratio",
    "O_C_char": "molar ratio",
    "FC_char": "%",
    "Yield": "%"
  }
}
```

## Google Apps Script ตัวอย่าง

```javascript
function predictBiochar() {
  const API_URL = "https://YOUR-APP.onrender.com/predict";
  
  const payload = {
    Temp: 500, RT: 60, HR: 10,
    C_bio: 45.0, H_bio: 6.0, N_bio: 1.0, O_bio: 40.0,
    Ash_bio: 5.0, VM_bio: 75.0, FC_bio: 15.0
  };

  const options = {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  const response = UrlFetchApp.fetch(API_URL, options);
  const result = JSON.parse(response.getContentText());
  
  Logger.log(result.predictions);
  return result.predictions;
}
```
