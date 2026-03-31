# MediSmart — Smart Medicine Recommendation System
### Lightweight ML-Based Backend · Flask + Decision Tree + Random Forest

---

## Project Structure

```
medismart/
├── app.py                    ← Flask backend (main file)
├── requirements.txt          ← Python dependencies
├── data/
│   ├── symtoms_df.csv        ← Disease → Symptom mapping
│   ├── medications.csv       ← Disease → Medications
│   ├── description.csv       ← Disease → Description
│   ├── precautions_df.csv    ← Disease → Precautions
│   └── Symptom-severity.csv  ← Symptom severity weights
├── templates/
│   └── index.html            ← Frontend HTML (served by Flask)
└── static/
    ├── style.css             ← Frontend CSS
    └── script.js             ← Frontend JS (calls Flask API)
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask server
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

---

## API Endpoints

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/`                   | Serve frontend HTML                  |
| GET    | `/api/health`         | Health check                         |
| GET    | `/api/symptoms`       | All 86 unique symptoms               |
| POST   | `/api/predict`        | Predict disease from symptoms        |
| GET    | `/api/diseases`       | All 41 diseases with full info       |
| GET    | `/api/disease/<name>` | Single disease details               |
| GET    | `/api/models/stats`   | ML model accuracy metrics            |
| GET    | `/api/severity`       | Symptom severity weights             |

---

## POST /api/predict — Request & Response

### Request
```json
{
  "symptoms": ["itching", "skin_rash", "fatigue"],
  "age":      25,
  "gender":   "female",
  "model":    "random_forest"
}
```

### Response
```json
{
  "success": true,
  "predictions": [
    {
      "rank":            1,
      "model":           "random_forest",
      "disease":         "Fungal infection",
      "confidence":      0.92,
      "confidence_pct":  92.0,
      "description":     "Fungal infection is a common skin condition...",
      "medications":     ["Antifungal Cream", "Fluconazole", ...],
      "precautions":     ["Bath twice", "Keep area dry", ...],
      "severity_label":  "Moderate",
      "severity_cls":    "sev-mod",
      "severity_score":  18,
      "matched_symptoms":["itching", "skin_rash", "fatigue"]
    }
  ],
  "meta": {
    "symptoms_received": 3,
    "symptoms_used":     ["itching", "skin_rash", "fatigue"],
    "age":               25,
    "gender":            "female",
    "model_used":        "random_forest"
  }
}
```

---

## ML Models

| Model          | Algorithm             | Accuracy |
|----------------|-----------------------|----------|
| Decision Tree  | Entropy / Gini Index  | ~99%     |
| Random Forest  | Ensemble (100 trees)  | ~99%     |

Both models are trained at startup on the integrated dataset
(4,920 training examples × 86 symptom features → 41 disease classes).

---

## Dataset Sources (5 CSV files)

| File                   | Records | Purpose                        |
|------------------------|---------|--------------------------------|
| symtoms_df.csv         | 4,920   | Disease → Symptom mapping      |
| medications.csv        | 41      | Disease → Medication list      |
| description.csv        | 41      | Disease → Clinical description |
| precautions_df.csv     | 41      | Disease → Precaution steps     |
| Symptom-severity.csv   | 133     | Symptom severity weights       |

---

## Team

- Palak (22103055)
- Yogita (22103118)
- Abhishek Kumar Gupta (22103048)

**Guided by:** Prof. Amandeep Kaur & Prof. Satnam Kaur  
**Department of CSE, Punjab Engineering College (Deemed to be University)**  
**February 2026**
