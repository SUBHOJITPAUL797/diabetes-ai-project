# 🩺 GlucoWise — Diabetes AI Project Context Document

> **For AI / Developer Reference** — Read this file first before making any changes to the codebase.

---

## 📌 Project Overview

**Project Name:** GlucoWise — Smart Diabetes Companion  
**Folder:** `DIABATIES AI PROJECT OF DON`  
**Purpose:** A health-tech application that predicts diabetes risk using machine learning and helps users track their glucose, activity, and wellness data daily.

This project is a **full-stack diabetes prediction system** composed of two independent layers:
1. A **Python ML backend** (FastAPI) that serves a pre-trained scikit-learn model
2. A **standalone frontend PWA** (single HTML file) that works independently via a rule-based scoring engine AND can optionally call the backend API

---

## 📁 File Structure

```
DIABATIES AI PROJECT OF DON/
├── main.py                  # FastAPI backend — production ML API server
├── app.py                   # Streamlit prototype UI (dev/testing only)
├── diabetes_model.joblib    # Pre-trained scikit-learn ML model (binary file)
├── index.html               # Frontend PWA — single file app (1086 lines)
├── manifest.json            # PWA manifest (installable as mobile app)
├── sw.js                    # Service Worker for offline caching
└── logo.jpeg                # App branding logo image
```

> **Note:** Icon files (`icon-192.png`, `icon-512.png`, `icon.ico`) referenced by the manifest and HTML are **not present** in the directory. These need to be added for full PWA functionality.

---

## 🔧 Tech Stack

### Backend (`main.py`)
| Component | Technology |
|-----------|------------|
| Framework | **FastAPI** (Python) |
| ML Library | **scikit-learn** (via joblib/pickle) |
| Data Validation | **Pydantic** (v1 style, `BaseModel`, `validator`) |
| Logging | Python standard `logging` |
| Server | **Uvicorn** (ASGI) |
| CORS | `fastapi.middleware.cors.CORSMiddleware` |

### Frontend (`index.html`)
| Component | Technology |
|-----------|------------|
| Framework | **Vanilla HTML/CSS/JS** (zero dependencies) |
| Fonts | Google Fonts — `Fraunces` (serif headings) + `DM Sans` (body) |
| PWA | Service Worker + Web App Manifest |
| State | `localStorage` (client-side only) |
| Styling | Custom CSS design tokens (CSS variables) |

### Prototype (`app.py`)
| Component | Technology |
|-----------|------------|
| UI Framework | **Streamlit** |
| API Client | Python `requests` library |

---

## 🤖 The ML Model

### Model File
- **File:** `diabetes_model.joblib`
- **Format:** joblib (scikit-learn compatible)
- **Type:** Likely Logistic Regression or Random Forest (supports `predict_proba`)

### Input Features (7 total)
| Feature | Type | Valid Range | Description |
|---------|------|-------------|-------------|
| `Pregnancies` | int | 0–17 | Number of pregnancies |
| `Glucose` | int | 0–200 | Plasma glucose (mg/dL) |
| `BloodPressure` | int | 0–122 | Diastolic blood pressure (mmHg) |
| `Insulin` | int | 0–846 | 2-hour serum insulin (mu U/ml) |
| `BMI` | float | 0–67 | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | float | 0–2.42 | Genetic risk score |
| `Age` | int | 21–81 | Age in years |

### Output
```json
{
  "prediction": 1,
  "probability": 0.7813,
  "risk_level": "High",
  "confidence": 56.26,
  "input_features": { ... },
  "timestamp": "2026-05-05T09:00:00"
}
```

### Risk Classification Logic
```
probability < 0.30  → "Low" risk
probability < 0.70  → "Medium" risk
probability >= 0.70 → "High" risk

confidence = abs(probability - 0.5) × 200  (scale: 0–100%)
```

---

## 🌐 Backend API (`main.py`)

**Default Port:** `8000`  
**API Docs:** `http://localhost:8000/api/docs` (Swagger UI)  
**ReDoc:** `http://localhost:8000/api/redoc`

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root — API info |
| `GET` | `/api/health` | Health check, model status |
| `POST` | `/api/predict` | Single patient prediction |
| `POST` | `/api/predict-batch` | Batch predictions (up to 1000) |
| `GET` | `/api/model-info` | Model metadata and feature bounds |

### How to Run Backend
```bash
# Install dependencies
pip install fastapi uvicorn scikit-learn pydantic numpy joblib

# Run server
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### CORS Policy
Currently configured with `"*"` (open) for development. The comment mentions restricting to Cloudflare domains in production:
- `https://*.cloudflare.com`
- `http://localhost:3000`, `5173`, `8080`

### Model Loading Strategy
The backend auto-discovers the model file from these paths (in order):
1. `models/diabetes_model.pkl`
2. `models/diabetes_model.joblib`
3. `diabetes_model.pkl`
4. `diabetes_model.joblib` ← **Current location (root directory)**

If no model is found, the server **starts in mock/demo mode** (does not crash).

---

## 📱 Frontend PWA (`index.html`)

A complete single-file mobile-first Progressive Web App. **Does NOT require the backend** — it has its own client-side scoring algorithm.

### App Name
**GlucoWise** (PWA short name)

### Screens / Navigation
The app has **4 screens** with a bottom navigation bar:

| Screen ID | Nav Label | Description |
|-----------|-----------|-------------|
| `#home` | Home | Dashboard: risk orb, daily tip, quick stats, streak |
| `#predict` | Predict | Risk assessment form with sliders and toggles |
| `#tips` | Tips | Health tips library, filterable by category |
| `#log` | Log | Daily entry form for glucose, weight, BP, steps, mood |

### Design System (CSS Variables)
```css
--bg:        #0b1a0e   /* Deep dark forest green (main background) */
--bg2:       #112016   /* Slightly lighter dark green */
--panel:     #162b1a   /* Panel background */
--card:      #1c3421   /* Card background */
--border:    #2d5237   /* Border color */
--accent:    #4ade80   /* Neon green (primary accent) */
--accent2:   #86efac   /* Light green */
--accent3:   #bbf7d0   /* Very light green */
--warn:      #fbbf24   /* Amber warning */
--danger:    #f87171   /* Coral danger/error */
--text:      #e8f5ec   /* Near-white text */
--muted:     #6b9b78   /* Muted green text */
--font-head: 'Fraunces', serif
--font-body: 'DM Sans', sans-serif
--radius:    18px
```

**Theme:** Dark mode, deep forest green palette with neon green accents. Ultra-premium mobile health app aesthetic.

### Frontend Scoring Algorithm
The frontend uses its **own independent rule-based scoring** (NOT the ML model):

```javascript
let score = 10;  // base
if (age > 45)    score += 15;
if (glu > 125)   score += 40;  // High glucose
else if (glu > 100) score += 15; // Borderline
if (bmi > 30)    score += 20;  // Obese
else if (bmi > 25) score += 10; // Overweight
if (bp > 90)     score += 10;
if (familyHistory === 'yes') score += 15;
if (activity === 'sedentary') score += 10;
score = Math.min(score, 100);
```

> **Important:** The frontend scoring is **not connected to the ML model**. The `Streamlit app.py` IS connected to the ML API via `POST http://127.0.0.1:8000/api/predict`.

### Data Persistence
- Uses **localStorage** (browser storage, client-side only)
- `gluLogs` — stores last 10 daily log entries
- `lastScore` — stores the most recent risk prediction score

### PWA Features
- **Installable** on Android/iOS home screens
- Offline-capable via Service Worker (`sw.js`)
- Cache strategy: Cache-first for `/`, `/index.html`, `/manifest.json`; network fallback

---

## 📊 Tips Database (Frontend Hardcoded)

5 health tip cards currently in `index.html`:

| Category | Title | Tag |
|----------|-------|-----|
| `diet` | Fiber First | Nutrition |
| `exercise` | Power Walk | Activity |
| `lifestyle` | Sleep Quality | Recovery |
| `monitoring` | Morning Check | Habit |
| `diet` | Green Tea | Hydration |

Tips are filterable by: `All`, `Diet`, `Exercise`, `Lifestyle`, `Monitor`

---

## 🚧 Known Issues / TODOs

1. **PWA icons missing** — `icon-192.png`, `icon-512.png`, `icon.ico` are referenced but not in the directory
2. **Frontend ↔ Backend not connected** — The main `index.html` uses client-side scoring only; it does NOT call the FastAPI API
3. **Skin Thickness feature missing** — The Pima Indian Diabetes dataset typically includes `SkinThickness` as a feature, but this model uses only 7 features (excludes SkinThickness)
4. **Streamlit app hardcoded URL** — `app.py` calls `http://127.0.0.1:8000/api/predict` which only works when both run locally
5. **CORS wildcard** — `"*"` in `ALLOWED_ORIGINS` should be replaced with specific domains for production
6. **No requirements.txt** — Python dependencies not documented in a requirements file
7. **No `.env` or config file** — API URL is hardcoded in `app.py`

---

## 🎯 Potential Next Steps (AI Suggestions)

- [ ] Connect `index.html` frontend to the FastAPI backend (replace client-side scoring with real ML API call)
- [ ] Add `requirements.txt` with all Python dependencies
- [ ] Generate and add PWA icon files (192×192, 512×512, favicon)
- [ ] Add a chart/graph in the Log screen to visualize glucose trends over time
- [ ] Deploy backend to a cloud server (Railway, Render, etc.) and update the frontend API URL
- [ ] Add user authentication (currently no login — all data is local per-device)
- [ ] Expand tips database with more categorized health content
- [ ] Add proper error handling in the frontend for when the API is unavailable (graceful fallback)

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     USER DEVICE                         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │            index.html (PWA Frontend)             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌──────┐ ┌─────────┐ │   │
│  │  │  Home   │ │ Predict │ │ Tips │ │   Log   │ │   │
│  │  │ Screen  │ │ Screen  │ │Screen│ │ Screen  │ │   │
│  │  └─────────┘ └────┬────┘ └──────┘ └────┬────┘ │   │
│  │                   │ Scoring Algorithm    │       │   │
│  │              ┌────▼────┐          ┌─────▼────┐  │   │
│  │              │Rule-Based│         │localStorage│  │   │
│  │              │ Scoring │         │  (gluLogs) │  │   │
│  │              └─────────┘         └────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   DEVELOPER MACHINE                     │
│                                                         │
│  ┌─────────────────┐         ┌──────────────────────┐  │
│  │    app.py        │ HTTP   │      main.py          │  │
│  │  (Streamlit UI)  │──POST─▶│   (FastAPI Backend)   │  │
│  │  localhost:8501  │        │   localhost:8000       │  │
│  └─────────────────┘         └──────────┬───────────┘  │
│                                          │              │
│                                  ┌───────▼──────┐      │
│                                  │ diabetes_model│      │
│                                  │   .joblib     │      │
│                                  │ (sklearn ML)  │      │
│                                  └──────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Quick Start Guide

### Run Frontend Only (No Backend Needed)
Just open `index.html` in a browser — works completely offline.

### Run Full Stack
```bash
# 1. Start Backend
pip install fastapi uvicorn scikit-learn joblib numpy pydantic
python main.py

# 2. Open Frontend (in browser)
# Navigate to index.html OR serve via a local HTTP server:
python -m http.server 3000
# Then visit: http://localhost:3000

# 3. (Optional) Run Streamlit prototype
pip install streamlit requests
streamlit run app.py
```

---

*Document created: 2026-05-05 | By: Antigravity AI*
