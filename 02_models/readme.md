# 📌 02_models  
This directory contains all machine learning and deep learning models used in **SafePath-AI**, including:

- **XGBoost** — Structured accident-risk prediction  
- **GAT (Graph Attention Network)** — Road-level baseline danger scoring  
- **YOLOv11m** — Real-time battlefield obstacle detection  

Each model contributes to generating safe and adaptive military routes during wartime.

---

## 🧠 1. XGBoost — Road Structural Risk Prediction  
**Folder:** `01_xgboost/`  
**Main file:** `xgboost_final.ipynb`

### 🎯 Purpose  
Predict the **structural accident-based risk score** for each road segment using TAAS accident data and OSM road attributes.

### 📥 Input Features  
Total: **17 road features**, including  
- `length`  
- `lanes`  
- `surface`  
- `bridge`  
- `tunnel`  
- `slope`  
- `road_type (highway)`  
- `speed_limit` (if available)  
- GPS geometry-derived features  

### 📤 Output  
- `risk_score_pred` for each `(u, v)` edge  
- Used as **pseudo-label** for GAT model  
- Stored in CSV (`xgboost_train_results.csv`)

### ⭐ Key Notes  
- XGBoost acts as a **feature-based risk estimator**  
- Provides coarse accident-prone risk patterns  
- Fast, interpretable, and stable baseline model

---

## 🧬 2. GAT (Graph Attention Network) — Baseline Road Danger  
**Folder:** `02_gat/`  
**Main file:** `gat_final.ipynb`

### 🎯 Purpose  
Use GAT to learn **spatial relational danger** between road segments.  
While XGBoost looks at features individually, GAT captures **graph structure + neighbor influence**.

### 🧩 Model Inputs  
- Road graph `G` (from OSMnx)  
- Node features: aggregated OSM properties  
- Edge features:  
  - `length`  
  - `speed`  
  - `lanes`  
  - `surface`  
- XGBoost-based `risk_score_pred` as training label

### 📤 Output  
- **`gat_weight`** assigned to each graph edge  
- Added to routing weight in pathfinding algorithms:
  ```
  final_cost = length + ALPHA * hazard_cost + gat_weight
  ```

### ⭐ Key Notes  
- GAT learns “which roads are inherently risky”  
- Captures structure like:  
  - narrow alleys → danger  
  - intersections → higher risk  
  - highway ramps → low risk  
- Works excellently with pathfinding algorithms (D* Lite, CCH, RTAA\*)

---

## 🔥 3. YOLOv11m — Real-Time Obstacle Detection  
**Folder:** `03_yolo/`  
**Main file:** `yolo_final_model.ipynb`

### 🎯 Purpose  
Detect real-time battlefield obstacles from drone/vision input.

### 📂 Dataset  
`01_data/yolo_dataset/` with:
- `images/`
- `labels/`
- Train/Val/Test splits  
- `data.yaml`

### 🏷 Classes (8 total)  
- `fire`  
- `explosion`  
- `road_collapse`  
- `bridge_collapse`  
- `ROK_soldier`  
- `DPRK_soldier`  
- `ROK_tank`  
- `DPRK_tank`

### 📊 Model Performance  
- **mAP50 = 0.952**  
- **mAP50-95 = 0.798**  
- Strong detection for structure-based hazards (collapse/tanks)  
- Fire/bridge-collapse slightly weaker due to data scarcity  

### 📤 Output  
- YOLO checkpoint `.pt`  
- Detection bounding boxes → added as hazard zones in simulation & Flask app

---

## 🔗 Model Integration Pipeline  
The three models operate sequentially:

```text
XGBoost → assigns initial road risk (risk_score_pred)
       ↓
GAT → learns relational risk patterns (gat_weight)
       ↓
Pathfinding (CCH, D*Lite, RTAA*) → uses:
    length + gat_weight + hazard_cost
       ↓
Flask App → real-time safe route generation
       ↓
YOLO → adds new hazards dynamically
```

---

## 📌 Folder Summary

| Folder | Purpose | Files |
|--------|---------|-------|
| **01_xgboost** | Structured risk prediction | xgboost_final.ipynb, xgboost_train_results.csv |
| **02_gat** | Graph Attention Network training | gat_final.ipynb |
| **03_yolo** | Obstacle detection | yolo_final_model.ipynb, YOLO .pt file |

---

## ✨ Notes  
- All models were trained independently and combined through graph-based routing.  
- For reproducibility, use original notebooks and ensure data paths match local structure.  
- Training may require GPU for YOLO and GAT.  
---