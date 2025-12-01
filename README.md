# 🚀 SafePath-AI  
**AI-Based Real-Time Route Planning System for Securing Military Supply Lines in Wartime**

_Hanwha Aerospace Smart Defense Data Analysis Bootcamp · Final Project_

---

## 📘 Overview

Ensuring stable logistics and supply lines is one of the most crucial factors in modern warfare.  
SafePath-AI integrates drone-based obstacle detection, road-level risk modeling, and intelligent pathfinding algorithms to generate **the safest and fastest route for military transport in real time**.

This system combines:

- **YOLOv11m** — Real-time battlefield obstacle detection  
- **XGBoost & Graph Attention Network (GAT)** — Road baseline risk prediction  
- **Three pathfinding algorithms:** D\* Lite, CCH + A\*, RTAA\*  
- **Flask-based interactive web application**

---

## 🎯 Objectives

- Detect battlefield obstacles from drone imagery  
- Predict road-level baseline danger using accident and road attribute data  
- Recalculate optimal routes when new threats appear  
- Compare pathfinding algorithms across multiple threat scenarios  
- Provide a user-interactive, simulation-ready web interface  

---

## 🗂️ Project Structure

```bash
SafePath-AI/
│
├── 01_data/
│   ├── road_network/
│   ├── yolo_dataset/
│
├── 02_models/
│   ├── 01_xgboost/
│   ├── 02_gat/
│   └── 03_yolo/
│
├── 03_algorithms/
│   ├── CCH_Astar/
│   ├── DStar_Lite/
│   └── RTAAstar/
│
├── 04_flask_app/
│   ├── algorithm/
│   ├── templates/
│   ├── static/
│   ├── app.py
│   ├── convert_geojson_to_wgs84.py
│   └── osmnx_geocode.py
│
├── 05_docs/
│   ├── presentation.pptx
│   └── 25_08_25_Project_Proposal.docx
│
└── README.md
```

---

## 🧩 System Architecture

### 1️⃣ Road Risk Prediction (XGBoost + GAT)

**Data Sources**
- OSMnx / Overpass API  
- TAAS accident dataset (Eunpyeong, Mapo, Seodaemun)

**Features (17 total)**  
`highway`, `length`, `lanes`, `slope`, `surface`, `bridge`, `tunnel`, etc.

**Pipeline**
1. **XGBoost** predicts structural accident-based risk  
2. Output used as pseudo-label for **GAT**  
3. GAT learns relational importance between adjacent roads  
4. Produces **baseline road risk score**

---

### 2️⃣ Obstacle Detection (YOLOv11m)

**8 Classes**
- Fire  
- Explosion  
- Road Collapse  
- Bridge Collapse  
- ROK Soldier / DPRK Soldier  
- ROK Tank / DPRK Tank  

**Model Performance**
- **mAP50 = 0.952**  
- **mAP50–95 = 0.798**

---

### 3️⃣ Real-Time Pathfinding Algorithms

| Algorithm | Strength | Notes |
|----------|----------|-------|
| **D\* Lite** | Most stable & best avoidance | Ideal for dynamic battlefield |
| **CCH + A\*** | Fastest search time | Suitable for large-scale maps |
| **RTAA\*** | High responsiveness | Good at rapid decisions |

Statistical tests (ANOVA + Tukey HSD) indicate **significant differences** in performance across all algorithms (p < 0.05).

---

## 🗺️ Region of Analysis

Seoul **Northwestern Zone (서북권)**:
- Eunpyeong-gu  
- Seodaemun-gu  
- Mapo-gu  

Chosen due to:
- Strategic Paju–Seoul corridor  
- High logistical relevance  
- Strong policy & spatial connectivity  

---

## 🧪 Key Results

### Road Risk Modeling
- Top 10% predicted danger explains **51% of real accidents** (Gains Curve)  
- GAT heatmaps match real urban road patterns  
- Successfully distinguishes arterial roads vs. residential areas  

### YOLO Detection
- Tanks & road collapses: near-perfect detection  
- Fire/bridge-collapse improved but require more samples  

### Pathfinding Evaluation
- **D\* Lite** → Best overall stability  
- **CCH + A\*** → Best efficiency & speed  
- **RTAA\*** → Fastest reaction to local changes  

---

## 🖥️ Flask Web Application

### Features
- Enter start/destination  
- Add obstacles interactively (map click)  
- YOLO detections reshapes route in real-time  
- Compare 3 algorithms visually  
- My Page: history, statistics, saved paths  

### Run Instructions

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the app**
```bash
python app.py
```

Open browser:
```
http://localhost:5000
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, YOLOv11m, DGL |
| ML | XGBoost |
| Geospatial | OSMnx, GeoPandas, Folium |
| Algorithms | D\* Lite, CCH+A\*, RTAA\* |
| Backend | Flask |
| Others | pandas, numpy |

---

## 🌱 Expected Impact

- Real-time safe route generation for wartime logistics  
- Supports R&D for unmanned/combined combat systems  
- Expandable to **civil disaster response**  
- Demonstrates feasibility of **AI-based tactical navigation**  

---

## 📚 References
- OSMnx, Overpass API  
- TAAS Accident Data  
- Ultralytics YOLO  
- Literature on D\* Lite, CCH, RTAA\* algorithms  
- Military logistics & battlefield mobility studies  

---

## ✨ Acknowledgements
- Mentor: 박도훈  
- Professor: 이경미  
