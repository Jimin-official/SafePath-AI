# ⚙️ Path Planning Algorithms  
**(CCH + A\*, D\* Lite, RTAA\*)**

This directory contains the core path-planning algorithms used in **SafePath-AI**, optimized for **real-time navigation**, **dynamic threats**, and **road-risk–aware routing** using GAT/XGBoost weights.

---

## 🚀 Overview  
SafePath-AI supports **three path-planning algorithms**, each serving different operational needs:

| Algorithm | Strength | Use Case |
|----------|----------|----------|
| **CCH + A\*** | Extremely fast static queries | Long-distance routing, frequent re-planning under moderate changes |
| **D\* Lite** | Optimal for dynamic obstacles | Environments where threats appear/disappear frequently |
| **RTAA\*** | Real-time heuristic local planning | Rapid updates under strict time limits |

Each algorithm integrates the project’s **risk-aware edge weights**:

edge_weight = length + gat_weight + ALPHA * danger_cost


---

# 🧩 1. CCH + A\*  
### **Contraction Hierarchies + A\***  
- Preprocesses the graph by contracting low-priority nodes  
- Creates **shortcut edges** to accelerate search  
- After preprocessing, A\* queries become **extremely fast (ms-level)**  
- In SafePath-AI, CCH is re-customized whenever danger zones change  
- Best suited for **large road networks (OSM)**

### 📁 Files  
- `CCH_Astar/`
  - `CCHAstar_with_GAT.ipynb` — Full CCH pipeline + interactive danger zones

### ✨ Key Features  
- Supports GAT/XGBoost risk integration  
- Rebuilds hierarchy when danger signature changes  
- A\* expanded-node/elapsed-time metrics collected automatically  
- Interactive danger placement with mouse click

---

# 🧩 2. D\* Lite  
### **Dynamic A\* for Changing Environments**  
- One of the most widely used dynamic-planning algorithms (used in NASA robots)  
- Efficiently updates paths **only where the graph changed**  
- Perfect for SafePath-AI because **danger zones are user-driven and unpredictable**

### 📁 Files  
- `DStar_Lite/`
  - `Dstarlite_with_GAT_final.ipynb` — GAT risk + dynamic obstacles + path recalculation

### ✨ Key Features  
- Supports rapid re-planning without recalculating entire map  
- Integrates hazard zones into edge weights  
- Tracks metrics:
  - success rate  
  - expanded nodes  
  - hazard avoidance  
  - cumulative GAT/risk-based cost  

---

# 🧩 3. RTAA\*  
### **Real-Time Adaptive A\***  
- Designed for **real-time constraints** (limited planning time per step)  
- Does not compute full path at once  
- Instead, performs:
  - local A\* search  
  - moves forward  
  - updates heuristics  
  - repeats  
- Ideal for fast-moving agents (e.g., vehicle/robot) needing instant response

### 📁 Files  
- `RTAAstar/`
  - `RTAAstar_with_GAT.ipynb` — RTAA* implementation + GAT risk + interactive hazards

### ✨ Key Features  
- Highly responsive in dynamic conditions  
- Adds risk-aware heuristics  
- Supports user-driven hazard placement  
- Monitors:
  - deviation from previous plan  
  - replanning frequency  
  - path stability  

---

# 📊 Comparative Summary

| Metric | CCH + A* | D* Lite | RTAA* |
|--------|----------|---------|--------|
| Preprocessing | ✔ Yes (heavy) | ❌ No | ❌ No |
| Dynamic Obstacle Handling | △ Medium (fast rebuild) | ⭐ Excellent | ⭐ Excellent |
| Real-time Response | Medium | High | **Very High** |
| Best Use | Large maps w/ many queries | Hazard-rich environments | Real-time navigation |
| Integration with GAT/XGBoost | ✔ | ✔ | ✔ |

---

# 🧠 Why Multiple Algorithms?

SafePath-AI simulates **realistic battlefield logistics**, where:

- threats may suddenly appear, disappear, or expand  
- roads collapse or burn  
- enemy units move  
- rerouting must be both **fast** and **safe**

No single algorithm is optimal for all conditions.

Using three allows:

- **CCH + A\*** → highest speed for stable maps  
- **D\* Lite** → robust re-planning for changing hazards  
- **RTAA\*** → instant decisions for real-time motion

---

# 📎 References  
- Koenig & Likhachev (2002). *D\* Lite.*  
- Geisberger et al. (2008). *Contraction Hierarchies.*  
- Korf & Barley (1993). *Real-Time Heuristic Search.*

---

# ✔ Done  
This README explains what each algorithm does, why it exists, and how it fits into SafePath-AI.


