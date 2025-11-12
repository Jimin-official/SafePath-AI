# -*- coding: utf-8 -*-
# ======================= 기본 임포트 =======================
import os, pickle, math, heapq, time
import osmnx as ox
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
from geopy.distance import geodesic
from contextlib import contextmanager
from collections import OrderedDict
import atexit

# ===== GUI 백엔드 (Windows) =====
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 캐시 활성화 
ox.settings.use_cache = True   # 다운로드한 OSM 데이터 캐싱
ox.settings.log_console = False

# 타일 캐시 디렉토리 지정 (프로젝트 안에 생성됨)
cx.set_cache_dir(os.path.join(os.getcwd(), "tile_cache"))

# ========= 위험 유형별 반경/색상/가중치 ==========
DANGER_RADIUS_METERS = {
    'road_collapse': 10,
    'bridge_collapse': 10,
    'tanks': 300,
    'enemies': 200,
    'fire': 300,
    'explosion': 300,
    'barbed_wire': 10,
    'rockfall': 10
}
DANGER_COLORS = {
    'road_collapse': 'red',
    'bridge_collapse': 'darkred',
    'tanks': 'brown',
    'enemies': 'orange',
    'fire': 'purple',
    'explosion': 'black',
    'barbed_wire': 'green',
    'rockfall': 'pink'
}
RISK_WEIGHTS = {
    'road_collapse': 1_00_000_000,
    'bridge_collapse': 1_00_000_000,
    'tanks': 1_000_000,
    'enemies': 500_000,
    'fire': 300_000,
    'explosion': 1_000_000,
    'barbed_wire': 1_00_000_000,
    'rockfall': 1_00_000_000
}
ALPHA = 1.0  # 위험 반영 계수

# ========= 전역 상태 ==========
selected_danger_type = None
active_danger_zones = []
danger_circle_artists = []
affected_edge_artists = []
current_robot_node = None
cch = None
ax = None
G = None

# ========= 휴리스틱 (지오데식) ==========
def heuristic(u, v):
    u_coord = (G.nodes[u]['y'], G.nodes[u]['x'])
    v_coord = (G.nodes[v]['y'], G.nodes[v]['x'])
    return geodesic(u_coord, v_coord).meters

# ========= 엣지 가중치(위험+GAT) ==========
def get_custom_weight(G, danger_zones):
    def weight_fn(u, v, d):
        base = d.get("length", 1.0)
        risk_score = 0.0
        # 엣지 중점 기준 위험 반경 체크
        ex = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        ey = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
        for z in danger_zones:
            cx, cy = z['center_lon'], z['center_lat']
            dist = ox.distance.great_circle(cy, cx, ey, ex)

            if dist <= z['effective_radius_meters']:
                count = z.get('count', 1)  # 1. zone에서 count 값을 가져옵니다.
                risk_score += RISK_WEIGHTS.get(z['type'], 0.0) * count # 2. risk_score에 count를 곱해서 더합니다.
        gat_weight = float(d.get('gat_weight', 0.0))
        return base + ALPHA * risk_score + gat_weight
    return weight_fn

# ========= 카운팅 A* (확장노드/시간) ==========
def astar_with_counts(Gs, s, t, heuristic, weight_attr='weight'):
    start_ms = time.time()
    openq = []
    heapq.heappush(openq, (0.0, s))
    came = {s: None}
    g = {s: 0.0}
    expanded = 0
    closed = set()

    while openq:
        _, u = heapq.heappop(openq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == t:
            # 경로 복원
            path = []
            cur = t
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            ms = (time.time() - start_ms) * 1000.0
            return path, expanded, ms

        for v, data in Gs[u].items():
            w = data.get(weight_attr, 1.0)
            cand = g[u] + w
            if cand < g.get(v, float('inf')):
                g[v] = cand
                f = cand + heuristic(v, t)
                heapq.heappush(openq, (f, v))
                came[v] = u

    ms = (time.time() - start_ms) * 1000.0
    raise nx.NetworkXNoPath(f"No path from {s} to {t} (expanded={expanded}, {ms:.1f}ms)")

# ========= CCH (단순화 버전) ==========
class CCH:
    def __init__(self, G):
        self.G = G
        self.order = []
        self.rank = {}
        self.shortcuts = nx.DiGraph()
        self.weight_fn = lambda u, v, d: d.get('length', 1.0)
        self._last_sig = None  # 위험존 변경 감지용

    def _danger_signature(self, zones):
        return tuple(sorted(
            (z['type'], round(z['center_lon'], 5), round(z['center_lat'], 5), int(z['effective_radius_meters']))
            for z in zones
        ))

    def _edge_cost(self, u, v):
        """MultiDiGraph 안전하게 가중치 뽑기"""
        data = self.G.get_edge_data(u, v, default=None)
        if data is None:
            return float('inf')
        # MultiDiGraph일 때: key -> dict 구조
        if isinstance(data, dict) and all(isinstance(val, dict) for val in data.values()) and 'length' not in data:
            return min(self.weight_fn(u, v, attr) for attr in data.values())
        return self.weight_fn(u, v, data)

    def build_hierarchy(self):
        self.order = sorted(self.G.nodes(), key=lambda n: len(self.G[n]))
        self.rank = {n: i for i, n in enumerate(self.order)}
        self.shortcuts.clear()

        for v in self.order:
            nbrs = list(self.G[v])
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    u, w = nbrs[i], nbrs[j]
                    if not (self.G.has_edge(u, v) and self.G.has_edge(v, w)):
                        continue
                    if self.rank[u] < self.rank[v] and self.rank[w] < self.rank[v]:
                        continue
                    c1 = self._edge_cost(u, v)
                    c2 = self._edge_cost(v, w)
                    tot = c1 + c2
                    if self.shortcuts.has_edge(u, w):
                        if tot < self.shortcuts[u][w]['weight']:
                            self.shortcuts[u][w]['weight'] = tot
                    else:
                        self.shortcuts.add_edge(u, w, weight=tot)

        # 원래 엣지도 포함 (Multi/DiGraph 둘 다)
        if isinstance(self.G, (nx.MultiGraph, nx.MultiDiGraph)):
            for u, w, key, data in self.G.edges(keys=True, data=True):
                wgt = self.weight_fn(u, w, data)
                if self.shortcuts.has_edge(u, w):
                    if wgt < self.shortcuts[u][w]['weight']:
                        self.shortcuts[u][w]['weight'] = wgt
                else:
                    self.shortcuts.add_edge(u, w, weight=wgt)
        else:
            for u, w, data in self.G.edges(data=True):
                wgt = self.weight_fn(u, w, data)
                if self.shortcuts.has_edge(u, w):
                    if wgt < self.shortcuts[u][w]['weight']:
                        self.shortcuts[u][w]['weight'] = wgt
                else:
                    self.shortcuts.add_edge(u, w, weight=wgt)

    def customize(self, weight_fn, danger_zones, mode="fast"):
        self.weight_fn = weight_fn
        sig = self._danger_signature(danger_zones)

        # 위험이 없고, GAT도 변화 없으면 스킵 (초기 속도 ↑)
        if not danger_zones and sig == self._last_sig:
            return
        self._last_sig = sig

        if mode == "exact":
            # 기존 방식: 모든 숏컷 A* 재평가 (느림)
            for u, v in list(self.shortcuts.edges()):
                try:
                    cost = nx.astar_path_length(self.G, u, v, heuristic=heuristic, weight=weight_fn)
                except nx.NetworkXNoPath:
                    cost = float('inf')
                self.shortcuts[u][v]['weight'] = cost
        else:
            # FAST: 수축 당시와 같이 두 엣지 합으로 재구성 (매우 빠름)
            self.build_hierarchy()


    def query(self, s, t):
        path, expanded, ms = astar_with_counts(self.shortcuts, s, t, heuristic, 'weight')
        return path, expanded, ms

# ========= 보조 함수 (반경/위경 변환, 위험검사, 길이/GAT) ==========
def convert_radius_meters_to_degrees(radius_meters, lat):
    lat_deg = radius_meters / 111139.0
    lon_deg = radius_meters / (111139.0 * math.cos(math.radians(lat)))
    return lat_deg, lon_deg

def edge_midpoint_xy(G, u, v):
    return ((G.nodes[u]['x'] + G.nodes[v]['x']) / 2.0,
            (G.nodes[u]['y'] + G.nodes[v]['y']) / 2.0)

def edge_is_in_any_hazard(G, u, v, danger_zones):
    ex, ey = edge_midpoint_xy(G, u, v)
    for z in danger_zones:
        cx, cy = z['center_lon'], z['center_lat']
        if ox.distance.great_circle(cy, cx, ey, ex) <= z['effective_radius_meters']:
            return True, z['type']
    return False, None

def edge_base_len_and_gat(G, u, v):
    data = G.get_edge_data(u, v, default={})
    if isinstance(data, dict) and 0 in data:  # MultiGraph 대응
        data = data[0]
    base = float(data.get('length', 0.0))
    gat = float(data.get('gat_weight', 0.0))
    return base, gat

# ========= 위험 유형 선택/그리기 ==========
def select_danger_type_prompt():
    global selected_danger_type
    print("\n위험 유형 선택:")
    print("1. road_collapse  2. bridge_collapse  3. tanks  4. enemies")
    print("5. fire           6. explosion        7. barbed_wire  8. rockfall")
    print("9. 선택 해제")
    choice = input("선택 (1~9): ")
    selected_danger_type = {
        '1': 'road_collapse', '2': 'bridge_collapse', '3': 'tanks', '4': 'enemies',
        '5': 'fire', '6': 'explosion', '7': 'barbed_wire', '8': 'rockfall', '9': None
    }.get(choice, None)
    print(f"선택된 위험: {selected_danger_type}")

def redraw_danger_zones(G, ax):
    global danger_circle_artists, affected_edge_artists
    for artist in danger_circle_artists + affected_edge_artists:
        try:
            artist.remove()
        except Exception:
            pass
    danger_circle_artists.clear()
    affected_edge_artists.clear()

    for z in active_danger_zones:
        cx, cy = z['center_lon'], z['center_lat']
        radius = z['effective_radius_meters']
        color = DANGER_COLORS.get(z['type'], 'gray')

        lat_off, lon_off = convert_radius_meters_to_degrees(radius, cy)
        circle_lons, circle_lats = [], []
        for i in range(100):
            ang = 2 * math.pi * i / 100
            circle_lons.append(cx + lon_off * math.cos(ang))
            circle_lats.append(cy + lat_off * math.sin(ang))
        h, = ax.plot(circle_lons, circle_lats, c=color, ls='--', lw=2, zorder=7)
        danger_circle_artists.append(h)

# ========= 마우스/키보드 이벤트 ==========
def on_click(event):
    global selected_danger_type, active_danger_zones, cch, ax
    if event.inaxes != ax:
        return
    if event.button == 1 and selected_danger_type:
        cx0, cy0 = event.xdata, event.ydata
        r = DANGER_RADIUS_METERS[selected_danger_type]
        active_danger_zones.append({
            'type': selected_danger_type,
            'center_lon': cx0,
            'center_lat': cy0,
            'effective_radius_meters': r
        })
        print(f"➡ 위험 추가: {selected_danger_type} @ ({cx0:.5f}, {cy0:.5f})")
        cch.customize(get_custom_weight(G, active_danger_zones),active_danger_zones, mode="fast")
        redraw_danger_zones(G, ax)
        plt.draw()
    elif event.button == 3:
        print("🧹 위험 초기화")
        active_danger_zones.clear()
        cch.customize(get_custom_weight(G, active_danger_zones),active_danger_zones, mode="fast")
        redraw_danger_zones(G, ax)
        plt.draw()

def on_key(event):
    if event.key == 't':
        select_danger_type_prompt()

# ========= GAT 가중치 주입 ==========
def add_gat_weights_to_graph(G, gat_df):
    print("GAT 위험도 가중치 추가 중...")
    risk_lookup = {
        (int(row['u']), int(row['v'])): float(row['risk_score_pred'])
        for _, row in gat_df.iterrows()
    }
    for u, v, data in G.edges(data=True):
        data['gat_weight'] = float(risk_lookup.get((u, v), 0.0))
    print("GAT 가중치 추가 완료.")

# ========= 메인 ==========
if __name__ == '__main__':
    GRAPH_PATH = "Seoul_graph.pkl"

    # ---- 그래프 로딩/생성 ----
    if os.path.exists(GRAPH_PATH):
        print("저장된 그래프 불러오는 중...")
        with open(GRAPH_PATH, 'rb') as f:
            G = pickle.load(f)
    else:
        print("강남구 그래프 다운로드 중...")
        G = ox.graph_from_place(["Eunpyeong-gu, South Korea","Seodaemun-gu, South Korea","Mapo-gu, South Korea"], network_type='drive')
        print("그래프 저장 중...")
        with open(GRAPH_PATH, 'wb') as f:
            pickle.dump(G, f)
    print(f"노드 수: {len(G.nodes)} | 엣지 수: {len(G.edges)}")

    # ---- GAT CSV (옵션) ----
    try:
        gat_df = pd.read_csv(r"static/GAT.csv")
        add_gat_weights_to_graph(G, gat_df)
    except Exception as e:
        print(f"[알림] GAT CSV 미사용({e}). 모든 'gat_weight' = 0.0")
        for _, _, d in G.edges(data=True):
            d['gat_weight'] = 0.0

    # ---- 출발/도착 ----
    s_lat, s_lon = 37.525208, 127.035256 # 도산공원 
    g_lat, g_lon = 37.490250, 127.061751 # 양재천길
    start = ox.distance.nearest_nodes(G, s_lon, s_lat)
    goal  = ox.distance.nearest_nodes(G, g_lon, g_lat)
    current_robot_node = start

    # ---- CCH ----
    cch = CCH(G)
    cch.build_hierarchy()
    cch.customize(get_custom_weight(G, active_danger_zones), active_danger_zones, mode="fast")

    # ---- 시각화 ----
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='gray', edge_linewidth=0.5)
    cx.add_basemap(ax, crs=G.graph['crs'], source=cx.providers.OpenStreetMap.Mapnik, zoom=12)
    ax.scatter(G.nodes[start]['x'], G.nodes[start]['y'], c='lime', s=200, label='출발')
    ax.scatter(G.nodes[goal]['x'],  G.nodes[goal]['y'],  c='purple', s=200, label='도착')
    robot_line, = ax.plot([], [], c='blue', lw=4, label='로봇 이동')
    plan_line,  = ax.plot([], [], c='orange', lw=2, label='현재 계획')
    robot_marker = ax.scatter([], [], c='cyan', s=150, marker='s', label='현재 위치')
    ax.legend()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.ion()
    plt.show()

    # ======== 메트릭 누적 변수 ========
    total_search_ms = 0.0
    total_expanded  = 0
    total_path_len_m = 0.0
    hazard_edges = 0
    safe_edges   = 0
    cum_risk_cost = 0.0
    cum_gat_cost  = 0.0
    replan_count  = 0
    deviation_sum = 0.0
    plan_stability_kept = 0
    prev_plan = None

    # ---- 시뮬 루프 ----
    robot_path = [start]
    cur = start
    step = 0
    max_steps = 150

    while cur != goal and step < max_steps:
        step += 1
        print(f"\n[STEP {step}] 현재 위치: {cur}")

        # 경로 질의 (탐색 시간/확장 노드 수 획득)
        try:
            path, expanded, ms = cch.query(cur, goal)
        except nx.NetworkXNoPath:
            print("❌ [경로 없음] 위험 요소 때문에 목적지 도달 불가")
            while True:
                resp = input("계속 시도하시겠습니까? (y/n): ").lower()
                if resp == 'y':
                    print("⏸️ 대기 중... 위험 조정 후 재계산")
                    plt.pause(2)
                    break
                elif resp == 'n':
                    print("🛑 시뮬레이션 종료")
                    plt.ioff(); plt.close()
                    exit()
                else:
                    print("❗ y 또는 n 입력")
            continue

        total_search_ms += ms
        total_expanded  += expanded

        # 재계획 감지/편차
        if prev_plan is not None and len(prev_plan) > 1 and len(path) > 1:
            if prev_plan[1] == path[1]:
                plan_stability_kept += 1
            else:
                replan_count += 1
                K = min(10, len(prev_plan), len(path))
                diff = sum(1 for i in range(K) if prev_plan[i] != path[i])
                deviation_sum += diff / max(1, K)
        prev_plan = path

        if len(path) < 2:
            print("✅ 더 이상 이동할 노드 없음.")
            break

        next_node = path[1]

        # 이동 엣지 메트릭 누적
        in_hazard, hz_type = edge_is_in_any_hazard(G, cur, next_node, active_danger_zones)
        base_len, gat_w = edge_base_len_and_gat(G, cur, next_node)
        total_path_len_m += base_len
        cum_gat_cost += gat_w
        if in_hazard:
            hazard_edges += 1
            if hz_type is not None:
                cum_risk_cost += RISK_WEIGHTS.get(hz_type, 0.0)
        else:
            safe_edges += 1

        # 사용자 경고/진행 여부(위험 엣지일 때)
        if in_hazard:
            print(f"⚠️ 위험 엣지 탐지 ({cur} ➔ {next_node}) → {hz_type}")
            while True:
                ans = input("이 엣지를 통과하시겠습니까? (y/n): ").lower()
                if ans == 'y':
                    print("➡️ 통과합니다.")
                    break
                elif ans == 'n':
                    print("⏸️ 대기. 위험 제거 또는 경로 변경을 기다립니다.")
                    plt.pause(2)
                    step -= 1
                    next_node = None
                    break
                else:
                    print("❗ y 또는 n 입력만 허용")
            if next_node is None:
                continue

        # 실제 이동
        cur = next_node
        robot_path.append(cur)
        print(f"➡️ 이동: {robot_path[-2]} → {cur}")

        # 시각화 업데이트
        rx = [G.nodes[n]['x'] for n in robot_path]
        ry = [G.nodes[n]['y'] for n in robot_path]
        robot_line.set_data(rx, ry)
        px = [G.nodes[n]['x'] for n in path]
        py = [G.nodes[n]['y'] for n in path]
        plan_line.set_data(px, py)
        robot_marker.set_offsets((G.nodes[cur]['x'], G.nodes[cur]['y']))
        ax.set_title(f"STEP {step}: 현재 {cur}")
        redraw_danger_zones(G, ax)
        plt.draw()
        plt.pause(0.8)

        if cur == goal:
            print("🎉 목표 도달!")

    plt.ioff(); plt.show()

