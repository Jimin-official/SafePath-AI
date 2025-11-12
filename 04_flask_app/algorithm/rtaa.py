# 아래 패키지들을 다운받으셔야합니다! 그냥 pip install 저것들 하면 저 버젼으로 받아지니까 그냥 pip install 하시면 됩니다
# jupyter 1.1.1
# numpy 2.0.2
# matplotlib 3.9.4
# contextily 1.6.2
# osmnx 2.0.5
# networkx 3.2.1
# sklearn 1.6.1
import osmnx as ox
import networkx as nx
import heapq
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc # 한글 폰트 설정을 위해 추가
import contextily as cx # contextily 라이브러리 임포트
import os
import pickle
import math
from geopy.distance import geodesic
import time
import pandas as pd

# GUI 백엔드 (윈도우/로컬 실행 환경)
matplotlib.use('TkAgg')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
# --- 한글 폰트 설정 끝 ---


# RoadNetworkMap 클래스 (변경 없음)
class RoadNetworkMap:
    def __init__(self, G):
        self.G = G
    def succ(self, u):
        return list(self.G.neighbors(u))
    def pred(self, u):
        return list(self.G.predecessors(u))
    def c(self, u, v):
        if not self.G.has_edge(u, v):
            return float('inf')
            
        GAT_RISK_MULTIPLIER = 100000  
        min_cost = float('inf')
        
        for key in self.G[u][v]:
            d = self.G[u][v][key]
            base_cost = d.get("length", 1)
            
            gat_risk_score = d.get("gat_weight", 0) # 1단계에서 추가한 속성
            gat_risk_cost = gat_risk_score * GAT_RISK_MULTIPLIER # 스케일링
            
            dynamic_danger_cost = (
                d.get("road_collapse", 0) * 1_00_000_000
                + d.get("bridge_collapse", 0) * 1_00_000_000
                + d.get("tanks", 0) * 1_000_000
                + d.get("enemies", 0) * 500_000
                + d.get("fire", 0) * 300_000
                + d.get("explosion", 0) * 1_000_000
                + d.get("barbed_wire",0) * 1_00_000_000
                + d.get("rockfall",0) * 1_00_000_000
            )
            current_cost = base_cost + gat_risk_cost + dynamic_danger_cost
            if current_cost < min_cost:
                min_cost = current_cost
        return min_cost if min_cost != float('inf') else float('inf')

# 휴리스틱 함수 (변경 없음)
def heuristic(a, b, G):
    y1, x1 = G.nodes[a]['y'], G.nodes[a]['x']
    y2, x2 = G.nodes[b]['y'], G.nodes[b]['x']
    return geodesic((y1, x1), (y2, x2)).meters

# RTAA* 클래스 (변경 없음)
class RTAAStar:
    def __init__(self, road_map, s_start, s_goal, N):
        self.road_map = road_map
        self.s_start = s_start
        self.s_goal = s_goal
        self.N = N
        self.s_current = s_start
        self.h_table = {}
        self.path = [self.s_current]
        self.last_closed_set = set()

    def init(self):
        print("RTAA* 플래너 초기화 중: 모든 노드의 휴리스틱 계산...")
        for node in self.road_map.G.nodes():
            self.h_table[node] = self.h(node)
        print("휴리스틱 테이블 초기화 완료.")

    def h(self, s):
        return heuristic(s, self.s_goal, self.road_map.G)

    def cost(self, s_start, s_goal):
        return self.road_map.c(s_start, s_goal)

    def get_neighbor(self, s):
        return self.road_map.succ(s)

    def Astar(self, x_start, N_limit):
        OPEN = [(self.h_table[x_start], x_start)] # 튜플형식으로(추정치, 노드)를 담는 우선순위 큐 추정치가 가장 작은 노드부터 처리
        CLOSED = set()
        g_table = {node: float('inf') for node in self.road_map.G.nodes()} # 실제 이동 비용을 저장하는 딕셔너리
        g_table[x_start] = 0 # 시작 부분은 0으로 초기화
        PARENT = {x_start: x_start} # 경로를 재구성하기 위해 각 노드의 이전 노드를 저장하는 딕셔너리
        count = 0 #  
        while OPEN and count < N_limit:
            count += 1
            f, s = heapq.heappop(OPEN)
            if s in CLOSED: continue
            CLOSED.add(s)
            if s == self.s_goal: return OPEN, CLOSED, g_table, PARENT
            for s_n in self.get_neighbor(s):
                new_cost = g_table[s] + self.cost(s, s_n)
                if new_cost < g_table[s_n]:
                    g_table[s_n] = new_cost
                    PARENT[s_n] = s
                    f_new = new_cost + self.h_table[s_n]
                    heapq.heappush(OPEN, (f_new, s_n))
        return OPEN, CLOSED, g_table, PARENT

    def update_h_values(self, OPEN, CLOSED, g_table):
        if not OPEN: return
        f_min = min(g_table[node] + self.h_table[node] for f_val, node in OPEN)
        for s in CLOSED:
            updated_h = f_min - g_table[s]
            if updated_h > self.h_table[s]:
                 self.h_table[s] = updated_h

    def get_next_move(self, current_node):
        best_neighbor, min_cost_to_go = None, float('inf')
        neighbors = self.get_neighbor(current_node)
        if not neighbors: return None
        for s_n in neighbors:
            cost_to_go = self.cost(current_node, s_n) + self.h_table[s_n]
            if cost_to_go < min_cost_to_go:
                min_cost_to_go, best_neighbor = cost_to_go, s_n
        return best_neighbor

    def extract_final_path(self, start_node, parent_dict):
        path = [self.s_goal]
        s = self.s_goal
        while s != start_node:
            try:
                s = parent_dict[s]
                path.append(s)
            except KeyError: return None
        return list(reversed(path))

    def search_step(self):
        if self.s_current == self.s_goal: return "GOAL"
        OPEN, CLOSED, g_table, PARENT = self.Astar(self.s_current, self.N)
        self.last_closed_set = CLOSED
        if self.s_goal in PARENT:
            final_path_segment = self.extract_final_path(self.s_current, PARENT)
            if final_path_segment:
                self.path.extend(final_path_segment[1:])
                self.s_current = self.s_goal
            return "GOAL"
        if not OPEN: return "STUCK"
        self.update_h_values(OPEN, CLOSED, g_table)
        next_node = self.get_next_move(self.s_current)
        if next_node is None: return "STUCK"
        self.s_current = next_node
        self.path.append(self.s_current)
        return "CONTINUE"
        
def add_gat_weights_to_graph(G, gat_risk_df):
    """
    Pandas DataFrame에 있는 GAT 위험도 점수를 NetworkX 그래프의 엣지 속성으로 추가합니다.
    (u, v) 쌍을 키로 하는 딕셔너리를 만들어 효율적으로 처리합니다.
    """
    print("그래프에 GAT 위험도 가중치를 추가하는 중...")
    
    # (u, v)를 키로, risk_score_pred를 값으로 하는 딕셔너리 생성 (빠른 조회를 위해)
    risk_lookup = {
        (int(row['u']), int(row['v'])): row['risk_score_pred']
        for _, row in gat_risk_df.iterrows()
    }

    # 그래프의 모든 엣지를 순회하며 'gat_weight' 속성 추가
    for u, v, data in G.edges(data=True):
        # 딕셔너리에서 해당 (u, v) 엣지의 GAT 위험도 조회
        # .get()을 사용하여 해당 엣지 위험도 점수가 없으면 0.0을 기본값으로 사용
        gat_risk = risk_lookup.get((u, v), 0.0)
        data['gat_weight'] = gat_risk
        
    print("GAT 위험도 가중치 추가 완료.")


# --- 전역 변수 및 헬퍼 함수 (대부분 변경 없음) ---
rtaa_planner = None 
active_danger_zones = []
danger_circle_artists = [] 
affected_edge_artists = [] 
selected_danger_type = None
rtaa_closed_set_marker = None
robot_marker = None
path_line = None

def convert_radius_meters_to_degrees(radius_meters, lat):
    lat_degree_diff = radius_meters / 111139.0
    lon_degree_diff = radius_meters / (111139.0 * math.cos(math.radians(lat)))
    return lat_degree_diff, lon_degree_diff

def select_danger_type_prompt():
    global selected_danger_type
    print("\n어떤 종류의 위험 지역을 추가하시겠습니까? (현재 선택: {})".format(selected_danger_type if selected_danger_type else "없음"))
    print("1. 도로 붕괴 | 2. 다리 붕괴 | 3. 탱크 | 4. 적군 | 5. 화재 | 6. 폭발 | 7. 철조망 | 8. 낙석 | 9. 선택 해제")
    choice = input("선택 (1-9): ")
    types = ['road_collapse', 'bridge_collapse', 'tanks', 'enemies', 'fire', 'explosion', 'barbed_wire', 'rockfall']
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(types):
            selected_danger_type = types[idx]
            print(f"{selected_danger_type} 위험 지역 추가 모드 활성화.")
        elif idx == 8:
            selected_danger_type = None
            print("위험 지역 선택 해제.")
    except (ValueError, IndexError):
        print("잘못된 선택입니다.")

def update_graph_risks(G_graph, danger_zones_list):
    for u, v, k, d in G_graph.edges(keys=True, data=True):
        d['road_collapse'], d['bridge_collapse'], d['tanks'], d['enemies'] = 0, 0, 0, 0
        d['fire'], d['explosion'], d['barbed_wire'], d['rockfall'] = 0, 0, 0, 0

    if not danger_zones_list: return

    for zone in danger_zones_list:
        center_lon, center_lat = zone['center_lon'], zone['center_lat']
        effective_radius_meters = zone['effective_radius_meters']
        danger_type = zone['type']

        count = zone.get('count', 1)  # 1. zone에서 count 값을 가져옵니다.

        for u, v, k, data in G_graph.edges(keys=True, data=True):
            u_x, u_y = G_graph.nodes[u]['x'], G_graph.nodes[u]['y']
            dist_u_to_center = ox.distance.great_circle(center_lat, center_lon, u_y, u_x)
            if dist_u_to_center <= effective_radius_meters:
                for edge_key in G_graph[u][v]: 
                    G_graph[u][v][edge_key][danger_type] += count # 2. 1 대신 count 값을 더합니다.
                    
def redraw_danger_zones(G_graph, ax_obj, danger_zones_list):
    global danger_circle_artists, affected_edge_artists
    danger_colors = {'road_collapse': 'saddlebrown', 'bridge_collapse': 'black', 'tanks': 'darkred', 'enemies': 'red', 'fire': 'orange', 'explosion':'yellow', 'barbed_wire': 'indigo', 'rockfall' : 'darkslategray'}
    for artist in danger_circle_artists + affected_edge_artists: artist.remove()
    danger_circle_artists.clear(); affected_edge_artists.clear()
    for zone in danger_zones_list:
        center_lon, center_lat, effective_radius_meters, danger_type = zone['center_lon'], zone['center_lat'], zone['effective_radius_meters'], zone['type']
        color = danger_colors.get(danger_type, 'gray')
        circle_lons, circle_lats = [], []
        for i in range(101):
            angle = 2 * math.pi * i / 100
            lat_offset, lon_offset = convert_radius_meters_to_degrees(effective_radius_meters, center_lat)
            circle_lons.append(center_lon + lon_offset * math.cos(angle))
            circle_lats.append(center_lat + lat_offset * math.sin(angle))
        circle_patch, = ax_obj.plot(circle_lons, circle_lats, c=color, ls='--', lw=2, zorder=7)
        danger_circle_artists.append(circle_patch)

def on_click(event):
    # 전역 변수 선언 부분은 동일
    global G, road_map, rtaa_planner, active_danger_zones, selected_danger_type, ax

    DANGER_RADIUS_METERS = {'road_collapse': 10, 'bridge_collapse': 10, 'tanks': 300, 'enemies': 200, 'fire': 300, 'explosion' : 300, 'barbed_wire' : 10, 'rockfall' : 10}
    if event.inaxes != ax: return
    
    if event.button == 1:
        if selected_danger_type is None:
            print("\n[알림] 먼저 콘솔에서 't'를 입력하여 위험 지역 유형을 선택해주세요.")
            return
        click_lon, click_lat = event.xdata, event.ydata
        effective_radius_meters = DANGER_RADIUS_METERS.get(selected_danger_type, 0)
        center_lon, center_lat = click_lon, click_lat
        if selected_danger_type in ['bridge_collapse', 'road_collapse', 'barbed_wire', 'rockfall']:
            u, v, _ = ox.distance.nearest_edges(G, click_lon, click_lat)
            center_lon, center_lat = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2, (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            
        active_danger_zones.append({'type': selected_danger_type, 'center_lon': center_lon, 'center_lat': center_lat, 'effective_radius_meters': effective_radius_meters})
        print(f"새로운 '{selected_danger_type}' 위험 지역 추가됨. 로봇이 실시간으로 경로에 반영합니다.")
        
        # 1. 그래프의 위험 정보 업데이트 (기존과 동일)
        update_graph_risks(G, active_danger_zones)
        
        # ✅ 2. 플래너에 변경사항 알리기 (이 부분이 핵심!)
        # 휴리스틱 테이블을 새로운 비용에 맞춰 다시 계산하도록 지시
        if rtaa_planner:
            print("그래프 비용 변경 감지: RTAA* 휴리스틱 테이블을 다시 초기화합니다.")
            rtaa_planner.init() 
        
        # 3. 시각화 업데이트 (기존과 동일)
        redraw_danger_zones(G, ax, active_danger_zones)
        
    elif event.button == 3:
        print("모든 위험 지역을 초기화합니다.")
        active_danger_zones.clear()
        update_graph_risks(G, active_danger_zones)
        
        # ✅ 플래너도 초기화된 비용에 맞춰 다시 계산
        if rtaa_planner:
            print("그래프 초기화 감지: RTAA* 휴리스틱 테이블을 다시 초기화합니다.")
            rtaa_planner.init()
            
        redraw_danger_zones(G, ax, active_danger_zones)


# 키보드 이벤트 핸들러: 't' 키만 처리하도록 수정
def on_key(event):
    if event.key == 't':
        select_danger_type_prompt()

############################################
# ✅ 메인 시뮬레이션
############################################
if __name__ == "__main__":
    GRAPH_PATH = "Seoul_graph.pkl"

    # 그래프 로딩 및 초기화 (변경 없음)
    if os.path.exists(GRAPH_PATH):
        print("저장된 그래프 불러오는 중...")
        with open(GRAPH_PATH, "rb") as f: G = pickle.load(f)
    else:
        print("서울 그래프 다운로드 중...")
        G = ox.graph_from_place(["Eunpyeong-gu, South Korea","Seodaemun-gu, South Korea","Mapo-gu, South Korea"], network_type='drive')
        print("그래프 저장 중...")
        with open(GRAPH_PATH, "wb") as f: pickle.dump(G, f)

    try:
        gat_df = pd.read_csv(r"static/GAT.csv")
        add_gat_weights_to_graph(G, gat_df)
    except FileNotFoundError:
        print("[경고] GAT 위험도 파일(GAT.csv)을 찾을 수 없습니다.")
        print("모든 엣지의 'gat_weight'를 0으로 설정합니다.")
        for u, v, data in G.edges(data=True):
            data['gat_weight'] = 0.0  

    
    for u, v, k, d in G.edges(keys=True, data=True):
        d['road_collapse'], d['bridge_collapse'], d['tanks'], d['enemies'] = 0, 0, 0, 0
        d['fire'], d['explosion'], d['barbed_wire'], d['rockfall'] = 0, 0, 0, 0
        
            
    road_map = RoadNetworkMap(G)
    s_lat, s_lon = 37.525208, 127.035256 # 도산공원 
    g_lat, g_lon = 37.490250, 127.061751 # 양재천길

    start = ox.distance.nearest_nodes(G, s_lon, s_lat)
    goal = ox.distance.nearest_nodes(G, g_lon, g_lat)

    # RTAA* 플래너 초기화
    rtaa_planner = RTAAStar(road_map, start, goal, N=1000)
    rtaa_planner.init()

    # 시각화 설정 (변경 없음)
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color="gray", edge_linewidth=0.5)
    cx.add_basemap(ax, crs=G.graph['crs'], source=cx.providers.CartoDB.Positron)
    ax.scatter(G.nodes[start]['x'], G.nodes[start]['y'], c='lime', s=100, zorder=10, label='Start')
    ax.scatter(G.nodes[goal]['x'], G.nodes[goal]['y'], c='red', s=100, marker='*', zorder=10, label='Goal')
    rtaa_closed_set_marker = ax.scatter([], [], c='cyan', s=10, alpha=0.5, zorder=3, label='RTAA* Explored')
    robot_marker = ax.scatter(G.nodes[start]['x'], G.nodes[start]['y'], c='blue', s=80, zorder=11, label='Robot')
    path_line, = ax.plot([], [], color='blue', linewidth=3, alpha=0.8, zorder=4, label='Path Taken')
    ax.legend()
    
    # 이벤트 핸들러 연결
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)

    print("\n--- RTAA* 자동 시뮬레이션 시작 ---")
    print("로봇이 목표 지점까지 자동으로 이동합니다.")
    print("진행 중에 't'를 눌러 위험 유형을 선택하고, 마우스로 위험 지역을 추가할 수 있습니다.")
    
    # 자동 진행 루프
    while True:
        status = rtaa_planner.search_step()
        
        # 실시간 시각화 업데이트
        robot_node = rtaa_planner.s_current
        robot_x, robot_y = G.nodes[robot_node]['x'], G.nodes[robot_node]['y']
        robot_marker.set_offsets([robot_x, robot_y])

        path_nodes = rtaa_planner.path
        if len(path_nodes) > 1:
            path_line.set_data([G.nodes[n]['x'] for n in path_nodes], [G.nodes[n]['y'] for n in path_nodes])

        closed_set_nodes = rtaa_planner.last_closed_set
        if closed_set_nodes:
            rtaa_closed_set_marker.set_offsets([(G.nodes[n]['x'], G.nodes[n]['y']) for n in closed_set_nodes])
        
        # 화면을 새로고침하고 이벤트를 처리
        plt.pause(0.05) # 0.05초 간격으로 진행, 이 값이 작을수록 애니메이션이 빨라짐

        if status == "GOAL":
            print("목표 지점에 도달했습니다!")
            break
        elif status == "STUCK":
            print("경로를 찾을 수 없습니다. (막힌 경로)")
            break
            
    print("시뮬레이션이 종료되었습니다. 창을 닫아주세요.")
    plt.show() # 시뮬레이션 종료 후에도 창을 계속 보여줌