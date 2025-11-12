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
import pandas as pd

# GUI 백엔드 (윈도우/로컬 실행 환경)
matplotlib.use('TkAgg')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
# --- 한글 폰트 설정 끝 ---


############################################
# ✅ Priority Queue (D* Lite용)
############################################
class Priority:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
    def __lt__(self, other):
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 < other.k2)

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.vertices_in_heap = set() # 힙에 있는 노드 추적 (중복 방지)
        self.vertex_map = {} # 노드 ID -> (priority, vertex) 튜플의 인덱스 매핑 (실제 D* Lite 구현에서는 잘 사용 안 함, 효율성 때문에)

    def insert(self, vertex, priority):
        # 이미 힙에 있는 경우 업데이트 처리 (여기서는 삭제 후 삽입으로 대체)
        if vertex in self.vertices_in_heap:
            self.remove(vertex)
        heapq.heappush(self.heap, (priority, vertex))
        self.vertices_in_heap.add(vertex)

    def top(self):
        # 힙에서 삭제된 항목이 있을 수 있으므로 실제 힙에서 검증 필요 (remove가 비효율적이므로 일단 현재 로직 유지)
        while self.heap and self.heap[0][1] not in self.vertices_in_heap:
            heapq.heappop(self.heap)
        return self.heap[0][1] if self.heap else None

    def top_key(self):
        while self.heap and self.heap[0][1] not in self.vertices_in_heap:
            heapq.heappop(self.heap)
        return self.heap[0][0] if self.heap else Priority(float('inf'), float('inf'))

    def remove(self, vertex):
        if vertex not in self.vertices_in_heap:
            return
        # Python heapq는 특정 항목을 효율적으로 제거하는 기능을 제공하지 않음
        # 비효율적이지만, 현재 구현에서는 리스트 필터링 후 heapify
        self.heap = [(p,v) for p,v in self.heap if v!=vertex]
        heapq.heapify(self.heap) # O(N)
        self.vertices_in_heap.discard(vertex)

    def update(self, vertex, priority):
        # 실제 D* Lite에서는 더 효율적인 업데이트 방법을 사용 (e.g., lazy deletion)
        # 여기서는 기존 항목 제거 후 새 우선순위로 삽입
        self.remove(vertex)
        self.insert(vertex, priority)

    def empty(self):
        # 힙에 남아있지만 제거 플래그가 설정된 항목들을 제거
        while self.heap and self.heap[0][1] not in self.vertices_in_heap:
            heapq.heappop(self.heap)
        return len(self.heap)==0


# RoadNetworkMap 클래스
class RoadNetworkMap:
    def __init__(self, G):
        self.G = G
        # YOLO 탐지 결과를 저장할 딕셔너리를 초기화합니다.
        self.yolo_detections = {}

    def succ(self, u):
        return list(self.G.neighbors(u))

    def pred(self, u):
        return list(self.G.predecessors(u))

    def c(self, u, v):
        """
        u에서 v로 가는 비용을 계산합니다.
        수동으로 설정된 위험 지역(스위치 역할) 내에서
        YOLO가 실시간으로 탐지한 객체 수(배수 역할)를 반영하여 비용을 동적으로 조절합니다.
        """
        if not self.G.has_edge(u, v):
            return float('inf')

        # --- 위험 요소별 기본 가중치 ---
        DANGER_WEIGHTS = {
            "tanks": 1_000_000,
            "enemies": 500_000,
            "fire": 300_000,
            "explosion": 1_000_000,
            "road_collapse": 100_000_000,
            "bridge_collapse": 100_000_000,
            "barbed_wire": 100_000_000,
            "rockfall": 100_000_000
        }

        # MultiDiGraph는 u, v 사이에 여러 엣지가 있을 수 있으므로, 최소 비용을 선택합니다.
        min_cost = float('inf')
        for key in self.G[u][v]:
            edge_data = self.G[u][v][key]

            # --- 1. 기본 비용 (도로의 물리적 길이) ---
            base_cost = edge_data.get("length", 1)

            # --- 2. (선택적) GAT 예측 위험도 비용 ---
            gat_risk_cost = edge_data.get("gat_weight", 0) * 100000

            # --- 3. 동적 위험 비용 계산 ---
            dynamic_danger_cost = 0

            # 모든 위험 유형에 대해 반복
            for danger_type, base_weight in DANGER_WEIGHTS.items():
                # 이 엣지가 해당 위험 유형의 '위험 지역'으로 설정되어 있는지 확인합니다. (값이 0보다 크면 '설정됨')
                # 이 플래그가 일종의 '스위치' 역할을 합니다.
                is_in_danger_zone = edge_data.get(danger_type, 0) > 0

                if is_in_danger_zone:
                    # '스위치'가 켜진 경우, YOLO가 탐지한 객체 수를 가져옵니다.
                    # 탐지된 객체가 없으면 0, YOLO 정보가 아예 없으면 기본값 1을 사용합니다.
                    # (YOLO 정보가 없을 때도 수동으로 추가한 위험 지역은 기본 비용을 가지게 하기 위함)
                    yolo_multiplier = self.yolo_detections.get(danger_type, 1)

                    # 최종 비용 = 기본 가중치 * YOLO 배수
                    dynamic_danger_cost += base_weight * yolo_multiplier

            # --- 4. 최종 비용 합산 ---
            current_cost = base_cost + gat_risk_cost + dynamic_danger_cost
            # current_cost = base_cost + dynamic_danger_cost # GAT를 사용하지 않을 경우
            
            if current_cost < min_cost:
                min_cost = current_cost
                
        return min_cost


    def set_dynamic_risk(self,u,v, danger_type, value):
        # 특정 엣지의 특정 위험 속성만 업데이트
        if self.G.has_edge(u,v):
            for key in self.G[u][v]:
                self.G[u][v][key][danger_type] = value


############################################
# ✅ 휴리스틱: 유클리드 거리
############################################
def heuristic(a,b,G):
    # 위도 경도에 기반한 유클리드 거리
    # 더 정확하게는 haversine 거리 함수를 사용하는 것이 좋지만, 계산 비용 및 D* Lite의 목적에 맞춰 단순화
    x1, y1 = G.nodes[a]['x'], G.nodes[a]['y']
    x2, y2 = G.nodes[b]['x'], G.nodes[b]['y']
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


############################################
# ✅ D* Lite
############################################
class DStarLite:
    def __init__(self,road_map,s_start,s_goal):
        self.map = road_map            # OSMnx 기반 그래프 래퍼
        self.s_start = s_start        # 로봇 시작 노드
        self.s_goal  = s_goal          # 목표 노드
        self.s_last  = s_start         # 이전 로봇 위치 (증분 업데이트용)
        self.k_m = 0                   # 키 오프셋(누적 이동거리)

        self.U=PriorityQueue()

        self.rhs={n:float('inf') for n in road_map.G.nodes()}
        self.g  ={n:float('inf') for n in road_map.G.nodes()}

        self.rhs[self.s_goal]=0
        self.U.insert(self.s_goal,self.calc_key(self.s_goal))

    def calc_key(self,s):
        return Priority(min(self.g[s],self.rhs[s])+heuristic(self.s_start,s,self.map.G)+self.k_m,
                        min(self.g[s],self.rhs[s]))

    def update_vertex(self,u):
        if u!=self.s_goal:
            # u에서 이웃으로 가는 모든 경로 중 최소 rhs 값 계산
            # 현재 노드 u의 후속 노드 s들을 기반으로 rhs[u]를 업데이트
            min_rhs_val = float('inf')
            # Changed logic to handle cases where there are no successors or all successors have infinite g values
            has_valid_succ = False
            for s in self.map.succ(u):
                cost_u_s = self.map.c(u,s)
                if self.g[s] != float('inf'): # g[s]가 무한대인 경우는 제외
                    min_rhs_val = min(min_rhs_val, cost_u_s + self.g[s])
                    has_valid_succ = True
            
            # If no valid successors found, rhs[u] remains infinity unless u is the goal
            self.rhs[u] = min_rhs_val if has_valid_succ else float('inf')

            # Goal node special handling (rhs is always 0)
            if u == self.s_goal:
                self.rhs[u] = 0
            
        if self.g[u]!=self.rhs[u]:
            self.U.update(u,self.calc_key(u))
        else:
            self.U.remove(u)

    def compute_shortest_path(self):
        while (not self.U.empty() and
               (self.U.top_key()<self.calc_key(self.s_start) or self.rhs[self.s_start]>self.g[self.s_start])):
            u=self.U.top()
            if u is None: # 큐가 비어있는데 top()이 None을 반환하는 경우 방지
                break
            k_old=self.U.top_key()
            k_new=self.calc_key(u)

            if k_old<k_new:
                self.U.update(u,k_new)
            elif self.g[u]>self.rhs[u]:
                self.g[u]=self.rhs[u]
                self.U.remove(u)
                for s in self.map.pred(u):
                    self.update_vertex(s)
            else: # g[u] <= rhs[u] 이지만, k_old >= k_new (즉, 비용이 증가한 경우)
                # 이 경우는 u의 g 값이 rhs 값보다 크거나 같지만, 키가 갱신되어야 하는 상황
                # g[u]를 무한대로 만들고, u와 그 선행 노드들을 다시 업데이트하여 재탐색 유도
                g_old=self.g[u]; self.g[u]=float('inf')
                for s in self.map.pred(u)+[u]: # u 자신도 업데이트 대상에 포함
                    # if s == self.s_goal: # Goal node should not be updated in this loop as its rhs is always 0
                    #     continue
                    self.update_vertex(s)

    def move_and_replan(self,cur):
        # 로봇이 이동했으므로 k_m 업데이트
        # k_m은 현재 위치(s_start)에서 마지막 위치(s_last)까지 이동한 비용을 반영
        # 환경 변화에 따라 k_m이 업데이트될 때, 모든 노드의 키가 이 변화를 반영하도록 함.
        self.k_m += heuristic(self.s_last, cur, self.map.G) # 로봇이 실제로 움직인 거리만큼 k_m 증가
        self.s_last = cur
        self.s_start = cur # 현재 로봇 위치가 새로운 시작점이 됨

        self.compute_shortest_path()

        if self.rhs[self.s_start]==float('inf'):
            return None # 경로를 찾을 수 없음

        # 현재 위치에서 목표까지의 최적 경로 생성 (A*와 유사한 역추적)
        path=[self.s_start]; c=self.s_start
        # Prevent infinite loop if path cannot be found to goal
        visited_in_path_reconstruction = {c}
        while c!=self.s_goal:
            nxt=None; best=float('inf')
            for nb in self.map.succ(c):
                cost = self.map.c(c,nb) # 엣지 비용
                if self.g[nb] != float('inf'): # 인접 노드의 g 값이 유효할 때만 고려
                    current_total_cost = cost + self.g[nb]
                    if current_total_cost < best:
                        best=current_total_cost; nxt=nb
            if nxt is None:
                # print(f"경로 소실 감지: 노드 {c}에서 다음 노드로 갈 수 없음.") # 디버깅
                return None # 더 이상 갈 곳이 없음 (경로 소실)
            
            # Prevent infinite loop in path reconstruction for cyclic graphs if an error occurred
            if nxt in visited_in_path_reconstruction:
                # print(f"경로 재구성 중 순환 감지: 노드 {nxt} 이미 방문됨. 경로 재구성 중단.")
                return None
            visited_in_path_reconstruction.add(nxt)

            c=nxt; path.append(c)
        return path
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
# --- 전역 변수 설정 (클릭 이벤트를 위해) ---
current_robot_node = None # 현재 로봇의 위치를 저장할 전역 변수 추가
d_star_planner = None # planner 객체를 전역으로 접근 가능하게

# 새로운 전역 변수: 여러 개의 위험 지역을 저장
# 각 위험 지역은 {type: str, center_lon: float, center_lat: float, effective_radius_meters: float} 형태로 저장
active_danger_zones = []

# 위험 지역 시각화 객체들을 저장 (갱신을 위해)
danger_circle_artists = [] # matplotlib.patches.Circle 객체
affected_edge_artists = [] # matplotlib.lines.Line2D 객체

selected_danger_type = None # 현재 선택된 위험 유형

# D* Lite가 업데이트한 노드들을 표시할 scatter 객체
dstar_nodes_marker = None


# 미터 단위의 반경을 위도/경도 차이로 변환 (근사치)
def convert_radius_meters_to_degrees(radius_meters, lat):
    # 지구 반경 (평균) 약 6371 km = 6371000 m
    # 1도 위도 간격: 약 111.139 km (赤道 기준) = 111139 m
    # 1도 경도 간격: 위도에 따라 달라짐 (111.139 km * cos(위도))
    lat_degree_diff = radius_meters / 111139.0
    lon_degree_diff = radius_meters / (111139.0 * math.cos(math.radians(lat)))
    return lat_degree_diff, lon_degree_diff

# 위험 요소 선택 함수
def select_danger_type_prompt(): # 이름 변경: select_danger_type -> select_danger_type_prompt
    global selected_danger_type
    print("\n어떤 종류의 위험 지역을 추가하시겠습니까? (현재 선택: {})".format(selected_danger_type if selected_danger_type else "없음"))
    print("1. 도로 붕괴 (Road Collapse) - 매우 높음")
    print("2. 다리 붕괴 (Bridge Collapse) - 매우 높음")
    print("3. 탱크 (Tanks) - 높음")
    print("4. 적군 (Enemies) - 보통")
    print("5. 화재 (Fire) - 높음")
    print("6. 폭발 (Explosion) - 높음")
    print("7. 철조망 (barbed_wire) - 매우 높음")
    print("8. 낙석 (rockfall) - 매우 높음")
    print("9. 선택 해제 (Reset)")
    
    choice = input("선택 (1-8) 또는 'q'로 종료: ")
    if choice == '1':
        selected_danger_type = 'road_collapse'
        print("도로 붕괴 위험 지역 추가 모드 활성화.")
    elif choice == '2':
        selected_danger_type = 'bridge_collapse'
        print("다리 붕괴 위험 지역 추가 모드 활성화.")
    elif choice == '3':
        selected_danger_type = 'tanks'
        print("탱크 위험 지역 추가 모드 활성화.")
    elif choice == '4':
        selected_danger_type = 'enemies'
        print("적군 위험 지역 추가 모드 활성화.")
    elif choice == '5':
        selected_danger_type = 'fire'
        print("화재 위험 지역 추가 모드 활성화.")
    elif choice == '6':
        selected_danger_type = 'explosion'
        print("폭발 위험 지역 추가 모드 활성화.")
    elif choice == '7':
        selected_danger_type = 'barbed_wire'
        print("철조망 위험 지역 추가 모드 활성화.")
    elif choice == '8':
        selected_danger_type = 'rockfall'
        print("낙석 위험 지역 추가 모드 활성화.")
    elif choice == '9':
        selected_danger_type = None
        print("위험 지역 선택 해제.")
    elif choice.lower() == 'q':
        return False # 종료 신호
    else:
        print("잘못된 선택입니다.")
    return True # 계속 진행 신호

# 모든 위험 지역을 기반으로 그래프의 엣지 가중치를 다시 설정하는 함수
def update_graph_risks(G_graph, road_map_obj, danger_zones_list, weight_multiplier=100):
    # 모든 엣지의 위험 속성을 0으로 초기화
    for u, v, k, d in G_graph.edges(keys=True, data=True):
        d['road_collapse'] = 0
        d['bridge_collapse'] = 0
        d['tanks'] = 0
        d['enemies'] = 0
        d['fire'] = 0
        d['explosion'] = 0
        d['barbed_wire'] = 0
        d['rockfall'] = 0
    
    # 각 활성화된 위험 지역에 따라 엣지 가중치 업데이트
    updated_nodes = set()
    for zone in danger_zones_list:
        center_lon, center_lat = zone['center_lon'], zone['center_lat']
        effective_radius_meters = zone['effective_radius_meters'] # 이미 계산된 유효 반경 사용
        danger_type = zone['type']

        for u, v, k, data in G_graph.edges(keys=True, data=True):
            u_x, u_y = G_graph.nodes[u]['x'], G_graph.nodes[u]['y']
            v_x, v_y = G_graph.nodes[v]['x'], G.nodes[v]['y']

            dist_u_to_center = ox.distance.great_circle(center_lat, center_lon, u_y, u_x)
            dist_v_to_center = ox.distance.great_circle(center_lat, center_lon, v_y, v_x)
            
            edge_mid_x = (u_x + v_x) / 2
            edge_mid_y = (u_y + v_y) / 2
            dist_edge_mid_to_center = ox.distance.great_circle(center_lat, center_lon, edge_mid_y, edge_mid_x)

            if dist_u_to_center <= effective_radius_meters or dist_v_to_center <= effective_radius_meters or dist_edge_mid_to_center <= effective_radius_meters:
                # 해당 위험 유형에 대해 가중치를 추가 (누적)
                for edge_key in G_graph[u][v]:
                    G_graph[u][v][edge_key][danger_type] += weight_multiplier # 누적
                updated_nodes.add(u)
                updated_nodes.add(v)
    return updated_nodes # D* Lite 업데이트를 위해 변경된 노드 반환

# 위험 지역 시각화를 갱신하는 함수
def redraw_danger_zones(G_graph, ax_obj, danger_zones_list):
    global danger_circle_artists, affected_edge_artists

    # 위험 유형별 색상 매핑
    danger_colors = {
        'road_collapse': 'red',
        'bridge_collapse': 'darkred',
        'tanks': 'brown',
        'enemies': 'orange',
        'fire': 'purple',
        'explosion':'yellow',
        'barbed_wire': 'red',
        'rockfall' :  'red'
    }

    # 기존 시각화 객체 제거
    for artist in danger_circle_artists + affected_edge_artists:
        if artist in ax_obj.lines or artist in ax_obj.collections: # Lines and collections can be removed
            artist.remove()
    danger_circle_artists.clear()
    affected_edge_artists.clear()

    # 모든 활성화된 위험 지역 다시 그리기
    for zone in danger_zones_list:
        center_lon, center_lat = zone['center_lon'], zone['center_lat']
        effective_radius_meters = zone['effective_radius_meters'] # 저장된 유효 반경 사용
        danger_type = zone['type']
        color = danger_colors.get(danger_type, 'gray') # 기본 색상은 회색

        # 위험 반경 원 시각화
        num_points = 100
        circle_lons = []
        circle_lats = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            lat_offset_degree, lon_offset_degree = convert_radius_meters_to_degrees(effective_radius_meters, center_lat)
            
            circle_lon = center_lon + lon_offset_degree * math.cos(angle)
            circle_lat = center_lat + lat_offset_degree * math.sin(angle)
            circle_lons.append(circle_lon)
            circle_lats.append(circle_lat)
        
        # Plot.line returns a list of lines, we need the first one
        circle_patch, = ax_obj.plot(circle_lons, circle_lats, c=color, ls='--', lw=2, zorder=7, label=f'{danger_type} 위험 지역')
        danger_circle_artists.append(circle_patch)

        # 영향을 받는 엣지들 시각화 (이미 update_graph_risks에서 가중치 부여됨)
        # 엣지를 다시 순회하며 해당 위험 유형이 설정된 엣지를 찾아서 그립니다.
        for u, v, k, d in G_graph.edges(keys=True, data=True):
            if d.get(danger_type, 0) > 0: # 해당 위험 유형이 0보다 큰 값으로 설정된 경우
                edge_x = [G_graph.nodes[u]['x'], G_graph.nodes[v]['x']]
                edge_y = [G_graph.nodes[u]['y'], G_graph.nodes[v]['y']]
                line, = ax_obj.plot(edge_x, edge_y, c=color, lw=4, alpha=0.7, zorder=3) # 원과 같은 색상
                affected_edge_artists.append(line)
    
    # 범례 갱신
    # ax_obj.legend() # 이 방식은 중복되거나 이상하게 보일 수 있음. 아래에서 일괄적으로 관리
    
    # plt.draw() # 전체 그리기는 on_click 마지막에 한 번만 호출

# D* Lite에 의해 업데이트된 노드들을 시각화하는 함수
def _plot_dstar_nodes(G_graph, ax_obj, d_star_planner_obj, dstar_nodes_marker_obj):
    # D* Lite의 g 또는 rhs 값이 inf가 아닌 모든 노드들을 수집
    dstar_updated_nodes_xy = []
    
    for node_id in G_graph.nodes():
        # 시작 노드와 목표 노드는 다른 마커로 표시되므로 제외
        if node_id == d_star_planner_obj.s_start or node_id == d_star_planner_obj.s_goal:
            continue

        if d_star_planner_obj.g.get(node_id, float('inf')) != float('inf') or \
           d_star_planner_obj.rhs.get(node_id, float('inf')) != float('inf'):
            
            dstar_updated_nodes_xy.append((G_graph.nodes[node_id]['x'], G_graph.nodes[node_id]['y']))
    
    if dstar_updated_nodes_xy:
        xs, ys = zip(*dstar_updated_nodes_xy)
        dstar_nodes_marker_obj.set_offsets(list(zip(xs, ys)))
    else:
        dstar_nodes_marker_obj.set_offsets([]) # 아무 노드도 없으면 빈 배열로 설정


# 클릭 이벤트 핸들러 함수
def on_click(event):
    global G, road_map, d_star_planner, ax, current_robot_node
    global active_danger_zones, selected_danger_type, dstar_nodes_marker
    
    # on_click 이벤트 내에서 사용되는 각 위험 유형의 유효 반경 (미터)
    # 랙 문제를 고려하여 fire 반경을 300m로 조정
    DANGER_RADIUS_METERS = {
        'road_collapse': 10,
        'bridge_collapse': 10, # 다리 붕괴 사이즈 10으로 조정
        'tanks': 300,
        'enemies': 200,
        'fire': 300, # 500m에서 300m로 줄임
        'explosion' : 300,
        'barbed_wire' : 10,
        'rockfall' : 10
    }
    WEIGHT_VALUE = 100 # 고정된 가중치 값

    if event.inaxes != ax:
        return

    if event.button == 1: # 왼쪽 클릭 (위험 지역 중심 노드 지정)
        if selected_danger_type is None:
            print("\n[알림] 먼저 콘솔에서 't'를 입력하여 위험 지역 유형을 선택해주세요.")
            return

        click_lon, click_lat = event.xdata, event.ydata # 클릭한 실제 경도, 위도
        
        effective_radius_meters = DANGER_RADIUS_METERS.get(selected_danger_type, 0)
        
        # 'bridge_collapse' 유형인 경우, 클릭 위치에서 가장 가까운 엣지를 찾고 그 엣지의 중심으로 설정
        if selected_danger_type == 'bridge_collapse' or selected_danger_type == 'road_collapse':
            # 가장 가까운 엣지 찾기 (u, v, key)
            nearest_edge = ox.distance.nearest_edges(G, click_lon, click_lat)
            if nearest_edge:
                u, v, key = nearest_edge
                # 엣지의 시작점과 끝점 좌표
                edge_start_lon, edge_start_lat = G.nodes[u]['x'], G.nodes[u]['y']
                edge_end_lon, edge_end_lat = G.nodes[v]['x'], G.nodes[v]['y']
                
                # 엣지의 중간점 (중심)을 위험 지역의 중심으로 사용
                center_lon = (edge_start_lon + edge_end_lon) / 2
                center_lat = (edge_start_lat + edge_end_lat) / 2
                print(f"가장 가까운 엣지 ({u}-{v}-{key})의 중간점을 다리 붕괴 위험 지역 중심으로 설정.")
            else:
                print("가장 가까운 엣지를 찾을 수 없습니다. 클릭 위치를 중심으로 사용합니다.")
                center_lon, center_lat = click_lon, click_lat
        else: # 다른 위험 유형은 클릭 위치를 그대로 사용
            center_lon, center_lat = click_lon, click_lat


        # 새로운 위험 지역 정보를 추가
        new_danger_zone = {
            'type': selected_danger_type,
            'center_lon': center_lon,
            'center_lat': center_lat,
            'effective_radius_meters': effective_radius_meters # 계산된 유효 반경 저장
        }
        active_danger_zones.append(new_danger_zone)

        print(f"새로운 '{selected_danger_type}' 위험 지역 추가: 중심=({center_lon:.4f}, {center_lat:.4f}), 유효 반경={effective_radius_meters:.2f}m")
        
        # 모든 활성화된 위험 지역을 기반으로 그래프 엣지 가중치 업데이트
        updated_nodes = update_graph_risks(G, road_map, active_danger_zones, WEIGHT_VALUE)
        
        # D* Lite에 변경 사항 알림 (영향을 받는 모든 노드 업데이트)
        # s_goal도 업데이트될 수 있으므로, pred와 succ 외에 u 자체도 확인
        for n in updated_nodes:
            d_star_planner.update_vertex(n)

        # 로봇이 이동하지 않았지만 환경이 변했으므로 k_m 및 s_last를 현재 로봇 위치로 업데이트
        # 이 로직은 D* Lite의 compute_shortest_path가 호출되기 전에 실행되어야 함
        if current_robot_node is not None:
            d_star_planner.k_m += heuristic(d_star_planner.s_last, current_robot_node, d_star_planner.map.G)
            d_star_planner.s_last = current_robot_node
            d_star_planner.s_start = current_robot_node # 현재 위치를 새로운 시작점으로
            
        d_star_planner.compute_shortest_path() # 지도 변경 후 경로 재계산

        # 모든 위험 지역 시각화 다시 그리기
        redraw_danger_zones(G, ax, active_danger_zones)
        _plot_dstar_nodes(G, ax, d_star_planner, dstar_nodes_marker) # D* Lite 계산 노드 시각화
        
        plt.draw() # 모든 업데이트를 한 번에 그림
        print("모든 위험 지역 및 경로 재계산 완료. 로봇이 다음 스텝에서 경로를 갱신합니다.")

    elif event.button == 3: # 오른쪽 클릭 (선택 리셋 및 위험 지역 초기화)
        print("모든 위험 지역 및 선택 초기화됨.")
        
        # 활성화된 위험 지역 목록 초기화
        active_danger_zones.clear()

        # 시각화 객체 제거 및 초기화
        for artist in danger_circle_artists + affected_edge_artists:
            if artist in ax.lines or artist in ax.collections:
                artist.remove()
        danger_circle_artists.clear()
        affected_edge_artists.clear()

        # 모든 엣지의 위험 속성 초기화 (update_graph_risks를 호출하여 수행)
        update_graph_risks(G, road_map, active_danger_zones, WEIGHT_VALUE) # 빈 리스트 전달하여 모두 0으로 초기화

        # D* Lite 플래너 초기화 (초기 상태로 되돌림)
        # 현재 로봇의 위치(current_robot_node)를 시작점으로 사용하도록 수정
        d_star_planner.__init__(road_map, current_robot_node, d_star_planner.s_goal)
        d_star_planner.s_last = current_robot_node
        d_star_planner.compute_shortest_path()

        # D* Lite 노드 시각화 초기화
        _plot_dstar_nodes(G, ax, d_star_planner, dstar_nodes_marker)

        plt.draw()
        print("모든 위험 지역 및 D* Lite 플래너가 초기화되었습니다.")

# 키보드 이벤트 핸들러 함수
def on_key(event):
    if event.key == 't': # 't' 키를 눌러 위험 유형 선택
        select_danger_type_prompt()


############################################
# ✅ 메인 시뮬레이션 (최적화 적용)
############################################
if __name__ == "__main__":

    GRAPH_PATH = "Seoul_graph.pkl"

    if os.path.exists(GRAPH_PATH):
        print("저장된 그래프 불러오는 중...")
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
    else:
        print("서울 그래프 다운로드 중...")
        # 기존 graph_from_place에서 simplify=True는 너무 많은 노드를 제거할 수 있습니다.
        # D* Lite는 그래프의 모든 노드를 다룰 수 있으므로, 여기서는 simplify를 제거합니다.
        G = ox.graph_from_place(["Eunpyeong-gu, South Korea","Seodaemun-gu, South Korea","Mapo-gu, South Korea"], network_type='drive')
        print("그래프 저장 중...")
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(G, f)
            
    try:
        gat_df = pd.read_csv(r"static/GAT.csv") 
        add_gat_weights_to_graph(G, gat_df)
    except FileNotFoundError:
        print("[경고] GAT 위험도 파일(GAT.csv)을 찾을 수 없습니다.")
        print("모든 엣지의 'gat_weight'를 0으로 설정합니다.")
        for u, v, data in G.edges(data=True):
            data['gat_weight'] = 0.0
            
    # 초기 엣지 속성 설정 (위험 요소 관련 속성 추가)
    for u, v, k, d in G.edges(keys=True, data=True):
        d['road_collapse'] = 0
        d['bridge_collapse'] = 0
        d['tanks'] = 0
        d['enemies'] = 0
        d['fire'] = 0
        d['explosion'] = 0
        d['barbed_wire'] = 0
        d['rockfall'] = 0

    print("노드 개수:", len(G.nodes))
    road_map = RoadNetworkMap(G)

    # 시작/목표 좌표 (동일)
    s_lat, s_lon = 37.525208, 127.035256 # 도산공원 
    g_lat, g_lon = 37.490250, 127.061751 # 양재천길

    start = ox.distance.nearest_nodes(G, s_lon, s_lat)
    goal = ox.distance.nearest_nodes(G, g_lon, g_lat)

    # 초기 A* 경로 (D* Lite와 무관, 시각화 참고용)
    print("초기 A* 경로 탐색...")
    try:
        init_path = nx.astar_path(G, start, goal, weight='length')
        print(f"초기 경로 : {len(init_path)}노드로 구성")
    except nx.NetworkXNoPath:
        init_path = None
        print("초기 경로를 찾을 수 없습니다.")

    # D* Lite 초기화 (동일)
    d_star_planner = DStarLite(road_map, start, goal)
    d_star_planner.s_last = start
    d_star_planner.compute_shortest_path()

    # === 🚀 시각화 최적화 부분 ===
    # 1. 엣지를 먼저 그립니다. 노드 크기는 0으로 설정하여 엣지만 표시합니다.
    #    ox.plot_graph 에는 zorder 인자가 없으므로, 기본적으로 낮은 zorder를 가집니다.
    fig, ax = ox.plot_graph(
        G, show=False, close=False, node_size=0,
        edge_color="gray", edge_linewidth=0.5
    )
    
    # 베이스맵 추가
    # contextily는 베이스맵을 ax에 추가하며, 이 또한 기본 zorder를 가집니다. (보통 0)
    cx.add_basemap(ax, crs=G.graph['crs'], source=cx.providers.OpenStreetMap.Mapnik, zoom=14)

    # 2. 모든 노드를 작은 점으로 그립니다.
    #    zorder=2로 설정하여 엣지(기본 zorder) 위에 그려지도록 합니다.
    all_node_x = [G.nodes[n]['x'] for n in G.nodes()]
    all_node_y = [G.nodes[n]['y'] for n in G.nodes()]
    ax.scatter(all_node_x, all_node_y, c='lightgray', s=1, marker='o', zorder=2)
    
    # 3. 시작/목표 노드 (zorder=5) - 다른 요소들보다 위에 명확하게 표시
    ax.scatter(G.nodes[start]['x'], G.nodes[start]['y'], c='lime', s=200, marker='o', zorder=5, label='시작 지점')
    ax.scatter(G.nodes[goal]['x'], G.nodes[goal]['y'], c='purple', s=200, marker='o', zorder=5, label='목표 지점')
    
    # 4. 로봇 이동 경로 및 D* Lite 계획 경로 (zorder=5, 6)
    robot_line, = ax.plot([], [], c='blue', lw=4, label='로봇 이동 경로', zorder=6)
    plan_line,  = ax.plot([], [], c='orange', lw=2, label='D* Lite 계획 경로', zorder=5)
    
    # 5. 로봇 현재 위치 마커 (zorder=7) - 가장 위에 표시
    robot_current_marker = ax.scatter([], [], c='cyan', s=150, marker='s', zorder=7, label='현재 로봇 위치')
    
    # 6. D* Lite가 계산한 노드들을 표시할 마커 (zorder=4)
    #    기본 노드(zorder=2) 위에, 경로 관련 마커(zorder=5 이상) 아래에 위치합니다.
    dstar_nodes_marker = ax.scatter([], [], c='darkgreen', s=30, marker='.', zorder=4, label='D* Lite 계산 노드')
    _plot_dstar_nodes(G, ax, d_star_planner, dstar_nodes_marker) # 초기 D* Lite 노드 시각화

    ax.legend(loc='upper right', fancybox=True, shadow=True, borderpad=1) # 범례 위치 조정
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key) # 키보드 이벤트 연결
    
    plt.ion() # 대화형 모드 활성화
    plt.show() # 초기 맵 표시

    # 사용자에게 위험 요소 선택 안내
    print("\n[안내] 키보드 't'를 눌러 위험 지역 유형을 선택하세요.")
    print("[안내] 지도에서 마우스 왼쪽 버튼을 클릭하여 선택된 유형의 위험 지역을 추가하세요.")
    print("여러 종류의 위험 지역을 원하는 위치에 추가할 수 있습니다.")
    print("모든 위험 지역을 초기화하려면 마우스 오른쪽 버튼을 클릭하세요.")


    # === 시뮬레이션 루프 (이하 동일) ===
    cur = start
    robot_path = [cur]
    steps = 0
    max_steps = 500
    current_robot_node = cur
    
    while cur != goal and steps < max_steps:
        steps+=1
        print(f"\nSTEP {steps}: 현재 위치={cur}")
        current_robot_node = cur

        replan = d_star_planner.move_and_replan(cur)
        
        # D* Lite 계산 노드 업데이트 시각화 (경로 재계산 후에)
        _plot_dstar_nodes(G, ax, d_star_planner, dstar_nodes_marker)

        # 현재 계획된 경로가 있고, 그 경로의 다음 엣지에 새로 추가된 위험이 있는지 확인
        # (on_click으로 'selected_danger_type'이 설정되었을 때만 확인)
        if replan and len(replan) >= 2 and selected_danger_type is not None:
            next_node = replan[1]
            # 현재 노드(cur)에서 다음 노드(next_node)로 가는 엣지들 확인
            for u, v, k, data in G.edges(keys=True, data=True):
                if (u == cur and v == next_node) or (u == next_node and v == cur): # 양방향 엣지 고려
                    # 이 엣지에 현재 선택된 위험 유형이 적용되었는지 확인
                    if data.get(selected_danger_type, 0) > 0:
                        print(f"\n[경고] STEP {steps}: 로봇이 다음 엣지 ({cur} -> {next_node})에서 '{selected_danger_type}' 위험 지역에 진입합니다!")
                        while True:
                            response = input("이 위험 엣지를 지나가시겠습니까? (y/n): ").lower()
                            if response == 'y':
                                print("로봇이 위험 엣지를 통과합니다.")
                                break # 루프를 벗어나 이동 계속
                            elif response == 'n':
                                print("로봇이 현재 위치에서 대기합니다. 경로를 다시 설정하거나 위험 지역을 제거하세요.")
                                # 로봇이 이동하지 않도록 현재 스텝을 다시 수행
                                steps -= 1 # 스텝 카운트를 되돌림
                                plt.pause(2) # 사용자가 지도에서 수정할 시간을 줌
                                replan = None # 경로를 없애서 아래 이동 로직을 건너뛰게 함
                                break # 루프를 벗어나 이동 중단
                            else:
                                print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")
                        if replan is None: # 사용자가 'n'을 선택하여 이동을 중단한 경우
                            break # 엣지 순회 중단
            if replan is None: # 내부 루프에서 이동 중단 결정이 내려졌으면 외부 루프도 다음 스텝으로 진행
                continue


        if not replan or len(replan) < 2:
            print(f"\n[경고] STEP {steps}: 목표까지 경로를 찾을 수 없습니다! (현재 위치: {cur})")
            robot_current_marker.set_offsets((G.nodes[cur]['x'], G.nodes[cur]['y']))
            plan_line.set_data([], [])
            ax.set_title(f"STEP {steps}: 경로 소실! 대기 @ {cur}")
            plt.draw()
            
            # 사용자에게 진행 여부 확인
            while True:
                response = input("이 경로로 계속 진행하시겠습니까? (y/n): ").lower()
                if response == 'y':
                    print("로봇이 현재 위치에서 대기하며, 다음 스텝에서 다시 경로를 탐색합니다.")
                    plt.pause(2) # 잠시 멈춰서 사용자가 지도를 수정할 시간을 줌
                    break
                elif response == 'n':
                    print("시뮬레이션을 종료합니다.")
                    plt.ioff()
                    plt.close()
                    exit() # 프로그램 종료
                else:
                    print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")

            # 경로 소실 시 재설정 옵션 (기존과 동일)
            if input("위험 요소를 재설정하고 다시 시도하시겠습니까? (y/n): ").lower() == 'y':
                select_danger_type_prompt() # 위험 유형 선택 프롬프트 다시 띄움
                # 모든 위험 지역 제거 및 엣지 초기화
                active_danger_zones.clear()
                updated_nodes = update_graph_risks(G, road_map, active_danger_zones, 100) # 모든 위험값 0으로
                
                # D* Lite 플래너 초기화
                d_star_planner = DStarLite(road_map, start, goal) # 시작점 재설정
                d_star_planner.s_last = start
                d_star_planner.compute_shortest_path()
                
                cur = start # 로봇을 시작 위치로 되돌림
                robot_path = [cur]
                steps = 0 # 스텝 초기화
                
                # 시각화 초기화
                redraw_danger_zones(G, ax, active_danger_zones) # 빈 리스트로 호출하여 모든 시각화 제거
                _plot_dstar_nodes(G, ax, d_star_planner, dstar_nodes_marker) # D* Lite 노드 시각화 초기화
                
                plt.draw()
                continue
            else:
                break # 시뮬레이션 종료
            

        nxt = replan[1]
        cur = nxt
        robot_path.append(cur)
        print(f"이동: {robot_path[-2]} → {cur}")

        rx = [G.nodes[n]['x'] for n in robot_path]
        ry = [G.nodes[n]['y'] for n in robot_path]
        robot_line.set_data(rx, ry)

        if replan:
            px = [G.nodes[n]['x'] for n in replan]
            py = [G.nodes[n]['y'] for n in replan]
            plan_line.set_data(px, py)
        else:
            plan_line.set_data([], [])

        robot_current_marker.set_offsets((G.nodes[cur]['x'], G.nodes[cur]['y']))
        ax.set_title(f"STEP {steps}: 현재 위치 {cur}")
        plt.draw()
        plt.pause(0.3)

    if cur == goal:
        print("\n=== 목표 도달! 시뮬레이션 종료 ===")
    else:
        print(f"\n=== 시뮬레이션 종료: {'경로 소실로 멈춤' if cur != goal and steps < max_steps else '최대 스텝 도달'} ===")

    plt.ioff()
    plt.show()