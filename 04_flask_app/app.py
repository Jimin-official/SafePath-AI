from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import os
import pickle
import threading
from shapely.geometry import LineString, mapping
from algorithm.dstarlite import DStarLite, RoadNetworkMap
import osmnx as ox
from flask_cors import CORS
from algorithm.cch_a import CCH, get_custom_weight, DANGER_RADIUS_METERS
import algorithm.cch_a as CCHMOD
from algorithm.rtaa import RTAAStar as RTAAStar, \
                           RoadNetworkMap as RTAA_RoadNetworkMap, \
                           update_graph_risks as rtaa_update_graph_risks
import networkx as nx
import requests
import time
import traceback, re
import math
from functools import wraps
from flask_mysqldb import MySQL
import json
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
import mgrs
from datetime import datetime
from ultralytics import YOLO
import cv2 
import decimal

app = Flask(__name__)
app.secret_key = "dev-secret-key"
app.config['ALLOW_ANY_LOGIN'] = True   # DB 없을 때 임시로 모두 통과
CORS(app)  # 배포 시 제한하기
try:
    # 프로젝트 루트에 있는 모델 파일 경로를 지정합니다.
    yolo_model = YOLO('yolomodel.pt', device="cpu") 
    print("✅ YOLOv11m 모델 로딩 성공.")
except Exception as e:
    print(f"🚨 YOLOv11m 모델 로딩 실패: {e}")
    yolo_model = None

# ========= MySQL 연결 설정 =========
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'jimin0518!!'
app.config['MYSQL_DB'] = 'final_project'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# app.config['MYSQL_HOST'] = '192.168.100.75'
# app.config['MYSQL_USER'] = 'teammate_user1'
# app.config['MYSQL_PASSWORD'] = 'acorn1234!'
# app.config['MYSQL_DB'] = 'final_project'
# app.config['MYSQL_CURSORCLASS'] = 'DictCursor' # 결과를 dictionary 형태로 받기

mysql = MySQL(app)
# ====================================


# ========= NAVER Maps v3 키 (신규: ncpKeyId) =========
# NAVER_MAPS_KEY = (
#     os.getenv("NAVER_MAPS_KEY")
#     or os.getenv("NCP_KEY_ID")
#     or os.getenv("NCP_CLIENT_ID")
#     or "fn56jqj6sp"  # 임시 구형 키 — 반드시 신규 ncpKeyId로 교체 + 도메인 등록
# )
# ====================================


# ========= 인메모리 사용자 저장소 (서버 재시작 시 초기화됨) =========
USERS = {}  # {"user_id": "plain_password"}  # 데모용! (운영에서는 해시 필수)

def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("home_page"))
        return view(*args, **kwargs)
    return wrapper
# ====================================

###########################
# 기본 페이지 라우트
###########################
# 로그인/회원가입
@app.route("/")
def home_page():
    return render_template("home_page.html")

# 추천 경로 안내 페이지
@app.route("/map_user")
def map_user():
    return render_template("map_user.html")

# 실시간 경로 탐색 페이지
# @app.route("/map")
# def map_page():
#     return render_template("map.html", NAVER_MAPS_KEY=NAVER_MAPS_KEY)
@app.route("/map")
def map_page():
    return render_template("map.html")

# 사용 가이드 페이지
@app.route("/index")
def index():
    return render_template("index.html")


###########################
# 카카오 Geocode 프록시
###########################
KAKAO_API_KEY = "f956ccbcb0adcc58706eff6e6a220f0e"

@app.route('/api/geocode')
def kakao_geocode():
    address = request.args.get('q')
    if not address:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=5)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        import traceback
        print('---- API GEOCODE ERROR ----')
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

###########################
# 회원가입
###########################
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", prev={})

    user_id  = (request.form.get("user_id")  or "").strip()
    user_pw  = (request.form.get("user_pw")  or "")
    user_pw2 = (request.form.get("user_pw2") or "")

    email    = (request.form.get("email")    or "").strip()
    username = (request.form.get("username") or "").strip()
    birth    = (request.form.get("birth")    or "").strip()
    address  = (request.form.get("address")  or "").strip()
    gender   = (request.form.get("gender")   or "").strip()
    phone    = (request.form.get("phone")    or "").strip()

    detail_address = (request.form.get("detailAddress") or "").strip()
    postcode       = (request.form.get("postcode") or "").strip()

    prev = {
        "user_id": user_id, "email": email, "username": username, "birth": birth,
        "gender": gender, "phone": phone,
        "detailAddress": detail_address, "postcode": postcode,
    }

    def fail(msg):
        return render_template("signup.html", error=msg, prev=prev)

    # --- 1. 필수값 검사 ---
    if not all([user_id, user_pw, user_pw2, email, username,
                birth, gender, phone, address, detail_address, postcode]):
        return fail("모든 필드를 입력하세요.")

    # --- 2. 비번 일치 검사 ---
    if user_pw != user_pw2:
        return fail("비밀번호가 일치하지 않습니다.")

    # --- 2.5. 아이디/비번 규칙 ---
    pattern = re.compile(
        r'^(?=.*[A-Za-z])(?=.*(\d|[!@#$%^&*()\-\_=+\.,\?]))[A-Za-z0-9!@#$%^&*()\-\_=+\.,\?]{8,20}$'
    )
    if not pattern.match(user_id):
        return fail("아이디는 영문 1자 이상 + 숫자 또는 특수문자 1자 이상, 8~20자입니다.")
    if not pattern.match(user_pw):
        return fail("비밀번호는 영문 1자 이상 + 숫자 또는 특수문자 1자 이상, 8~20자입니다.")

    # --- 3. DB 중복 검사 ---
    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT 1 FROM Users WHERE user_id = %s LIMIT 1", (user_id,))
        if cur.fetchone():
            cur.close()
            return fail("이미 사용 중인 아이디입니다.")

        cur.execute("SELECT 1 FROM Users WHERE phone = %s LIMIT 1", (phone,))
        if cur.fetchone():
            cur.close()
            return fail("이미 등록된 전화번호입니다.")

        cur.execute("SELECT 1 FROM Users WHERE email = %s LIMIT 1", (email,))
        if cur.fetchone():
            cur.close()
            return fail("이미 등록된 이메일입니다.")

    except Exception as e:
        traceback.print_exc()
        return fail("DB 조회 중 오류가 발생했습니다.")

    # --- 4. 생년월일 형식 검사 ---
    try:
        datetime.strptime(birth, "%Y-%m-%d")
    except ValueError:
        return fail("생년월일 형식은 YYYY-MM-DD 입니다.")

    # --- 5. 전화번호 숫자만 추출 후 길이 검사 ---
    digits = re.sub(r"\D", "", phone)
    if not (10 <= len(digits) <= 11):
        return fail("전화번호는 숫자 10~11자리여야 합니다.")
    phone = digits

    # --- 6. 약관 동의 확인 ---
    agree_privacy = request.form.get("agree_privacy")
    agree_unique  = request.form.get("agree_unique")
    agree_tos     = request.form.get("agree_tos")

    if not (agree_privacy and agree_unique and agree_tos):
        return fail("모든 필수 약관에 동의해야 회원가입이 가능합니다.")

    # --- 7. 비밀번호 해싱 + DB 저장 ---
    try:
        cur = mysql.connection.cursor()

        # --- DB에서 아이디/이메일 중복 확인 (기존과 동일) ---
        cur.execute("SELECT user_id FROM Users WHERE user_id = %s", (user_id,))
        if cur.fetchone():
            return fail("이미 사용 중인 아이디입니다.")
        
        cur.execute("SELECT email FROM Users WHERE email = %s", (email,))
        if cur.fetchone():
            return fail("이미 등록된 이메일입니다.")

        # --- 비밀번호 암호화 (Werkzeug 사용) ---
        # ⚠️ bcrypt.hashpw(...) 대신 이 함수를 사용합니다.
        hashed_password = generate_password_hash(user_pw)

        # --- DB에 사용자 정보 저장 (INSERT) ---
        sql = """
            INSERT INTO Users (
                user_id, user_pw, username, email, phone, address, gender, birth
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            user_id, hashed_password, username, email, phone, 
            f"{address} {detail_address}".strip(), # 주소 합치기
            gender, birth
        )
        cur.execute(sql, values)
        mysql.connection.commit()

    except Exception as e:
        traceback.print_exc()
        mysql.connection.rollback()
        return fail("데이터 저장 중 오류가 발생했습니다.")
    finally:
        if cur:
            cur.close()
    # --- 8. 성공 ---
    session.clear()
    flash("회원가입 완료! 로그인해 주세요.", "success")
    return redirect(url_for("home_page"))

###########################
# 회원탈퇴
###########################
@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    try:
        user_pk = session['user']['id']
        
        cur = mysql.connection.cursor()
        
        # 1. 사용자와 관련된 모든 RouteRequest의 input_id를 먼저 가져옵니다.
        cur.execute("SELECT input_id FROM RouteRequest WHERE user_pk = %s", (user_pk,))
        results = cur.fetchall()
        input_ids_to_delete = [row['input_id'] for row in results]
        
        # 2. 외래 키 제약 조건에 따라 자식 테이블부터 순서대로 삭제합니다.
        # (RouteResult, DangerZone, RouteRequest, UserInput, Favorites, Users 순)
        
        # 2-1. request_id를 사용하는 테이블들 삭제
        # request_id 목록을 가져옵니다.
        cur.execute("SELECT request_id FROM RouteRequest WHERE user_pk = %s", (user_pk,))
        request_ids_to_delete = [row['request_id'] for row in cur.fetchall()]

        if request_ids_to_delete:
            placeholders = ','.join(['%s'] * len(request_ids_to_delete))
            cur.execute(f"DELETE FROM RouteResult WHERE request_id IN ({placeholders})", request_ids_to_delete)
            cur.execute(f"DELETE FROM DangerZone WHERE request_id IN ({placeholders})", request_ids_to_delete)
        
        # 2-2. RouteRequest 테이블 삭제
        cur.execute("DELETE FROM RouteRequest WHERE user_pk = %s", (user_pk,))
        
        # 2-3. UserInput 테이블 삭제
        if input_ids_to_delete:
            placeholders = ','.join(['%s'] * len(input_ids_to_delete))
            cur.execute(f"DELETE FROM UserInput WHERE input_id IN ({placeholders})", input_ids_to_delete)

        # 2-4. Favorites 테이블 삭제
        cur.execute("DELETE FROM Favorites WHERE user_pk = %s", (user_pk,))
        
        # 3. 마지막으로 Users 테이블에서 사용자 본인을 삭제합니다.
        cur.execute("DELETE FROM Users WHERE id = %s", (user_pk,))
        
        mysql.connection.commit()
        cur.close()
        
        # 4. 세션을 클리어하여 로그아웃 처리합니다.
        session.clear()
        
        return jsonify({"success": True, "message": "회원 탈퇴가 성공적으로 처리되었습니다."})

    except Exception as e:
        if mysql.connection: mysql.connection.rollback()
        traceback.print_exc()
        return jsonify({"success": False, "message": "회원 탈퇴 중 오류가 발생했습니다."}), 500


###########################
# 로그인
###########################
@app.route("/login", methods=["POST"])
def login():
    user_id = request.form.get("user_id", "").strip()
    user_pw = request.form.get("user_pw", "")

    if not user_id or not user_pw:
        flash("아이디와 비밀번호를 모두 입력해주세요.", 'error')
        return redirect(url_for("home_page"))

    cur = None
    try:
        # --- DB에서 사용자 정보 조회 ---
        cur = mysql.connection.cursor()
        # id, user_pw 컬럼을 함께 가져옵니다.
        cur.execute("SELECT id, user_id, user_pw FROM Users WHERE user_id = %s", (user_id,))
        user_data = cur.fetchone() # DictCursor 덕분에 딕셔너리로 결과를 받습니다.

    except Exception as e:
        traceback.print_exc()
        flash("데이터베이스 조회 중 오류가 발생했습니다.", "error")
        return redirect(url_for("home_page"))
    finally:
        if cur:
            cur.close()

    # --- 사용자 존재 여부 및 비밀번호 확인 ---
    # user_data가 존재하고, 암호화된 비밀번호(user_data['user_pw'])와
    # 사용자가 입력한 비밀번호(user_pw)가 일치하는지 확인합니다.
    if user_data and check_password_hash(user_data['user_pw'], user_pw):
        # --- 로그인 성공: 세션에 사용자 정보(딕셔너리) 저장 ---
        session.clear() # 이전 세션이 남아있을 경우를 대비해 비워줍니다.
        session['user'] = {
            'id': user_data['id'],
            'user_id': user_data['user_id']
        }
        # 로그인 후 map.html 페이지로 이동합니다.
        return redirect(url_for("map_user"))
    else:
        # --- 로그인 실패 ---
        flash("아이디 또는 비밀번호가 올바르지 않습니다.", 'error')
        return redirect(url_for("home_page"))


###########################
# 로그아웃
###########################
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()  # 로그인 세션 제거
    flash("로그아웃 되었습니다. 다시 로그인해주세요.")
    return redirect(url_for("home_page"))  # 엔드포인트로 이동


###########################
# Leaflet에 보내기 전, 각 노드 간의 실제 도로 지오메트리
###########################
def path_nodes_to_linestring(graph, path):
    # 경로를 구성하는 전체 좌표 리스트
    full_coords = []

    for u, v in zip(path[:-1], path[1:]):
        data = graph.get_edge_data(u, v)
        if data is None:
            continue

        # 다중 엣지 그래프인 경우
        if isinstance(data, dict):
            edge_info = list(data.values())[0]
        else:
            edge_info = data

        # geometry 정보 있으면 사용
        if "geometry" in edge_info:
            coords = list(edge_info["geometry"].coords)
        else:
            # 없으면 두 노드의 위치로 직선 연결
            coords = [
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ]

        # 중복 방지: 마지막 점 제거하고 이어붙이기
        if full_coords and coords[0] == full_coords[-1]:
            coords = coords[1:]

        full_coords.extend(coords)

    return LineString(full_coords)


###########################
# 사용자가 입력한 출발지, 도착지 정보를 UserInput 테이블에 저장
###########################
@app.route('/input', methods=['GET', 'POST'])
def add_user_input():
    if request.method == 'POST':
        # 1. 폼에서 데이터 가져오기
        details = request.form
        user_pk = details['user_pk']
        start_address = details['start_address']
        start_lat = details['start_lat']
        start_lon = details['start_lon']
        start_mgrs = details['start_mgrs']
        goal_address = details['goal_address']
        goal_lat = details['goal_lat']
        goal_lon = details['goal_lon']
        goal_mgrs = details['goal_mgrs']    

        # 2. cursor 생성
        cur = mysql.connection.cursor()
        
        # 3. SQL 쿼리 실행 (SQL Injection 방지를 위해 %s 사용)
        sql = """
            INSERT INTO UserInput(user_pk, start_address, start_lat, start_lon, start_mgrs, goal_address, goal_lat, goal_lon, goal_mgrs) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """ # 실제로는 모든 컬럼을 다 넣어줘야 합니다.
        values = (
            user_pk, start_address, start_lat, start_lon, start_mgrs,
            goal_address, goal_lat, goal_lon, goal_mgrs)
        cur.execute(sql, values)
        
        # 4. 변경사항을 DB에 최종 반영 (INSERT, UPDATE, DELETE 시 필수!)
        mysql.connection.commit()
        
        # 5. cursor 닫기
        cur.close()
        
        return redirect(url_for('map_user')) # 결과 페이지로 이동
        
    return render_template('input_form.html')

###########################
# --- 즐겨찾기 기능 API ---
###########################

###########################
# 1. 즐겨찾기 추가 API
###########################

@app.route('/add_favorite', methods=['POST'])
@login_required
def add_favorite():
    try:
        user_pk = session['user']['id']
        data = request.json
        
        name = data.get('name')
        if not name:
            return jsonify({"success": False, "message": "즐겨찾기 이름이 필요합니다."}), 400

        cur = mysql.connection.cursor()
        sql = """
            INSERT INTO Favorites (user_pk, name, start_address, start_lat, start_lon, goal_address, goal_lat, goal_lon)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            user_pk, name,
            data.get('start_address'), data.get('start_lat'), data.get('start_lon'),
            data.get('goal_address'), data.get('goal_lat'), data.get('goal_lon')
        )
        cur.execute(sql, values)
        mysql.connection.commit()
        
        new_id = cur.lastrowid
        cur.close()
        
        return jsonify({"success": True, "message": "즐겨찾기에 추가되었습니다.", "favorite_id": new_id})

    except Exception as e:
        if mysql.connection: mysql.connection.rollback()
        traceback.print_exc()
        return jsonify({"success": False, "message": "즐겨찾기 추가 중 오류가 발생했습니다."}), 500
###########################
# 2. 즐겨찾기 목록 조회 API
###########################

@app.route('/get_favorites', methods=['GET'])
@login_required
def get_favorites():
    try:
        user_pk = session['user']['id']
        cur = mysql.connection.cursor()
        
        cur.execute("SELECT * FROM Favorites WHERE user_pk = %s ORDER BY created_at DESC", (user_pk,))
        favorites = cur.fetchall()
        cur.close()

        # DB에서 가져온 Decimal 타입을 float으로 변환 (JSON 호환을 위해)
        for fav in favorites:
            for key, value in fav.items():
                if isinstance(value, decimal.Decimal):
                    fav[key] = float(value)

        return jsonify({"success": True, "favorites": favorites})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": "즐겨찾기 목록을 불러오는 중 오류가 발생했습니다."}), 500
###########################
# 3. 즐겨찾기 삭제 API
###########################

@app.route('/delete_favorite', methods=['POST'])
@login_required
def delete_favorite():
    try:
        user_pk = session['user']['id']
        data = request.json
        favorite_id = data.get('favorite_id')
        
        if not favorite_id:
            return jsonify({"success": False, "message": "삭제할 항목의 ID가 필요합니다."}), 400

        cur = mysql.connection.cursor()
        # 보안: 반드시 현재 로그인한 사용자의 즐겨찾기만 삭제하도록 user_pk를 함께 확인
        cur.execute("DELETE FROM Favorites WHERE favorite_id = %s AND user_pk = %s", (favorite_id, user_pk))
        
        # 실제로 행이 삭제되었는지 확인
        if cur.rowcount == 0:
            mysql.connection.rollback()
            cur.close()
            return jsonify({"success": False, "message": "삭제할 즐겨찾기를 찾을 수 없거나 권한이 없습니다."}), 404
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({"success": True, "message": "즐겨찾기에서 삭제되었습니다."})

    except Exception as e:
        if mysql.connection: mysql.connection.rollback()
        traceback.print_exc()
        return jsonify({"success": False, "message": "즐겨찾기 삭제 중 오류가 발생했습니다."}), 500

###########################
# 개인 통계 대시보드 API
###########################
# app.py 파일의 기존 get_user_stats 함수를 아래 코드로 교체하세요.

# app.py 파일의 기존 get_user_stats 함수를 아래 코드로 교체하세요.

@app.route('/get_user_stats')
@login_required
def get_user_stats():
    try:
        user_pk = session['user']['id']
        cur = mysql.connection.cursor()

        # 1. 총 경로 탐색 횟수
        cur.execute("SELECT COUNT(*) as count FROM RouteRequest WHERE user_pk = %s", (user_pk,))
        total_searches = cur.fetchone()['count']

        # 2. 총 이동 거리 (km)
        cur.execute("""
            SELECT SUM(res.total_distance_km) as total_km
            FROM RouteResult res
            JOIN RouteRequest rr ON res.request_id = rr.request_id
            WHERE rr.user_pk = %s
        """, (user_pk,))
        total_distance = cur.fetchone()['total_km'] or 0

        # 3. 알고리즘별 총 이동 거리 (km) - 기존 '사용 횟수'에서 변경
        cur.execute("""
            SELECT res.algorithm_type, SUM(res.total_distance_km) as total_km
            FROM RouteResult res
            JOIN RouteRequest rr ON res.request_id = rr.request_id
            WHERE rr.user_pk = %s
            GROUP BY res.algorithm_type
        """, (user_pk,))
        # 결과를 algorithm_distance 라는 새로운 키로 저장합니다.
        algo_distance = {row['algorithm_type']: round(float(row['total_km'] or 0), 2) for row in cur.fetchall()}


        # 4. 월별 경로 탐색 횟수 (최근 6개월)
        cur.execute("""
            SELECT DATE_FORMAT(ui.created_at, '%%Y-%%m') as month, COUNT(*) as count
            FROM RouteRequest rr
            JOIN UserInput ui ON rr.input_id = ui.input_id
            WHERE rr.user_pk = %s AND ui.created_at >= DATE_FORMAT(NOW() - INTERVAL 5 MONTH, '%%Y-%%m-01')
            GROUP BY month
            ORDER BY month ASC
        """, (user_pk,))
        monthly_activity = {row['month']: row['count'] for row in cur.fetchall()}
        
        cur.close()

        # 최종 통계 데이터를 딕셔너리로 묶어서 반환
        stats = {
            'total_searches': total_searches,
            'total_distance_km': round(float(total_distance), 2),
            'algorithm_distance': algo_distance, # '사용 횟수' 대신 '이동 거리' 데이터를 전달
            'monthly_activity': monthly_activity
        }

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": "통계 데이터를 불러오는 중 오류가 발생했습니다."}), 500
###########################
# D*lite
###########################
# Leaflet에 보내기 전, 각 노드 간의 실제 도로 지오메트리
def path_nodes_to_linestring(graph, path):
    full_coords = []
    for u, v in zip(path[:-1], path[1:]):
        data = graph.get_edge_data(u, v)
        if data is None:
            # 역방향 시도
            data = graph.get_edge_data(v, u)
        if data is None:
            # 여전히 없으면 두 노드 좌표로 직선
            coords = [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                      (graph.nodes[v]["x"], graph.nodes[v]["y"])]
        else:
            edge_info = list(data.values())[0] if isinstance(data, dict) else data
            if "geometry" in edge_info:
                coords = list(edge_info["geometry"].coords)
            else:
                coords = [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                          (graph.nodes[v]["x"], graph.nodes[v]["y"])]
        if full_coords and coords[0] == full_coords[-1]:
            coords = coords[1:]
        full_coords.extend(coords)
    return LineString(full_coords)


###########################
# D*lite 시뮬
###########################
@app.route("/simulation_final")
def final_simulation_page():
    """네이버 지도 기반의 최종 시뮬레이션 페이지(map.html)를 렌더링합니다."""
    # NAVER_MAPS_KEY는 파일 상단에 이미 정의되어 있어야 합니다.
    return render_template("map.html", NAVER_MAPS_KEY=NAVER_MAPS_KEY)


# GeoJSON 경로 반환
@app.route('/route_geojson')
def route_geojson():
    try:
        start_lat = float(request.args.get('start_lat'))
        start_lon = float(request.args.get('start_lon'))
        goal_lat = float(request.args.get('goal_lat'))
        goal_lon = float(request.args.get('goal_lon'))
    except (TypeError, ValueError):
        return jsonify({"error": "start_lat, start_lon, goal_lat, goal_lon 파라미터가 필요하며 숫자여야 합니다."}), 400

    # 경로 계산: (lat, lon) → (lon, lat)로 변환 필요 여부는 알고리즘 내부에 맞춰야 함
    path_coords, distance, steps = run_dlite_algorithm(
        (start_lat, start_lon),
        (goal_lat, goal_lon),
        []
    )

    # 좌표가 [lat, lon] 순이면 → GeoJSON은 [lon, lat] 이어야 하므로 변환 필요
    geojson_coords = [[lon, lat] for lat, lon in path_coords]

    # LineString 객체로 생성
    line = LineString(geojson_coords)

    # GeoJSON 반환
    return jsonify({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(line),
                "properties": {
                    "distance": distance,
                    "steps": steps
                }
            }
        ]
    })

# 알고리즘 복수 선택
@app.route('/find_path', methods=['POST'])
@login_required
def find_path():    
    if WARMING_UP:
        return jsonify({"error": "engine warming up"}), 503
    
    data = request.json or {}
    cur = None

    try:
        # --- 1. 요청 데이터 추출 및 보완 ---
        user_pk = session['user']['id']
        
        start = data.get('start')
        end = data.get('end')
        start_node_id = data.get('start_node_id')
        end_node_id = data.get('end_node_id')
        algorithms = data.get('algorithms', [])
        danger_coords = data.get('danger_coords', [])

        # 💡 [수정된 부분] start 또는 end 좌표가 없을 경우, node_id로 좌표를 찾아옵니다.
        if not start and start_node_id and start_node_id in G_SEOUL.nodes:
            node = G_SEOUL.nodes[start_node_id]
            start = [node['y'], node['x']] # [lat, lon]

        if not end and end_node_id and end_node_id in G_SEOUL.nodes:
            node = G_SEOUL.nodes[end_node_id]
            end = [node['y'], node['x']]

        # 좌표가 여전히 없으면 에러 처리
        if not start or not end:
            return jsonify({"error": "출발지 또는 도착지 좌표를 확인할 수 없습니다."}), 400

        # --- 2. 데이터베이스 저장 로직 (이전과 동일) ---
        cur = mysql.connection.cursor()
        
        start_address = data.get('start_address', 'N/A')
        goal_address = data.get('goal_address', 'N/A')
        start_mgrs = data.get('start_mgrs', 'N/A')
        goal_mgrs = data.get('goal_mgrs', 'N/A')

        sql_input = """
            INSERT INTO UserInput (
                user_pk, start_address, start_lat, start_lon, start_mgrs,
                goal_address, goal_lat, goal_lon, goal_mgrs
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values_input = (
            user_pk, start_address, start[0], start[1], start_mgrs,
            goal_address, end[0], end[1], goal_mgrs
        )
        cur.execute(sql_input, values_input)
        input_id = cur.lastrowid
        # RouteRequest 테이블에 저장
        algo_str = ",".join(algorithms)
        sql_request = "INSERT INTO RouteRequest (user_pk, input_id, algorithm_type) VALUES (%s, %s, %s)"
        cur.execute(sql_request, (user_pk, input_id, algo_str))
        request_id = cur.lastrowid

        # DangerZone 테이블에 방해요소 목록 저장
        if danger_coords:
            # 1. MGRS 변환 객체 생성
            m = mgrs.MGRS()
            
            # 2. SQL 쿼리에 danger_mgrs와 radius_m 컬럼 추가
            sql_danger = """
                INSERT INTO DangerZone (
                    request_id, danger_type, lat, lon, danger_mgrs, radius_m
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            # 3. 각 장애물의 MGRS와 반경(radius) 값을 계산하여 values 리스트에 추가
            danger_values = []
            for obs in danger_coords:
                lat = obs.get('lat')
                lon = obs.get('lon')
                obs_type = obs.get('type', 'road_collapse')
                
                # 위도, 경도를 MGRS 좌표 문자열로 변환
                mgrs_coord = m.toMGRS(lat, lon)
                
                # 장애물 종류에 따른 반경(radius) 값을 가져옴
                # CCHMOD.DANGER_RADIUS_METERS 딕셔너리를 사용
                radius = CCHMOD.DANGER_RADIUS_METERS.get(obs_type, 100) # 기본값 100m
                
                danger_values.append(
                    (request_id, obs_type, lat, lon, mgrs_coord, radius)
                )
            
            # 4. executemany로 한 번에 모든 장애물 정보 저장
            cur.executemany(sql_danger, danger_values)
        
        mysql.connection.commit()

    except Exception as e:
        if mysql.connection:
            mysql.connection.rollback()
        traceback.print_exc() # 서버 로그에 전체 에러 출력
        return jsonify({"error": "요청을 데이터베이스에 저장하는 중 오류가 발생했습니다."}), 500
    finally:
        if cur:
            cur.close()

    # --- 2. 알고리즘 실행 (기존 코드와 동일) ---
    results = {}
    start_node_id = data.get('start_node_id')
    end_node_id = data.get('end_node_id')
    radius = data.get('danger_radius', 100)
    blocked_edges = data.get('blocked_edges', [])
    yolo_results = data.get('yolo_results', {})

    for algo in algorithms:
        try:
            if algo == 'dlite':
                if start_node_id is not None and end_node_id is not None:
                    path_coords, distance, steps = run_dlite_algorithm_by_node_ids(
                        start_node_id, end_node_id, danger_coords,
                        yolo_detections=yolo_results, # yolo 결과 전달 
                        radius=radius, blocked_edges=blocked_edges)
                else:
                    path_coords, distance, steps = run_dlite_algorithm(
                        start, end, danger_coords,
                        yolo_detections=yolo_results) # yolo 결과 전달
                results[algo] = {"path": path_coords, "distance": distance, "steps": steps}

            elif algo == 'cch_a':
                if start_node_id is not None and end_node_id is not None:
                    path_coords, distance, steps = run_cch_a_by_node_ids(
                        start_node_id, end_node_id, danger_coords)
                else:
                    path_coords, distance, steps = run_cch_a_algorithm(
                        start, end, danger_coords)
                results[algo] = {"path": path_coords, "distance": distance, "steps": steps}

            elif algo == 'rtaa':
                if start_node_id is not None and end_node_id is not None:
                    path_coords, distance, steps = run_rtaa_by_node_ids(
                        start_node_id, end_node_id, danger_coords)
                else:
                    path_coords, distance, steps = run_rtaa_algorithm(
                        start, end, danger_coords)
                results[algo] = {"path": path_coords, "distance": distance, "steps": steps}

        except Exception as e:
            traceback.print_exc()
            results[algo] = {"error": str(e)}

    # --- 3. 알고리즘 결과 DB 저장 ---
    cur = None
    try:
        cur = mysql.connection.cursor()
        sql_result = """
            INSERT INTO RouteResult (
                request_id, algorithm_type, total_distance_km, total_time_min, path_coords
            ) VALUES (%s, %s, %s, %s, %s)
        """
        result_values = []
        for algo, result in results.items():
            if "error" not in result and result.get("path"):
                distance_km = result.get("distance", 0) / 1000.0
                
                # --- 예상 소요 시간 계산 (이 부분 추가) ---
                # 평균 속력을 40km/h로 가정하여 분 단위로 시간을 계산합니다.
                # (시간 = 거리 / 속력) * 60분
                time_min = (distance_km / 40.0) * 60 if distance_km > 0 else 0
                
                path_json = json.dumps(result.get("path"))
                
                # values 튜플에 계산된 time_min을 추가합니다.
                result_values.append(
                    (request_id, algo, distance_km, time_min, path_json)
                )
        
        if result_values:
            cur.executemany(sql_result, result_values)
        mysql.connection.commit()

    except Exception as e:
        if mysql.connection:
            mysql.connection.rollback()
        traceback.print_exc()
        # 결과 저장은 실패하더라도, 이미 계산된 경로는 사용자에게 보여줄 수 있습니다.
        print(f"DB에 경로 결과 저장 실패: {e}")
    finally:
        if cur:
            cur.close()

    return jsonify({"paths": results, "request_id": request_id})


# G_SEOUL 그래프에서 가장 가까운 노드 id(osmid)를 찾아 JSON으로 반환
@app.route('/get_node_id')
def get_node_id():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        node_id = get_nearest_node(G_SEOUL, (lat, lon))
        print("node_id:", node_id, type(node_id))
        return jsonify({"node_id": node_id})
    except Exception as e:
        app.logger.error(f"get_node_id error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    

###########################
# 정적 장애물 정보 API (새로 추가)
###########################
@app.route('/api/static_obstacles')
@login_required # 이 라우트 역시 로그인이 필요합니다.
def get_static_obstacles():
    """DB에 저장된 고정 장애물(철조망, 낙석 등) 목록을 JSON으로 반환합니다."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT obstacle_type, lat, lon, description FROM StaticObstacles")
        obstacles = cur.fetchall()
        cur.close()
        return jsonify(obstacles)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"정적 장애물 조회 중 DB 오류 발생: {e}"}), 500


###########################
# D* Lite 알고리즘 연동
###########################
def get_nearest_node(G, point):
    return ox.distance.nearest_nodes(G, point[1], point[0])

# networkx 활용
def get_largest_component(G, strongly=True):
    if strongly:
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G.to_undirected()))
    largest_component = max(components, key=len)
    return G.subgraph(largest_component).copy()

# Pickle 파일에서 미리 처리된 그래프 불러오기
try:
    with open("seoul_graph.pkl", "rb") as f:
        G_SEOUL = pickle.load(f)
    print("✅ 미리 생성된 'seoul_graph.pkl' 파일을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("🚨 'seoul_graph.pkl' 파일이 없습니다. 먼저 prepare_graph.py를 실행해주세요.")
    # 파일이 없으면 서버를 종료하거나 비상 조치를 취할 수 있습니다.
    G_SEOUL = None # 또는 exit()
G_SEOUL = ox.distance.add_edge_lengths(G_SEOUL)

CCHMOD.G = G_SEOUL

DYNAMIC_RISK_KEYS = (
    "road_collapse", "bridge_collapse", "tanks", "enemies",
    "fire", "explosion", "barbed_wire", "rockfall"
)

def reset_dynamic_risks(G):
    for u, v, k, d in G.edges(keys=True, data=True):
        for key in DYNAMIC_RISK_KEYS:
            d[key] = 0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 지구 반경 (미터 단위)
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# D* Lite 알고리즘 실행 함수
def run_dlite_algorithm(start_coord, goal_coord, obstacles, yolo_detections={}):
    # 노드 탐색 (공용 그래프 사용)
    s_start = get_nearest_node(G_SEOUL, start_coord)
    s_goal = get_nearest_node(G_SEOUL, goal_coord)
    print(f"Start node: {s_start}, Goal node: {s_goal}")

    # 연결 경로 존재 여부 확인
    if not nx.has_path(G_SEOUL, s_start, s_goal):
        print("경로가 없습니다: 출발지와 도착지가 연결되어 있지 않습니다.")
        return [], 0, 0

    # 새로운 RoadNetworkMap 인스턴스 생성 
    road_map = RoadNetworkMap(G_SEOUL)
    dstar = DStarLite(road_map, s_start, s_goal)
    dstar.map.yolo_detections = yolo_detections

    # 위험 요소를 road_map 객체에만 반영
    for obs in obstacles:
        lat = obs.get("lat")
        lon = obs.get("lon")
        if lat is None or lon is None:
            continue

        node_id = get_nearest_node(G_SEOUL, (lat, lon))
        print(f"위험 지역 노드 ID: {node_id}")

        # 양방향 위험도 설정
        for v in road_map.succ(node_id):
            road_map.set_dynamic_risk(node_id, v, "road_collapse", 1)
        for u in road_map.pred(node_id):
            road_map.set_dynamic_risk(u, node_id, "road_collapse", 1)

        # D* Lite 알고리즘에 반영
        dstar.update_vertex(node_id)
        for pred in road_map.pred(node_id):
            dstar.update_vertex(pred)

    # 경로 계산
    path_nodes = dstar.move_and_replan(s_start)
    if not path_nodes:
        print("경로를 찾지 못했습니다.")
        return [], 0, 0
    print(f"경로 노드 리스트: {path_nodes}")

    # 실제 도로 지오메트리 기반 좌표 계산
    line = path_nodes_to_linestring(G_SEOUL, path_nodes)
    path_coords = list(line.coords)  # 이걸 GeoJSON으로 넘기기 좋게 변환

    # 거리 계산
    distance = 0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G_SEOUL.has_edge(u, v):
            edge_data = list(G_SEOUL[u][v].values())[0]
            distance += edge_data.get('length', 0)

    steps = len(path_nodes)
    return path_coords, distance, steps

# 노드 ID를 직접 받아서 D* Lite 경로 탐색
def run_dlite_algorithm_by_node_ids(start_node_id, goal_node_id, obstacles, yolo_detections={}, radius=100, blocked_edges=None):
    if blocked_edges is None:
        blocked_edges = []
        
    reset_dynamic_risks(G_SEOUL)
    road_map = RoadNetworkMap(G_SEOUL.copy(as_view=False))
    dstar = DStarLite(road_map, start_node_id, goal_node_id)
    dstar.map.yolo_detections = yolo_detections
    
    for obs in obstacles:
        lat = obs.get("lat")
        lon = obs.get("lon")
        count = obs.get("count", 1)
        
        # [수정 1] 각 장애물의 실제 'type'을 가져옵니다.
        obs_type = obs.get("type", "road_collapse")
        
        if lat is None or lon is None:
            continue
        
        # [수정 2] 장애물 종류(type)에 맞는 정확한 반경(radius) 값을 가져옵니다.
        # CCH 알고리즘에서 사용하던 DANGER_RADIUS_METERS 딕셔너리를 재사용합니다.
        radius_for_this_obs = CCHMOD.DANGER_RADIUS_METERS.get(obs_type, 100)

        # [수정 2 적용] 고정된 100m 대신, 위에서 찾은 개별 반경 값을 사용합니다.
        nodes_in_radius = [n for n in G_SEOUL.nodes if haversine(lat, lon, G_SEOUL.nodes[n]['y'], G_SEOUL.nodes[n]['x']) <= radius_for_this_obs]

        for nid in nodes_in_radius:
            for v in road_map.succ(nid):
                # [수정 1 적용] 하드코딩된 "road_collapse" 대신 실제 장애물 타입(obs_type)을 사용합니다.
                road_map.set_dynamic_risk(nid, v, obs_type, count)
            for u in road_map.pred(nid):
                # [수정 1 적용] 하드코딩된 "road_collapse" 대신 실제 장애물 타입(obs_type)을 사용합니다.
                road_map.set_dynamic_risk(u, nid, obs_type, count)
            dstar.update_vertex(nid)
            for pred in road_map.pred(nid):
                dstar.update_vertex(pred)

    path_nodes = dstar.move_and_replan(start_node_id)
    if not path_nodes:
        return [], 0, 0

    line = path_nodes_to_linestring(G_SEOUL, path_nodes)
    path_coords = [(y, x) for (x, y) in line.coords]
    
    distance = sum(G_SEOUL[u][v][0].get('length', 0) for u, v in zip(path_nodes[:-1], path_nodes[1:]) if G_SEOUL.has_edge(u, v))
    steps = len(path_nodes)
    return path_coords, distance, steps


###########################
# YOLO 연동
###########################
def run_yolo_on_image(hazard_type):
    """
    주어진 위험 타입에 해당하는 이미지 파일을 로드하고,
    YOLO 모델로 객체를 탐지하여 그 수를 반환합니다.
    """
    if not yolo_model:
        print("YOLO 모델이 로드되지 않았습니다. 분석을 건너뜁니다.")
        return {}

    # 1. 위험 타입과 '이미지' 파일 경로를 매핑합니다.
    image_map = {
        'tanks': 'static/img/hazard_tanks.jpg',
        'fire': 'static/img/hazard_fire.jpg',
        'explosion': 'static/img/hazard_explosion.jpg',
        'road_collapse': 'static/img/hazard_road.jpg'
    }

    image_path = image_map.get(hazard_type)
    if not image_path:
        print(f"'{hazard_type}'에 해당하는 이미지 파일이 없습니다.")
        return {}

    try:
        image = cv2.imread(image_path) if image_path and os.path.exists(image_path) else None
        if image is None:
            print(f"이미지를 로드할 수 없어 분석을 건너뜁니다: {image_path}")
            return {}

        # 3. YOLO 모델로 이미지를 분석합니다. (이 부분은 동일)
        results = yolo_model(image)

        # 4. 탐지된 객체들의 수를 셉니다. (이 부분은 동일)
        detected_counts = {}
        for cls_id in results[0].boxes.cls:
            class_name = yolo_model.names[int(cls_id)]
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
        print(f"YOLO 탐지 결과 ({hazard_type}): {detected_counts}")

        # --- 💡 [YOLO 연동] 탐지된 이름을 우리 시스템의 위험 타입으로 변환 ---
        CLASS_MAP = {
            # 시스템 위험 타입 : [YOLO가 탐지하는 실제 객체 이름 목록]
            'tanks':           ['north tank', 'korea tank'],
            'enemies':         ['north army', 'korea army'],
            'fire':            ['fire'],
            'road_collapse':   ['road collapse'],
            'bridge_collapse': ['bridge collapse'],
            'explosion':       ['explosion']
        }

        final_counts = {}
        for detected_name, count in detected_counts.items():
            # CLASS_MAP을 순회하며 어디에 속하는지 찾음
            for system_name, yolo_names in CLASS_MAP.items():
                if detected_name in yolo_names:
                    # 해당하는 시스템 이름(예: 'tanks')으로 수를 누적
                    final_counts[system_name] = final_counts.get(system_name, 0) + count
                    break # 찾았으면 다음 탐지 객체로 넘어감
        
        print(f"최종 변환된 위험 요소 수: {final_counts}")
        return final_counts

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
# --- 2. YOLO 탐지 API 라우트 ---
@app.route('/api/detect_from_image', methods=['POST'])
@login_required
def detect_from_image():
    """
    프론트엔드로부터 위험 타입을 받아, 해당 이미지로 YOLO 탐지를 수행하고 결과를 반환합니다.
    """
    data = request.json
    hazard_type = data.get('hazard_type')
    if not hazard_type:
        return jsonify({"error": "hazard_type이 필요합니다."}), 400

    try:
        # YOLO 탐지 시뮬레이션 함수 호출
        detection_results = run_yolo_on_image(hazard_type)
        return jsonify(detection_results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

###########################
# 마이페이지 라우트 및 데이터 조회 추가
###########################
@app.route('/mypage')
@login_required
def mypage():
    try:
        user_pk = session['user']['id']
        
        # --- 1. 프론트엔드에서 보낸 검색/필터 값 받기 ---
        search_query = request.args.get('q', '').strip()
        start_date = request.args.get('start_date', '').strip()
        end_date = request.args.get('end_date', '').strip()

        cur = mysql.connection.cursor()

        # --- 2. 사용자 정보 조회 (기존과 동일) ---
        cur.execute("SELECT user_id, username, email, phone, address FROM Users WHERE id = %s", (user_pk,))
        user_info = cur.fetchone()

        # --- 3. 검색/필터 조건에 따라 동적으로 SQL 쿼리 만들기 ---
        # 기본 SQL 쿼리
        base_sql = """
            SELECT 
                rr.request_id, ui.created_at, ui.start_address, ui.goal_address,
                res.algorithm_type, res.path_coords
            FROM RouteRequest rr
            JOIN UserInput ui ON rr.input_id = ui.input_id
            LEFT JOIN RouteResult res ON rr.request_id = res.request_id
            WHERE rr.user_pk = %s
        """
        
        # 파라미터와 WHERE 조건을 담을 리스트
        params = [user_pk]
        where_conditions = []

        # 주소 검색어가 있는 경우
        if search_query:
            where_conditions.append("(ui.start_address LIKE %s OR ui.goal_address LIKE %s)")
            params.extend([f"%{search_query}%", f"%{search_query}%"])

        # 시작 날짜가 있는 경우
        if start_date:
            where_conditions.append("DATE(ui.created_at) >= %s")
            params.append(start_date)

        # 종료 날짜가 있는 경우
        if end_date:
            where_conditions.append("DATE(ui.created_at) <= %s")
            params.append(end_date)
        
        # 모든 WHERE 조건을 'AND'로 연결
        if where_conditions:
            base_sql += " AND " + " AND ".join(where_conditions)

        # 최종 정렬 순서 추가
        final_sql = base_sql + " ORDER BY ui.created_at DESC, rr.request_id DESC;"

        cur.execute(final_sql, tuple(params))
        all_results = cur.fetchall()
        cur.close()

        # --- 4. 데이터 재조립 (기존과 동일) ---
        history_dict = {}
        for row in all_results:
            req_id = row['request_id']
            if req_id not in history_dict:
                history_dict[req_id] = {
                    'request_id': req_id,
                    'created_at': row['created_at'],
                    'start_address': row['start_address'],
                    'goal_address': row['goal_address'],
                    'paths_by_algo': {}
                }
            
            algo_type = row['algorithm_type']
            path_coords = row['path_coords']
            if algo_type and path_coords:
                clean_algo_type = algo_type.lower().replace(' ', '').replace('*','')
                if 'cch' in clean_algo_type: clean_algo_type = 'cch_a'
                elif 'd' in clean_algo_type: clean_algo_type = 'dlite'
                
                try:
                    history_dict[req_id]['paths_by_algo'][clean_algo_type] = json.loads(path_coords)
                except (json.JSONDecodeError, TypeError):
                    history_dict[req_id]['paths_by_algo'][clean_algo_type] = None

        history = list(history_dict.values())

        if not user_info:
            flash("사용자 정보를 찾을 수 없습니다.", "error")
            return redirect(url_for('home_page'))

        # --- 5. 템플릿에 검색 값 전달 ---
        # 사용자가 입력했던 검색 조건을 다시 화면에 표시해주기 위함
        search_values = {
            'q': search_query,
            'start_date': start_date,
            'end_date': end_date
        }

        return render_template('mypage.html', user_info=user_info, history=history, search_values=search_values)

    except Exception as e:
        traceback.print_exc()
        flash("마이페이지를 불러오는 중 오류가 발생했습니다. 서버 로그를 확인해주세요.", "error")
        return redirect(url_for('map_user'))
    
###########################
# 사용자 정보 업데이트 API 추가
###########################

@app.route('/update_user_info', methods=['POST'])
@login_required
def update_user_info():
    try:
        user_pk = session['user']['id']
        
        # 폼에서 새로운 데이터 가져오기
        new_user_id = request.form.get('user_id')
        new_address = request.form.get('address')
        new_email = request.form.get('email')
        new_phone = request.form.get('phone')
        
        # (보안을 위해 실제 서비스에서는 더 엄격한 유효성 검사가 필요합니다)
        
        cur = mysql.connection.cursor()

        # 아이디, 이메일, 전화번호 중복 확인
        cur.execute("SELECT id FROM Users WHERE user_id = %s AND id != %s", (new_user_id, user_pk))
        if cur.fetchone():
            return jsonify({"success": False, "message": "이미 사용 중인 아이디입니다."})
        
        cur.execute("SELECT id FROM Users WHERE email = %s AND id != %s", (new_email, user_pk))
        if cur.fetchone():
            return jsonify({"success": False, "message": "이미 사용 중인 이메일입니다."})

        cur.execute("SELECT id FROM Users WHERE phone = %s AND id != %s", (new_phone, user_pk))
        if cur.fetchone():
            return jsonify({"success": False, "message": "이미 사용 중인 전화번호입니다."})

        # DB 업데이트 쿼리 수정
        cur.execute(
            """UPDATE Users 
               SET user_id = %s, address = %s, email = %s, phone = %s 
               WHERE id = %s""",
            (new_user_id, new_address, new_email, new_phone, user_pk)
        )
        mysql.connection.commit()
        cur.close()

        # 세션 정보도 업데이트
        session['user']['user_id'] = new_user_id
        session.modified = True
        
        return jsonify({"success": True, "message": "정보가 성공적으로 업데이트되었습니다."})

    except Exception as e:
        traceback.print_exc()
        if mysql.connection: mysql.connection.rollback()
        return jsonify({"success": False, "message": "업데이트 중 오류가 발생했습니다."}), 500
@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history():
    try:
        user_pk = session['user']['id']
        data = request.json
        request_ids = data.get('request_ids', [])

        if not request_ids:
            return jsonify({"success": False, "message": "삭제할 항목이 선택되지 않았습니다."}), 400

        cur = mysql.connection.cursor()

        # --- 보안: 현재 로그인한 사용자의 기록이 맞는지 확인하고, 관련된 input_id를 가져옵니다. ---
        # %s 플레이스홀더를 동적으로 생성하여 SQL Injection을 방지합니다.
        placeholders = ','.join(['%s'] * len(request_ids))
        sql_get_inputs = f"SELECT input_id FROM RouteRequest WHERE user_pk = %s AND request_id IN ({placeholders})"
        params = [user_pk] + request_ids
        cur.execute(sql_get_inputs, params)
        results = cur.fetchall()
        
        if not results:
             return jsonify({"success": False, "message": "삭제 권한이 없거나 유효하지 않은 요청입니다."}), 403

        input_ids_to_delete = [row['input_id'] for row in results]

        # --- 트랜잭션 시작: 관련된 모든 데이터를 한 번에 삭제 ---
        # 외래 키 제약조건 위반을 피하기 위해 자식 테이블부터 삭제합니다. (연쇄 작용)
        # 1. RouteResult 삭제
        cur.execute(f"DELETE FROM RouteResult WHERE request_id IN ({placeholders})", request_ids)
        # 2. DangerZone 삭제
        cur.execute(f"DELETE FROM DangerZone WHERE request_id IN ({placeholders})", request_ids)
        # 3. RouteRequest 삭제 (다시 한번 user_pk 확인)
        cur.execute(f"DELETE FROM RouteRequest WHERE request_id IN ({placeholders}) AND user_pk = %s", request_ids + [user_pk])
        
        # 4. UserInput 테이블 삭제
        if input_ids_to_delete:
            input_placeholders = ','.join(['%s'] * len(input_ids_to_delete))
            cur.execute(f"DELETE FROM UserInput WHERE input_id IN ({input_placeholders})", input_ids_to_delete)

        mysql.connection.commit()
        cur.close()
        
        return jsonify({"success": True, "message": f"{len(request_ids)}개의 기록이 삭제되었습니다."})

    except Exception as e:
        if mysql.connection: mysql.connection.rollback()
        traceback.print_exc()
        return jsonify({"success": False, "message": "기록 삭제 중 오류가 발생했습니다."}), 500
        
###########################
# CCH + A*알고리즘 연동
###########################
# danger_coords [{lat, lon, type}, ...]를 CCH가 요구하는 danger_zones로 변환
def build_danger_zones(danger_coords):
    zones = []
    if not danger_coords:
        return zones
    for obs in danger_coords:
        lat = obs.get("lat")
        lon = obs.get("lon")
        typ = obs.get("type")
        count = obs.get("count", 1) # 💡 개수(count)를 가져옵니다. 기본값은 1.
        if lat is None or lon is None or typ is None:
            continue
        radius = DANGER_RADIUS_METERS.get(typ, 100)
        zones.append({
            "type": typ,
            "center_lat": float(lat),
            "center_lon": float(lon),
            "effective_radius_meters": float(radius),
            "count": int(count) # 💡 개수 정보 추가
        })
    return zones


WARMING_UP = True

def _warmup_once():
    global WARMING_UP
    try:
        # 1) CCH 계층 미리 구축 
        _ = _get_cch()

        # 2) OSMnx 최근접노드 인덱스 준비(더미 호출)
        any_node = next(iter(G_SEOUL.nodes))
        cy = float(G_SEOUL.nodes[any_node]['y']); cx = float(G_SEOUL.nodes[any_node]['x'])
        ox.distance.nearest_nodes(G_SEOUL, X=[cx], Y=[cy])

        app.logger.info("Warmup completed.")
    except Exception:
        app.logger.exception("Warmup failed")
    finally:
        WARMING_UP = False

_CCH_CACHE = {"obj": None}

def _get_cch():
    if _CCH_CACHE["obj"] is None:
        cch = CCHMOD.CCH(G_SEOUL)
        cch.build_hierarchy() 
        _CCH_CACHE["obj"] = cch
    return _CCH_CACHE["obj"]

def _to_cch_zones(danger_coords):
    zones = []
    for z in (danger_coords or []):
        lat = (z.get("lat") or z.get("latitude"))
        lon = (z.get("lon") or z.get("lng") or z.get("longitude"))
        typ = (z.get("type") or z.get("kind") or "road_collapse")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            zones.append({
                "type": typ,
                "center_lat": float(lat),
                "center_lon": float(lon),
                "effective_radius_meters": CCHMOD.DANGER_RADIUS_METERS.get(typ, 100)
            })
    return zones

def run_cch_a_by_node_ids(start_node_id, end_node_id, danger_coords):
    cch = CCH(G_SEOUL)
    cch.build_hierarchy()

    danger_zones = build_danger_zones(danger_coords)
    cch.customize(get_custom_weight(G_SEOUL, danger_zones), danger_zones, mode="fast")

    path_nodes, expanded, ms = cch.query(start_node_id, end_node_id)

    line = path_nodes_to_linestring(G_SEOUL, path_nodes)
    path_coords = [(y, x) for (x, y) in line.coords]

    distance = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G_SEOUL.has_edge(u, v):
            edge_data = list(G_SEOUL[u][v].values())[0]
            distance += edge_data.get('length', 0.0)

    steps = len(path_nodes)
    return path_coords, distance, steps


def run_cch_a_algorithm(start, end, danger_coords):
    # 좌표 → 노드
    s_start = get_nearest_node(G_SEOUL, (start[0], start[1]))
    s_goal  = get_nearest_node(G_SEOUL, (end[0],   end[1]))

    # CCH 인스턴스
    cch = CCH(G_SEOUL)
    cch.build_hierarchy()

    # 좌표 → danger_zones
    danger_zones = build_danger_zones(danger_coords)

    # danger_zones를 두 번째 인자로 
    cch.customize(get_custom_weight(G_SEOUL, danger_zones), danger_zones, mode="fast")

    # 경로 질의
    path_nodes, expanded, ms = cch.query(s_start, s_goal)

    # 실제 도로 지오메트리로 변환
    line = path_nodes_to_linestring(G_SEOUL, path_nodes)
    path_coords = [(y, x) for (x, y) in line.coords]

    # 길이 합산
    distance = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G_SEOUL.has_edge(u, v):
            edge_data = list(G_SEOUL[u][v].values())[0]
            distance += edge_data.get('length', 0.0)

    steps = len(path_nodes)
    return path_coords, distance, steps


###########################
# RTAA* 알고리즘 연동
###########################
def _to_rtaa_zones(danger_coords):
    zones = []
    for z in (danger_coords or []):
        lat = z.get("lat") or z.get("latitude")
        lon = z.get("lon") or z.get("lng") or z.get("longitude")
        typ = (z.get("type") or z.get("kind") or "road_collapse")
        count = z.get("count", 1)  # 💡 count 값을 가져옵니다.

        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            zones.append({
                "type": typ,
                "center_lat": float(lat),
                "center_lon": float(lon),
                "effective_radius_meters": CCHMOD.DANGER_RADIUS_METERS.get(typ, 100),
                "count": int(count)  # 💡 zone 정보에 count를 포함시킵니다.
            })
    return zones

def run_rtaa_by_node_ids(start_node_id, end_node_id, danger_coords, N=800, max_steps=30000):
    # 1) 그래프 복사 후 위험 반영
    Gtmp = G_SEOUL.copy(as_view=False)
    zones = _to_rtaa_zones(danger_coords)
    rtaa_update_graph_risks(Gtmp, zones)

    # 2) RTAA* 초기화
    road_map = RTAA_RoadNetworkMap(Gtmp)
    planner  = RTAAStar(road_map, start_node_id, end_node_id, N=N)
    planner.init()  # 휴리스틱 테이블 구성

    status = "CONTINUE"
    steps  = 0
    while steps < max_steps:
        status = planner.search_step()
        if status in ("GOAL", "STUCK"):
            break
        steps += 1

    if status != "GOAL" or len(planner.path) < 2:
        return [], 0.0, 0

    node_path = planner.path

    # 3) 엣지 geometry 그대로 이어서 라인 구성 
    line = path_nodes_to_linestring(Gtmp, node_path)
    latlon_path = [(y, x) for (x, y) in line.coords]

    # 4) 총 길이(m)
    total_len = 0.0
    for u, v in zip(node_path[:-1], node_path[1:]):
        data = Gtmp.get_edge_data(u, v)
        if not data:
            continue
        d0 = next(iter(data.values())) if isinstance(data, dict) else data
        total_len += float(d0.get("length", 0.0))

    return latlon_path, total_len, len(node_path) - 1


def run_rtaa_algorithm(start, end, danger_coords):
    s = get_nearest_node(G_SEOUL, (start[0], start[1]))
    t = get_nearest_node(G_SEOUL, (end[0],   end[1]))
    return run_rtaa_by_node_ids(s, t, danger_coords)


if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    try:
        threading.Thread(target=_warmup_once, daemon=True).start()
    except Exception:
        app.logger.exception("Failed to start warmup thread")

if __name__ == "__main__":
    app.run(debug=True)
    # 배포 시에는 debug=False로 변경