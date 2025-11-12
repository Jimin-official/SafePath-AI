프로젝트명 : 전시상황에서 군수품 수송로 확보를 위한 실시간 경로 탐색 시스템 개발
팀명 : TacNac -R

1. 개요 
 - 방해요소 회피 경로 탐색 시스템
 - Flask + Leaflet 기반 웹 UI 
 - D* Lite, CCH + A*, RTAA* 경로 탐색 알고리즘 적용  - Render PaaS를 활용한 웹, API, DB 통합 배포 

2. 주요 기능
 - 회원가입 및 로그인, 사용자별 즐겨찾기 경로 관리
 - 주소 검색 및 지도 마커 표시
 - 위경도 변환 및 MGSR 군사좌표 표시
 - 방해요소 추가(북한군, 북한전차, 화재, 폭발, 다리붕괴, 도로붕괴) 및 회피 경로 탐색
 - 알고리즘별 경로 비교 및 시뮬레이션
 - 지도 스타일 전환(일반/위성/지형) 

3. 실행 환경
 - Python 3.9
 - Flask, requests, osmnx, networkx 등
 - requirements.txt 참고 

4. 설치 및 실행 방법
 - 의존성 설치: pip install -r requirements.txt 
 - 서버 실행: python app.py
 - 브라우저에서 접속: http://127.0.0.1:5000 

5. 배포
 - Render PaaS를 이용하여 웹 UI, API 서버, DB를 통합 배포
 - GitHub 연동을 통한 자동 빌드 및 배포(CI/CD)
 - Managed DB(PostgreSQL/MySQL) 사용
 - HTTPS 지원 및 자동 스케일링 

6. 폴더 구조
 - app.py                  # 메인 Flask 서버
 - templates/              # HTML 템플릿
 - static/                 # JS, CSS, 이미지
 - algorithm/              # 경로 탐색 알고리즘 코드
 - data/                   # GeoJSON, OSM 데이터
 - requirements.txt
 - README.txt 