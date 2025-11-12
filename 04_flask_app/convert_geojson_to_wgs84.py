# import geopandas as gpd

# input_path = "static/nodes.geojson"
# output_path = "static/converted_nodes.geojson"

# # 1. GeoJSON 파일 불러오기
# gdf = gpd.read_file(input_path)  

# # 2️. 원래 좌표계 출력
# print("원래 좌표계:", gdf.crs)

# # 3️. WGS84로 변환 (EPSG:4326)
# if gdf.crs != "EPSG:4326":
#     gdf = gdf.to_crs(epsg=4326)
#     print("변환 후 좌표계:", gdf.crs)

# else:
#     print("이미 WGS84 좌표계입니다.")

# # 4️. 저장
# gdf.to_file(output_path, driver="GeoJSON")
# print(f"✅ 변환된 GeoJSON 저장 완료: {output_path}")

import geopandas as gpd

input_path = "static/edges.geojson"
output_path = "static/converted_edges.geojson"

# 1. GeoJSON 파일 불러오기
gdf = gpd.read_file(input_path)  

# 2️. 원래 좌표계 출력
print("원래 좌표계:", gdf.crs)

# 3️. WGS84로 변환 (EPSG:4326)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)
    print("변환 후 좌표계:", gdf.crs)

else:
    print("이미 WGS84 좌표계입니다.")

# 4️. 저장
gdf.to_file(output_path, driver="GeoJSON")
print(f"✅ 변환된 GeoJSON 저장 완료: {output_path}")