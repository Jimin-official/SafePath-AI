import osmnx as ox

G = ox.graph_from_place(["Eunpyeong-gu, South Korea","Seodaemun-gu, South Korea","Mapo-gu, South Korea"], network_type='all')
nodes, edges = ox.graph_to_gdfs(G)

# GeoJSON으로 저장
nodes.to_file(r"static\nodes.geojson", driver="GeoJSON")
edges.to_file(r"static\edges.geojson", driver="GeoJSON")