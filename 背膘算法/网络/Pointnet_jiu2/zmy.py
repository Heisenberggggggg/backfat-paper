import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import json
from shapely.geometry import shape
import pandas as pd

def download_map_data():
    # 下载地图数据
    url = 'https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json'
    response = requests.get(url)
    return response.json()

def plot_china_map():
    # 获取地图数据
    geojson_data = download_map_data()
    
    # 从GeoJSON创建GeoDataFrame
    features = geojson_data['features']
    geometries = [shape(feature['geometry']) for feature in features]
    properties = [feature['properties'] for feature in features]
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 绘制地图
    gdf.plot(ax=ax, edgecolor='black', facecolor='white')
    
    # 设置标题
    plt.title('中国省份地图', fontsize=16, fontfamily='SimHei')
    
    # 去除坐标轴
    ax.axis('off')
    
    # 保存图片
    plt.savefig('china_map.png', dpi=300, bbox_inches='tight')
    
    # 显示图片
    plt.show()

if __name__ == '__main__':
    plot_china_map() 