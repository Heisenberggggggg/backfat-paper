def check_packages():
    # 检查包是否安装成功
    try:
        import geopandas
        print("geopandas 安装成功，版本号:", geopandas.__version__)
    except ImportError:
        print("geopandas 未安装")

    try:
        import matplotlib
        print("matplotlib 安装成功，版本号:", matplotlib.__version__)
    except ImportError:
        print("matplotlib 未安装")

    try:
        import shapely
        print("shapely 安装成功，版本号:", shapely.__version__)
    except ImportError:
        print("shapely 未安装")

    try:
        import requests
        print("requests 安装成功，版本号:", requests.__version__)
    except ImportError:
        print("requests 未安装")

if __name__ == '__main__':
    check_packages() 