def analyze_pig_data():
    # 各省生猪出栏量数据（单位：万头）
    pig_data = {
        '四川省': 5800,
        '河南省': 5500,
        '山东省': 4800,
        '湖南省': 4200,
        '广东省': 3900,
        '湖北省': 3500,
        '广西壮族自治区': 3300,
        '河北省': 3100,
        '安徽省': 2900,
        '江西省': 2800,
        '江苏省': 2700,
        '云南省': 2600,
        '福建省': 2200,
        '贵州省': 2000,
        '黑龙江省': 1900,
        '重庆市': 1800,
        '陕西省': 1700,
        '辽宁省': 1600,
        '浙江省': 1500,
        '山西省': 1400,
        '吉林省': 1300,
        '内蒙古自治区': 1200,
        '甘肃省': 900,
        '新疆维吾尔自治区': 800,
        '海南省': 700,
        '宁夏回族自治区': 500,
        '青海省': 400,
        '天津市': 300,
        '西藏自治区': 300,
        '北京市': 200,
        '上海市': 100,
    }
    
    # 将字典转换为排序后的列表
    sorted_data = sorted(pig_data.items(), key=lambda x: x[1], reverse=True)
    
    # 找到湖北省的排名
    hubei_rank = 0
    for i, (province, output) in enumerate(sorted_data, 1):
        if province == '湖北省':
            hubei_rank = i
            break
    
    # 打印排名情况
    print("\n=== 生猪出栏量排名情况 ===")
    print(f"湖北省排名第 {hubei_rank} 位")
    print(f"出栏量：{pig_data['湖北省']} 万头")
    print("\n前十名省份：")
    for i, (province, output) in enumerate(sorted_data[:10], 1):
        print(f"{i}. {province}: {output} 万头")

if __name__ == '__main__':
    analyze_pig_data() 