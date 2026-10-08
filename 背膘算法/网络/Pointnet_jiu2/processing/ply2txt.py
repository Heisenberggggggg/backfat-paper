import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd

path=r'E:\first_pcl\ply'
txtpath=r'E:\first_pcl\txt'
#file_dir = 'D:\ltf\cixiangshuanzuo\\train\\ts_0_0b5e12a364b4efd8536afdc4243985d3_l.ply'  # 文件的路径
namelist=[x for x in os.listdir(path)]
for i in range( len(namelist) ):
    file_dir =os.path.join(path,namelist[i])
    plydata = PlyData.read(file_dir)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.float64)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for j, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, j] = data_pd[name]
    data_np=np.float32(data_np)
    np.set_printoptions(suppress=True)
    np.savetxt('%s/%s'%(txtpath,namelist[i].split('.')[0]+'.txt'), data_np,fmt='%.06f')

#print(data_np)