"""
.niigz converts to .npy V2
- 16bit
- standardization
- 256x256x256
- 5 categories
@Author:    leowizard
@Version:   2024-06-12
"""
import os
import numpy as np
import nibabel as nib
from multiprocessing import Pool, Process

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def numpy_to_standardization(ct_data):
    """
    Converting 3D ct numpy array to tensor standardize as [-1.0,1.0]
    args:
        [channels, height, width, depth]
    """
    ct_value_mean = 0.10209798251413127
    ct_value_std = 0.18859518628903846
    #仅在非零区域执行标准化操作
    nonzero_mask = ct_data[...,0] > 0        
    ct_data[...,0] = np.where(nonzero_mask, (ct_data[...,0] - ct_value_mean) / ct_value_std, ct_data[...,0])       
    return ct_data

def interpolate_cube(data):
    # 假设你有一个512x512x512的体素数据，名为data
    # data = your_512_voxel_data

    # 创建一个三维坐标网格
    x = np.linspace(0, 511, 512)
    y = np.linspace(0, 511, 512)
    z = np.linspace(0, 511, 512)

    # 创建一个RegularGridInterpolator对象
    interpolator = RegularGridInterpolator((x, y, z), data, method='linear', bounds_error=False, fill_value=0)

    # 新的256x256x256体素数据的网格
    new_x = np.linspace(0, 511, 256)
    new_y = np.linspace(0, 511, 256)
    new_z = np.linspace(0, 511, 256)

    # 在新的网格上进行插值
    new_grid = np.meshgrid(new_x, new_y, new_z, indexing='ij')
    new_data = interpolator(np.stack(new_grid, axis=-1))
    return new_data


# new_data 现在包含了插值后的256立方体素数据
def process_file(file_path):
    
    # 读取 .nii.gz 文件
    nii_data = nib.load(file_path).get_fdata()    
    nii_data = interpolate_cube(nii_data).astype(np.float16)
    nii_data = numpy_to_standardization(nii_data)
    # 将数据保存为 .npy 文件，文件名保持一致，但扩展名变为 .npy
    # print(os.path.splitext(file_path)[0][:-4])        
    npy_file_path = os.path.splitext(file_path)[0][:-4] + '.npy'        
    npy_file_path = os.path.join('/opt/data/private/dataset/CT_Database/ctdata_fti_npy', npy_file_path.split('/')[-1])
    np.save(npy_file_path, nii_data)
    print(f"Saved {npy_file_path}")


def single_process_file(files):
    print('starting convertion......')
    for file in files:
         process_file(file)

if __name__ == "__main__":
    # 指定包含 .nii.gz 文件的文件夹路径
    folder_path = "/opt/data/private/dataset/CT_Database/FTI"

    # 获取文件夹下的所有 .nii.gz 文件路径
    nii_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.nii.gz')]

    # p1 = Process(target=single_process_file, args=(nii_files[:2000],))
    # p2 = Process(target=single_process_file, args=(nii_files[2000:4000],))
    # p3 = Process(target=single_process_file, args=(nii_files[4000:6000],))
    # p4 = Process(target=single_process_file, args=(nii_files[6000:8802],))
    p1 = Process(target=single_process_file, args=(nii_files[:70],))
    p2 = Process(target=single_process_file, args=(nii_files[70:140],))
    p3 = Process(target=single_process_file, args=(nii_files[140:220],))
    p4 = Process(target=single_process_file, args=(nii_files[220:],))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()



    # # multi process
    # process_list = []    
    # for index in range(10):        
    #         selected_files = nii_files[index:(index+1)*880]
    #         if index == 9:
    #              selected_files = nii_files[index:(index+1)*880+2]
    #         process_list.append(Process(target=single_process_file, args=(
    #                   [selected_files]
    #         )))
    # for single_process in process_list:
    #     single_process.start()
    # for single_process in process_list:
    #     single_process.join()
