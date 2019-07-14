# coding=utf-8
# 读取MNIST数据

import numpy as np
import struct

def img_loader(filename):
    img_file = open(filename, 'rb')
    file_buffer = img_file.read()
    
    # 读取文件头的魔数、图片数、行数及列数
    magic_num, n_img, n_row, n_col = struct.unpack_from('>IIII', file_buffer, 0)
    assert(magic_num == 2051)
    
    offset = struct.calcsize('>IIII') # 计算数据开始位置的偏移量
    
    # 读取所有图片数据
    img_data = struct.unpack_from('>' + str(n_img * n_row * n_col) + 'B', file_buffer, offset)
    img_data = np.reshape(img_data, (n_img, n_row * n_col)) # 每行存储一张图片
    
    img_file.close()
    print('Load ' + filename + ' succeeded...')
    
    return img_data
    
    
def label_loader(filename):
    label_file = open(filename, 'rb')
    file_buffer = label_file.read()
    
    # 读取文件头的魔数和标签数
    magic_num, n_label = struct.unpack_from('>II', file_buffer, 0)
    assert(magic_num == 2049)
    
    offset = struct.calcsize('>II') # 计算数据开始位置的偏移量
    
    # 读取所有标签数据
    label_data = struct.unpack_from('>' + str(n_label) + 'B', file_buffer, offset)
    label_data = np.reshape(label_data, n_label)
    
    label_file.close()
    print('Load ' + filename + ' succeeded...')
    
    return label_data