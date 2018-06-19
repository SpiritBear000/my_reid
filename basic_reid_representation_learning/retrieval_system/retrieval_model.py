
# ———————— 函数定义区 ————————————

# 计算两个向量的欧氏距离
def calEuclideanDistance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

# ————————————————————————————————



# ■■■■■■■■■■■■■ [0]、读取Market-1501样本数据 ■■■■■■■■■■■■■■■

# # 训练样本
# import h5py
# f = h5py.File('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/data_extract/Market1501_extractor/Sample_train.h5','r')   #打开h5文件
# print(list(f.keys()))                            #可以查看所有的主键
# x_train = f['data'][:]                    #取出主键为data的所有的键值
# y_train = f['labels'][:]
# f.close()
#
# # 测试样本
# import h5py
# f = h5py.File('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/data_extract/Market1501_extractor/Sample_test.h5','r')   #打开h5文件
# print(list(f.keys()))                            #可以查看所有的主键
# x_test = f['data'][:]                    #取出主键为data的所有的键值
# y_test = f['labels'][:]
# f.close()

# [训练]样本的 ResNet_fea
import numpy as np
train_data = np.load('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/ResNet50_features_train.npy')

# [测试]样本的 ResNet_fea
test_data = np.load('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/ResNet50_features_test.npy')


# ■■■■■■■■■■■■■ [6]、检索系统建立 ■■■■■■■■■■■■■■■

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Activation,MaxPooling2D,Flatten,merge,Conv2DTranspose,ZeroPadding2D
from keras.regularizers import l2
from keras.models import load_model

# ———— (1)用训练好的模型 提取特征 ——
model = load_model('fine-tune_model.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('feature_out').output)

# [训练集]图片特征
feature_out_x_train = intermediate_layer_model.predict(train_data)
# [测试集]图片特征
feature_out_x_test = intermediate_layer_model.predict(test_data)

# ———— (2) 整个测试集当做query 选一张测试集图片当做probe ————

# 获取图片名称list
from data_extract.Market1501_extractor.Market1501_extractor import get_image_path_list

test_list=get_image_path_list(train_or_test = 'test') # 获取test样本的图片名称列表。
test_list.remove('Thumbs.db') # 删除列表中最后一项'Thumbs.db'文件名

# [待检索图片输入]:
probe_image = '0745_c1s4_044206_04.jpg' # probe图片用 测试集 中的图片

probe_index = 0
for index, image in enumerate(test_list):
    if probe_image == image:
        probe_index = index

#  计算特征向量之间的距离
distance=[]
for test_fea in feature_out_x_test:
    dis = calEuclideanDistance(test_fea, feature_out_x_test[probe_index])
    distance.append(dis.item())
distance_temp = distance

# 提取出与待检索图片最相似的 前rank个图片
index_min_distance = [] # 用于下面for循环存储 最小距离 在 distance列表中的位置。
Inf = 1000000000
rank = 10 # rank-k的值
for r_k in range(rank):
    index_min_distance.append(distance_temp.index(min(distance_temp)))  # 选出distance_temp(type:list)中的最小值所在的位置,并放入index_min_distance中
    distance_temp[distance_temp.index(min(distance_temp))]=Inf # 将之前最小值的位置用一个特别大的值Inf来代替。

print('index_min_distance',index_min_distance)

# ———— (3) 显示图片 ————
from PIL import Image
from matplotlib import pyplot as plt

fig = plt.figure()

# 显示待检索图片
from data_extract.Market1501_extractor.Market1501_extractor import get_path # 获取图片库路径

probe_img = Image.open(get_path(train_or_test = 'test')+ '/' + probe_image)
pi = fig.add_subplot(2,6,1) # 建立一个2*6的图片区域.
pi.set_xlabel('probe')
pi.imshow(probe_img)

# 显示被检索出的图片
for imd_index, imd in enumerate(index_min_distance):
    img = Image.open(get_path(train_or_test = 'test') +'/' + test_list[imd])
    print('imd_index',imd_index)
    ax = fig.add_subplot(2,6,3+imd_index)
    ax.set_xlabel('rank' + str(1+imd_index))
    ax.imshow(img)

plt.show()#显示刚才所画的所有操作


# ■■■■■■■■■■■■■ [7]、检索识别率 ■■■■■■■■■■■■■■■

def retrieval_accuracy(test_list, feature_out_x_test):
    test_list_temp = test_list
    true_sample = 0
    total_sample = 0
    for probe_image_acc in test_list_temp:
        for index, image in enumerate(test_list):
            if probe_image_acc == image:
                probe_index = index

        # 计算特征向量之间的距离
        distance = []
        for test_fea in feature_out_x_test:
            dis = calEuclideanDistance(test_fea, feature_out_x_test[probe_index])
            distance.append(dis.item())
        distance_temp = distance

        # 提取出与待检索图片最相似的 前rank个图片
        index_min_distance = []  # 用于下面for循环存储 最小距离 在 distance列表中的位置。
        Inf = 1000000000
        rank = 10  # rank-k的值
        for r_k in range(rank):
            index_min_distance.append(distance_temp.index(min(distance_temp)))  # 选出distance_temp(type:list)中的最小值所在的位置,并放入index_min_distance中
            distance_temp[distance_temp.index(min(distance_temp))] = Inf  # 将之前最小值的位置用一个特别大的值Inf来代替。

        print(probe_image_acc,'的index_min_distance:', index_min_distance)

        if probe_image_acc[0:4]==test_list[index_min_distance[1]][0:4]:
            true_sample = true_sample + 1
        total_sample = total_sample + 1
    accuracy = float(true_sample)/float(total_sample)

    return accuracy


print(retrieval_accuracy(test_list, feature_out_x_test)) # accuracy = 0.4661075104841784



# ————————————————————————————————————————————————————————————
from keras import backend as K
K.clear_session()