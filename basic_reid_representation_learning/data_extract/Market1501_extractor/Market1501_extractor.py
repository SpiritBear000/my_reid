
import numpy as np
from PIL import Image
import os



# —————————— 获取[train、test、query]文件夹路径 ————————
def get_path(train_or_test = 'train'):
    if train_or_test == 'train':
        image_path = 'D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/_datasets_/Market-1501/bounding_box_train'
        return image_path
    elif train_or_test == 'test':
        image_path = 'D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/_datasets_/Market-1501/bounding_box_test'
        return image_path
    elif train_or_test == 'query':
        image_path = 'D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/_datasets_/Market-1501/query'
        return image_path


# —————————— 获取[train、test、query]文件夹中的图片名称列表 ————————
def get_image_path_list(train_or_test = 'train'):

    global folder_path # 这样，在函数外面修改folder_path这个变量，这个变量的值也会改变。
    if train_or_test == 'train':
        folder_path = 'D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/_datasets_/Market-1501/bounding_box_train'
    elif train_or_test == 'test':
        folder_path = 'D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/_datasets_/Market-1501/bounding_box_test'
    elif train_or_test == 'query':
        folder_path = 'D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/_datasets_/Market-1501/query'
    assert os.path.isdir(folder_path)

    if train_or_test == 'train' or train_or_test == 'query':
        return sorted(os.listdir(folder_path))
        # 返回folder_path路径下的文件名列表，并且通过sorted()
        # 进行排序(顺序就是文件夹中按照名称的顺序),返回是列表。

    elif train_or_test == 'test':
        return sorted(os.listdir(folder_path))[6617:]
        # 返回folder_path路径下的文件名列表，并且通过sorted()进行排序,
        # 返回列表第6618位置及其之后的元素。


# ■■■■■■■■■■■■■ [1]、数据准备 ■■■■■■■■■■■■■■■

'''
# □□□□□□ [训练]样本准备 □□□□□□

train_list=get_image_path_list(train_or_test = 'train') # 获取train样本的图片名称列表。
train_list.remove('Thumbs.db') # 删除列表中最后一项'Thumbs.db'文件名

train_sample = []
for i, list_index in enumerate(train_list):
    print('第i=',i,'张图片。\t','图片名称：',list_index)

    train_image_temp = Image.open('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/'
                                      'PycharmProjects/[My_re-id]/_datasets_/Market-1501/'
                                      'bounding_box_train/' + list_index)
                                # 提取的图片的array形状：(128, 64, 3(R,G,B))
    train_image_temp = train_image_temp.resize((224, 224), Image.ANTIALIAS)
    train_image_temp = np.array(train_image_temp)
    train_sample.append([train_image_temp,int(list_index[0:4])])

print(train_sample[0][0].shape)
x_train = []
y_train = []
for train_index in train_sample:
    x_train.append(train_index[0])
    y_train.append(train_index[1])

# ———— 转换一下标签y_train中的标签名称：(2, 4, ... , 1500) ——> ( 1, 2 , 3 ..., 750 );目的:好进行one-hot编码.
y_train_temp = []
for m_index in y_train:
    if not m_index in y_train_temp:
        y_train_temp.append(m_index)
print('y_train_temp',y_train_temp)

k = 0
new_y_train = []
for index in y_train:
    if index == y_train_temp[k]:
        new_y_train.append(k)
    else:
        k=k+1
        new_y_train.append(k)
print('new_y_train',new_y_train)

y_train = new_y_train
# ———— 转换一下标签名称[结束]

x_train = np.array(x_train)

y_train = np.array(y_train)
y_train = (np.arange(751) == y_train[:, None]).astype(int)
print('y_train',y_train)


# ———— 存储样本数据 ————

# 存储
import h5py
f = h5py.File('Sample_train.h5','w')   #创建一个h5文件，文件指针是f
f['data'] = x_train                 #将数据写入文件的主键data下面
f['labels'] = y_train               #将数据写入文件的主键labels下面
f.close()                           #关闭文件

# # 读取
# import h5py
# f = h5py.File('Sample.h5','r')   #打开h5文件
# print(list(f.keys()))                            #可以查看所有的主键
# x_train = f['data'][:]                    #取出主键为data的所有的键值
# y_train = f['labels'][:]
# f.close()
#
# print('x_train',x_train)
# print('x_train',x_train.shape)
# print('y_train',y_train)
# print('y_train',y_train.shape)

'''
# □□□□□□ [测试]样本准备 (使用这个时注释掉上面的 训练样本准备) □□□□□□

if __name__ == '__main__':
    test_list=get_image_path_list(train_or_test = 'test') # 获取test样本的图片名称列表。
    test_list.remove('Thumbs.db') # 删除列表中最后一项'Thumbs.db'文件名
    print('test_list',test_list) # 不含有 '-1' 和 '0000' 的标记数据.

    x_test = []
    y_test = []
    for test_im in test_list:

        test_image_temp = Image.open('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/'
                                     'PycharmProjects/[My_re-id]/_datasets_/Market-1501/'
                                     'bounding_box_test/' + test_im)
                                     # 提取的图片的array形状：(128, 64, 3(R,G,B))
        test_image_temp1 = test_image_temp.resize((224, 224), Image.ANTIALIAS)
        test_image_temp2 = np.array(test_image_temp1)
        x_test.append(test_image_temp2)

        y_test.append(int(test_im[0:4]))

    # 转换一下标签y_test中的标签名称：(1, 3, ... , 1501) ——> ( 1, 2 , 3 ..., 750 );目的:好进行one-hot编码.
    y_test_temp = []
    for m_index in y_test:
        if not m_index in y_test_temp:
            y_test_temp.append(m_index)
    print('y_test_temp',y_test_temp)

    k = 0
    new_y_test = []
    for index in y_test:
        if index == y_test_temp[k]:
            new_y_test.append(k)
        else:
            k=k+1
            new_y_test.append(k)
    print('new_y_test',new_y_test)
    y_test = new_y_test
    # 转换一下标签名称[结束]


    x_test = np.array(x_test)
    print('x_test.shape',x_test.shape)

    y_test = np.array(y_test)
    print('y_test',y_test)
    print('y_test.shape',y_test.shape)
    y_test = (np.arange(750) == y_test[:, None]).astype(int) # one-hot编码
    print('y_test_2',y_test)
    print('y_test_2.shape',y_test.shape)


    # ———— 存储样本数据 ————

    # 存储
    import h5py
    f = h5py.File('Sample_test.h5','w')   #创建一个h5文件，文件指针是f
    f['data'] = x_test                 #将数据写入文件的主键data下面
    f['labels'] = y_test               #将数据写入文件的主键labels下面
    f.close()                           #关闭文件

    # # 读取
    # import h5py
    # f = h5py.File('Sample.h5','r')   #打开h5文件
    # print(list(f.keys()))                            #可以查看所有的主键
    # x_test = f['data'][:]                    #取出主键为data的所有的键值
    # y_test = f['labels'][:]
    # f.close()
    #
    # print('x_test',x_test)
    # print('x_test',x_test.shape)
    # print('y_test',y_test)
    # print('y_test',y_test.shape)
