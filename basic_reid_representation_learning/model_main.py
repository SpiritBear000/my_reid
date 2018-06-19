
# —————— 读取Market-1501样本数据 ————

# # 训练样本
# import h5py
# f = h5py.File('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/data_extract/Market1501_extractor/Sample_train.h5','r')   #打开h5文件
# print(list(f.keys()))                            #可以查看所有的主键
# x_train = f['data'][:]                    #取出主键为data的所有的键值
# y_train = f['labels'][:]
# f.close()

# 测试样本
import h5py
f = h5py.File('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/data_extract/Market1501_extractor/Sample_test.h5','r')   #打开h5文件
print(list(f.keys()))                            #可以查看所有的主键
x_test = f['data'][:]                    #取出主键为data的所有的键值
y_test = f['labels'][:]
f.close()


'''
# ■■■■■■■■■■■■■ [2]、模型搭建 ■■■■■■■■■■■■■■■

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Activation,MaxPooling2D,Flatten,merge,Conv2DTranspose,ZeroPadding2D
from keras.regularizers import l2

weight_decay=0.0005

# ———————— (1)主模型 ——————
inputs_res = Input(shape=(224,224,3))

# ①抽取模型ResNet50层的输出。
from keras.applications.resnet50 import ResNet50
resnet = ResNet50(include_top=False,input_tensor=inputs_res,weights='imagenet',pooling='max')

# intermediate_layer_model = Model(input=inputs_res,output=resnet.output)
intermediate_layer_model = resnet

# ②[训练样本]的ResNet50特征输出
x_train_ResNet50 = intermediate_layer_model.predict(x_train) # 将样本x_train输入得到'dense_1'层输出。
# 保存[训练样本]提取出的feature
import numpy as np
np.save('ResNet50_features_train.npy', x_train_ResNet50)
train_data = np.load('ResNet50_features_train.npy')
print('x_train_ResNet50', train_data)
print('x_train_ResNet50.shape', train_data.shape)

# ************* 测个时间模块 ************
import datetime
start = datetime.datetime.now()
# ************* 测个时间模块 ************

# ③[测试样本]的ResNet50特征输出
x_test_ResNet50 = intermediate_layer_model.predict(x_test) # 将样本x_test输入得到'dense_1'层输出。
# 保存[测试样本]提取出的feature
import numpy as np
np.save('ResNet50_features_test.npy', x_test_ResNet50)
test_data = np.load('ResNet50_features_test.npy')
print('x_train_ResNet50', test_data)
print('x_train_ResNet50.shape', test_data.shape)

# ************* 测个时间模块 ************
end = datetime.datetime.now()
print(end-start) # 运行时间: 2:28:40'
# ************* 测个时间模块 ************


# (分开保存ResNet生成向量部分)

# ———————— fine-tune模型 ————————

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Activation,MaxPooling2D,Flatten,merge,Conv2DTranspose,ZeroPadding2D
from keras.regularizers import l2

weight_decay=0.0001

inputs = Input(shape=(2048,))
x = Dense(1024,activation='relu', W_regularizer=l2(l=weight_decay))(inputs)
x = Dense(768,activation='relu', W_regularizer=l2(l=weight_decay),name='feature_out')(x)
outputs = Dense(751,activation='softmax', W_regularizer=l2(l=weight_decay))(x)

model = Model(input=inputs,output=outputs)
model.summary()


# ■■■■■■■■■■■■■ [3]、模型编译 ■■■■■■■■■■■■■■■

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# ■■■■■■■■■■■■■ [4]、模型训练 ■■■■■■■■■■■■■■■

# model.fit(x_train,y_train,batch_size=200,) # 主模型训练用这个。starts training

x_fea_train = train_data
model.fit(x_fea_train,y_train,batch_size=256,epochs=18) # fine-tune模型训练用这个。starts training

# # 模型保存
# model.save('./retrieval/fine-tune_model.h5')

'''
# ■■■■■■■■■■■■■ [5]、模型[测试](这个测试不是严格意义上的测试) ■■■■■■■■■■■■■■■

import numpy as np
test_data = np.load('D:/SubjectDownload/PyCharm Community Edition 2017.2.3/PycharmProjects/[My_re-id]/basic_reid_representation_learning/ResNet50_features_test.npy')

from keras.models import load_model
from keras.models import Model
model = load_model('./retrieval_system/fine-tune_model.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('feature_out').output)
feature_out_x_test = intermediate_layer_model.predict(test_data)


from keras.layers import Input,Dense,Conv2D,Activation,MaxPooling2D,Flatten,merge,Conv2DTranspose,ZeroPadding2D
from keras.regularizers import l2

weight_decay=0.0005

inputs_test = Input(shape=(768,))
outputs_test = Dense(750,activation='softmax', W_regularizer=l2(l=weight_decay))(inputs_test)
model_test = Model(inputs_test, outputs_test)
model_test.summary()

model_test.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model_test.fit(feature_out_x_test, y_test ,batch_size=256,epochs=18) # fine-tune模型训练用这个。starts training


loss, accuracy = model_test.evaluate(feature_out_x_test,y_test,batch_size=256)
print('loss:',loss,'\n','accuracy:',accuracy)


# ————————————————————————————————————————————————————————————
from keras import backend as K
K.clear_session()

