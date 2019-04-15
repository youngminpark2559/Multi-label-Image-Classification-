# coding: utf-8

# ================================================================================
# conda activate py36gputf && \
# cd /mnt/1T-5e7/mycodehtml/ensemble/xgboost/Keess324/prj_root/src && \
# rm e.l && python ECE657_Approach1_CNN.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import cv2
import random
import numpy as np 
import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
pal=sns.color_palette()
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from tqdm import tqdm

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# ================================================================================
# c train_label: loaded label data
train_label=pd.read_csv('../Data/train_v2.csv/train_v2.csv')
# print(train_label.head())
#   image_name                                       tags
# 0    train_0                               haze primary
# 1    train_1            agriculture clear primary water
# 2    train_2                              clear primary
# 3    train_3                              clear primary
# 4    train_4  agriculture clear habitation primary road

# ================================================================================
# print("train_label",train_label.shape)
# (40479, 2)

tags=train_label.iloc[:,1]
# print("tags",tags)
# 0                                             haze primary
# 1                          agriculture clear primary water
# 2                                            clear primary

tags=tags.values
# print("tags",tags)
# ['haze primary' 'agriculture clear primary water'

multiclass=[words for segments in tags for words in segments.split()]
# print("multiclass",multiclass)
# ['haze', 'primary',

class_label=set(multiclass)
# print("class_label",class_label)
# {'road', 'habitation',

# print("len(class_label)",len(class_label))
# 17

# ================================================================================
# @ Distribution of label data

def visualize1():
  sumcount=pd.Series(multiclass).value_counts() 
  # print("sumcount",sumcount)
  # primary              37513
  # clear                28431
  # agriculture          12315

  index=sumcount.sort_values(ascending=False).index 
  # print("index",index)
  # Index(['primary', 'clear',

  values=sumcount.sort_values(ascending=False).values
  # print("values",values)
  # [37513 28431 12315

  ax=sns.barplot(y=values,x=index)
  plt.xlabel('Labels',fontsize=18)
  plt.ylabel('Count',fontsize=18)
  ax.set_title("Class Distribution Overview",fontsize=20)
  plt.xticks(rotation=90)
  plt.tick_params(labelsize=12)
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_08:29:23.png

visualize1()

# ================================================================================
# @ Display image data and its corresponding labels (in other words, image's tags)

def visualize2():
  styletype={'grid':False}
  plt.rc('axes',**styletype)
  _,ax=plt.subplots(4,4,sharex='col',sharey='row',figsize=(15,13))

  i=0
  alist=random.sample(range(1,train_label.shape[0]),16)
  # print("alist",alist)
  # [4821, 38060,
  for m in alist:
    one_img_name,one_img_tags=train_label.values[m]
    # print("one_img_name",one_img_name)
    # train_31414
    # print("one_img_tags",one_img_tags)
    # agriculture partly_cloudy primary road water

    name="../Data/train-jpg"
    extion=".jpg"
    name+="/"+one_img_name+extion
    # print("name",name)
    # ../Data/train-jpg/train_11441.jpg
    
    img=cv2.imread(name)
    # print("img",img.shape)
    # (256, 256, 3)

    ax[i//4,i%4].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax[i//4,i%4].set_title('{}-{}'.format(one_img_name,one_img_tags))
    i+=1
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_08:40:00.png

visualize2()

# ================================================================================
# @ Ordering the label class

def fun(l):
  return [item for sublist in l for item in sublist]

tag_to_int_lbl={i:l for l,i in enumerate(class_label)}
# print("tag_to_int_lbl",tag_to_int_lbl)
# tag_to_int_lbl {'agriculture': 0, 'cultivation': 1,

tag_to_int_lbl_itms=tag_to_int_lbl.items()
# print("tag_to_int_lbl_itms",tag_to_int_lbl_itms)
# dict_items([('agriculture', 0), ('cultivation', 1),

order_maplabel={i:l for l,i in tag_to_int_lbl_itms}
# print("order_maplabel",order_maplabel)
# {0: 'agriculture', 1: 'cultivation',

# ================================================================================
tem=[el[1] for el in train_label.values]
# print("tem",tem)
# ['haze primary', 'agriculture clear primary water',

ylabel=[]
for i in tqdm(tem):
  tar_y=np.zeros(len(class_label))
  # print("tar_y",tar_y.shape)
  # (17,)

  str_spl=i.split()
  # print("str_spl",str_spl)
  # ['haze', 'primary']

  for t in str_spl:
    one_lbl=tag_to_int_lbl[t]
    # print("one_lbl",one_lbl)
    # 5

    tar_y[one_lbl]=1

  ylabel.append(tar_y)

# print("ylabel",ylabel)
# [array([1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
#  array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.]),
    
ylabel=np.asarray(ylabel)  
# print("ylabel",ylabel)
# [[1. 0. 1. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 1. 0.]

# print("len(ylabel)",len(ylabel))
# 40479

# ================================================================================
# @ Resize all images into (64,64)

image_resize=(64,64)
batch_size=128

# @ After I save h5 file, I comment this block out
# ximage=[]
# j=0
# for i,value in tqdm(train_label.values):
#   name="../Data/train-jpg"
#   extion=".jpg"
#   name+="/"+i+extion
#   # print("name",name)
#   # ../Data/train-jpg/train_0.jpg

#   load_one_img=cv2.imread(name)
#   # print("load_one_img",load_one_img.shape)
#   # (256, 256, 3)

#   rs_img=cv2.resize(load_one_img,image_resize)
#   # print("rs_img",rs_img.shape)
#   # (64, 64, 3)

#   ximage.append(rs_img)

#   j+=1
    
# ximage=np.asarray(ximage)
# # print("ximage",ximage.shape)
# # (40479, 64, 64, 3)

# ================================================================================
# @ Save numpy array as h5 format file

# @ After I save h5 file, I comment this block out
# with h5py.File('../Data/train_jpg_40479_64_64_3.h5','w') as hf:
#   hf.create_dataset('train_jpg_40479_64_64_3',data=ximage)
#   # train_jpg_40479_64_64_3.h5 file: 497 MB
#   # train-jpg directory: 687M

# ================================================================================
# @ Try to load saved h5 file

with h5py.File('../Data/train_jpg_40479_64_64_3.h5','r') as hf:
  ximage=hf['train_jpg_40479_64_64_3'].value
  # data = hf.get('dataset_name').value
  # print("ximage",ximage.shape)
  # (40479, 64, 64, 3)

# ================================================================================
# @ Training and Testing dataset ratio: 60:40
# @ Validation and Training ration: 60:40 ===> 24287:16192

thre=0.6
threhold=int(np.floor(thre*len(train_label)))
# print("threhold",threhold)
# 24287

# ================================================================================
trainx=ximage[:threhold,:,:,:]
# print("trainx",trainx.shape)
# (24287, 64, 64, 3)

trainy=ylabel[:threhold]
# print("trainy",trainy.shape)
# (24287, 17)

validationx=ximage[threhold:,:,:,:]
# print("validationx",validationx.shape)
# (16192, 64, 64, 3)

validationy=ylabel[threhold:]
# print("validationy",validationy.shape)
# (16192, 17)

# ================================================================================
num_lbls_on_each_trn_img=[]
for i in trainy:
  # print("i",i)
  # i [1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

  num_lbls_on_each_trn_img.append(sum(i))

# print("num_lbls_on_each_trn_img",num_lbls_on_each_trn_img)
# [2.0, 4.0, 2.0,

# ================================================================================
# print("Training Set:")
# print("Number of images labeled with 1 label:{}".format(num_lbls_on_each_trn_img.count(1)))
# print("Number of images labeled with 2 labels:{}".format(num_lbls_on_each_trn_img.count(2)))
# print("Number of images labeled with 3 labels:{}".format(num_lbls_on_each_trn_img.count(3)))
# print("Number of images labeled with 4 labels:{}".format(num_lbls_on_each_trn_img.count(4)))
# print("Number of images labeled with 5 labels:{}".format(num_lbls_on_each_trn_img.count(5)))
# print("Number of images labeled with 6 labels:{}".format(num_lbls_on_each_trn_img.count(6)))
# print("Number of images labeled with 7 labels:{}".format(num_lbls_on_each_trn_img.count(7)))
# print("Number of images labeled with 8 labels:{}".format(num_lbls_on_each_trn_img.count(8)))
# print("Number of images labeled with 9 labels:{}".format(num_lbls_on_each_trn_img.count(9)))

# Training Set:
# Number of images labeled with 1 label: 1280
# Number of images labeled with 2 labels: 11453
# Number of images labeled with 3 labels: 4318
# Number of images labeled with 4 labels: 4336
# Number of images labeled with 5 labels: 2212
# Number of images labeled with 6 labels: 603
# Number of images labeled with 7 labels: 75
# Number of images labeled with 8 labels: 9
# Number of images labeled with 9 labels: 1

# ================================================================================
cnt_train=pd.Series(num_lbls_on_each_trn_img).value_counts() 
# print("cnt_train",cnt_train)
# 2.0    11453
# 4.0     4336
# 3.0     4318

idx_train=cnt_train.sort_values(ascending=False).index 
# print("idx_train",idx_train)
# Float64Index([2.0, 4.0,

vls_train=cnt_train.sort_values(ascending=False).values
# print("vls_train",vls_train)
# [11453  4336

# ================================================================================
# @ Visualize distribution of train dataset

def visualize3():
  # Use bar plot
  ax1=sns.barplot(y=vls_train,x=idx_train)

  # Configure x and y texts
  plt.xlabel('Labels',fontsize=18)
  plt.ylabel('Count',fontsize=18)

  # Configure plot title
  ax1.set_title("Train Multi-label overview",fontsize=20)

  # Rotate x text
  plt.xticks(rotation=90)

  plt.tick_params(labelsize=12)

  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_09:59:02.png

visualize3()

# ================================================================================
num_lbls_on_each_vali_img=[]
for i in validationy:
  # print("i",i)
  # i [1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

  num_lbls_on_each_vali_img.append(sum(i))

# ================================================================================
# print("Validation Set:")
# print("Number of images with 1 label:{}".format(num_lbls_on_each_vali_img.count(1)))
# print("Number of images with 2 labels:{}".format(num_lbls_on_each_vali_img.count(2)))
# print("Number of images with 3 labels:{}".format(num_lbls_on_each_vali_img.count(3)))
# print("Number of images with 4 labels:{}".format(num_lbls_on_each_vali_img.count(4)))
# print("Number of images with 5 labels:{}".format(num_lbls_on_each_vali_img.count(5)))
# print("Number of images with 6 labels:{}".format(num_lbls_on_each_vali_img.count(6)))
# print("Number of images with 7 labels:{}".format(num_lbls_on_each_vali_img.count(7)))
# print("Number of images with 8 labels:{}".format(num_lbls_on_each_vali_img.count(8)))

# Validation Set:
# Number of images with 1 label:811
# Number of images with 2 labels:7723
# Number of images with 3 labels:2876
# Number of images with 4 labels:2855
# Number of images with 5 labels:1485
# Number of images with 6 labels:390
# Number of images with 7 labels:49
# Number of images with 8 labels:3

# ================================================================================
cnt=pd.Series(num_lbls_on_each_vali_img).value_counts()
idx=cnt.sort_values(ascending=False).index
vls=cnt.sort_values(ascending=False).values

# ================================================================================
def visualize4():
  ax=sns.barplot(y=vls,x=idx)
  plt.xlabel('Labels',fontsize=18)
  plt.ylabel('Count',fontsize=18)
  ax.set_title("Validation multi-label Overview",fontsize=20)
  plt.xticks(rotation=90)
  plt.tick_params(labelsize=12)

visualize4()

# ================================================================================
# @ CNN Modelling 

# c model_1: model1 sequence
model_1=Sequential()

# Conv1
model_1.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
# Conv2
model_1.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
# Maxpooling1
model_1.add(MaxPooling2D(pool_size=(2,2)))
# Conv3
model_1.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
# Dropout
model_1.add(Dropout(0.5))
# Flat array
model_1.add(Flatten())
# FC
model_1.add(Dense(128,activation='relu'))
# Dropout
model_1.add(Dropout(0.5))
# FC
model_1.add(Dense(17,activation='sigmoid'))
# We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
model_1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# Run train
model_1.fit(
  trainx,trainy,batch_size=128,epochs=5,verbose=1,validation_data=(validationx,validationy))

# ================================================================================
# @ CNN Model 2 (It is called model 1 in report)

# c model_2: model2 sequence
model_2=Sequential()

# Conv1
model_2.add(Conv2D(32,kernel_size=(3,3),strides=2,activation='relu',input_shape=(64,64,3)))
# Maxpooling1
model_2.add(MaxPooling2D(pool_size=(2,2)))
# Conv2
model_2.add(Conv2D(48,(3,3),activation='relu'))
# Maxpooling2
model_2.add(MaxPooling2D(pool_size=(2,2)))
# Dropout
model_2.add(Dropout(0.5))
# Flat array
model_2.add(Flatten())
# FC
model_2.add(Dense(128,activation='relu'))
# Dropout
model_2.add(Dropout(0.5))
# FC
model_2.add(Dense(17,activation='sigmoid'))
# We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
model_2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# CNN Model2 --- 3 epochs
model_2.fit(
  trainx,trainy,batch_size=128,epochs=3,verbose=1,validation_data=(validationx,validationy))
# CNN Model2 --- 5 epochs
model_2.fit(
  trainx,trainy,batch_size=128,epochs=5,verbose=1,validation_data=(validationx,validationy))

# ================================================================================
# @ Use validation data

test_validationy=model_2.predict(validationx)
# print("test_validationy",test_validationy)
# [[4.1022849e-01 1.6631652e-01 2.1586376e-03 ... 6.4991966e-02
#   5.9920963e-04 8.9649647e-01]
#  [3.1384403e-01 2.0036967e-01 5.0277100e-03 ... 1.5764853e-02
#   1.8442185e-03 7.5833780e-01]

# ================================================================================
# @ CNN Model2 --- 10 epochs

model_2.fit(
  trainx,trainy,batch_size=128,epochs=10,verbose=1,validation_data=(validationx,validationy))

# ================================================================================
# @ CNN model2 ----15 epochs 

model_2.fit(
  trainx,trainy,batch_size=128,epochs=15,verbose=1,validation_data=(validationx,validationy))

# Test results after 15 epochs
test_validationy15ep=model_2.predict(validationx)
print("test_validationy15ep",test_validationy15ep)

# ================================================================================
# @ CNN model2 --- 20 epochs

model_2.fit(
  trainx,trainy,batch_size=128,epochs=20,verbose=1,validation_data=(validationx,validationy))

# Test result after 20 epochs
test_validationy20ep=model_2.predict(validationx)
print("test_validationy20ep",test_validationy20ep)

# ================================================================================
# @ Threshold modification session

# @ Evaluations ---- Train  vs. Validation 
print(test_validationy20ep[0])
print(validationy[0])
print("-------")
Threshold = 0.55

newtestvali = []
for i in test_validationy20ep:
    i = np.array(i)
    i[i>= Threshold]=1
    i[i<= Threshold]=0
    newtestvali.append(i)
    
# Examples
print(newtestvali[0])
print(validationy[0])

# ================================================================================
# @ Evaluations

# ================================================================================
# @ 1. Exact Match Ratio (MR)

def MR_Measure (target,predict):
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        if i==j:
            msurelist.append(0)
        else:
            msurelist.append(1)


    Exact_Match_Ratio = 1- sum(msurelist)/len(msurelist)

    print("Exact Match Ratio is: {} ".format(Exact_Match_Ratio))
    

MR_Measure(validationy,newtestvali)

# ================================================================================
# @ 2. Hamming Loss (HL)

def HL_measure(target,predict):
    target_flat = [item for sublist in target for item in sublist]
    predict_flat = [item for sublist in predict for item in sublist]
    print (len(target_flat))
    print (len(predict_flat))

    #TP:
    list1 = [x + y for x, y in zip(target_flat, predict_flat)]
    TP = list1.count(2)
    print("True Positives: {} ".format(TP))
    #TN:
    list1 = [x + y for x, y in zip(target_flat, predict_flat)]
    TN = list1.count(0)
    print("True Negatives: {} ".format(TN))
    #FP:
    list2 = [x - y for x, y in zip(target_flat, predict_flat)]
    FP = list2.count(-1)
    print("False Positives: {} ".format(FP))
    #FN:
    list2 = [x - y for x, y in zip(target_flat, predict_flat)]
    FN = list2.count(1)
    print("False Negatives: {} ".format(FN))
    
    #Precision
    Precision = TP/(TP+FP)
    print("Precision is: {} ".format(Precision))
    
    #Recall
    Recall = TP/(TP+FN)
    print("Recall is: {} ".format(Recall))
    
    #Accuracy
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    print("Accuracy is: {} ".format(Accuracy))  
    
    #F_measure
    F_measure = (TP*2)/(2*TP+FP+FN)
    print("F1_Measure is: {} ".format(F_measure)) 
    
    # Hamming Loss 
    Hamming_Loss = 1-Accuracy

    print("Hamming Loss is: {} ".format(Hamming_Loss))
    

HL_measure(validationy,newtestvali)
    
# ================================================================================
# @ 3. Godbole et Measure (Considering the partical correct)

def GodBole(target,predict):
    target_flat = [item for sublist in target for item in sublist]
    predict_flat = [item for sublist in predict for item in sublist]
    print (len(target_flat))
    print (len(predict_flat))
    
    #Accuracy
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        A_lower = len(list1) - list1.count(0)
        Acc = A_upper/ A_lower
        msurelist.append(Acc)
    GB_Accuracy = sum(msurelist)/len(target)
    print("GodBole Accuracy is: {} ".format(GB_Accuracy)) 
    
    
    #Precision
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        A_lower = sum(i)
        Pre = A_upper/ A_lower
        msurelist.append(Pre)
    GB_Precision = sum(msurelist)/len(target)
    print("GodBole Precision is: {} ".format(GB_Precision))
    
    #Recall
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        if sum(j) != 0:
            A_lower = sum(j)
            Rec = A_upper/ A_lower
            msurelist.append(Rec)
        else:
            A_lower = 1
            Rec = A_upper/ A_lower
            msurelist.append(Rec)
                
    GB_Recll = sum(msurelist)/len(target)
    print("GodBole Recall is: {} ".format(GB_Recll))
    

    #F_measure
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        A_lower = sum(j)+sum(i)
        F = (2*A_upper)/ A_lower
        msurelist.append(F)
    GB_F_measure = sum(msurelist)/len(target)
    print("GodBole F1_Measure is: {} ".format(GB_F_measure)) 

GodBole(validationy,newtestvali)
