# coding: utf-8

# conda activate py36gputf && \
# cd /mnt/1T-5e7/mycodehtml/ensemble/xgboost/Keess324/prj_root/src && \
# rm e.l && python ECE657_Approach2_XGBoost.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
pal=sns.color_palette()
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from tqdm import tqdm
import cv2
import statistics
import pylab as pl
import scipy
from collections import Counter
import time
from xgboost import XGBClassifier
from sklearn.all_kind_of_label import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# ================================================================================
# c train_label: label data for train images
train_label=pd.read_csv('../Data/train_v2.csv/train_v2.csv')
# print("train_label.head()",train_label.head())
#   image_name                                       tags
# 0    train_0                               haze primary
# 1    train_1            agriculture clear primary water
# 2    train_2                              clear primary
# 3    train_3                              clear primary
# 4    train_4  agriculture clear habitation primary road

# ================================================================================
tags_of_trn_lbl=train_label['tags']
# print("tags_of_trn_lbl",tags_of_trn_lbl)
# 0                                             haze primary
# 1                          agriculture clear primary water
# 2                                            clear primary
# 3                                            clear primary

tags_of_trn_lbl_li=tags_of_trn_lbl.values
# print("tags_of_trn_lbl_li",tags_of_trn_lbl_li)
# ['haze primary' 'agriculture clear primary water' 'clear primary' ...

all_kind_of_label=[]
for segments in tags_of_trn_lbl_li:
  for words in segments.split():
    all_kind_of_label.append(words)

all_kind_of_label=set(all_kind_of_label)
# print("all_kind_of_label",all_kind_of_label)
# ['haze', 'primary', 'agriculture',

# ================================================================================
def fun(l):
  return [item for sublist in l for item in sublist]

# ================================================================================
lbls_to_ints={}
for l,i in enumerate(all_kind_of_label):
  lbls_to_ints[i]=l

# print("lbls_to_ints",lbls_to_ints)
# {'water': 0, 'habitation': 1,

# ================================================================================
itms_of_int_lbls=lbls_to_ints.items()
# print("itms_of_int_lbls",itms_of_int_lbls)
# dict_items([('road', 0), ('selective_logging', 1),

pair_of_idx_and_int_lbl={}
for l,i in itms_of_int_lbls:
  pair_of_idx_and_int_lbl[i]=l
# print("pair_of_idx_and_int_lbl",pair_of_idx_and_int_lbl)
# {0: 'partly_cloudy', 1: 'agriculture',

# ================================================================================
# @ Convert each label into one hot representation

lbls_of_each_img=[]
for el in train_label.values:
  one_tag=el[1]
  # print("one_tag",one_tag)
  # haze primary

  lbls_of_each_img.append(one_tag)
# print("lbls_of_each_img",lbls_of_each_img)
# ['haze primary', 'agriculture clear primary water',

# ================================================================================
ylabel=[]
for i in tqdm(lbls_of_each_img):
  # c tar_y: placeholder (17,) shape 1D array for each single label
  tar_y=np.zeros(17)

  # print("i",i)
  # haze primary
  str_spl=i.split()
  # print("str_spl",str_spl)
  # ['haze', 'primary']

  for t in str_spl:
    idx_of_single_lbl=lbls_to_ints[t]
    # print("idx_of_single_lbl",idx_of_single_lbl)
    # 0

    tar_y[idx_of_single_lbl]=1
  
  ylabel.append(tar_y)        

# print("ylabel",ylabel)
# [array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
#  array([1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.]),

ylabel=np.asarray(ylabel)
# print("ylabel",ylabel)
# [[1. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 1. 0. 1.]

# print("ylabel",ylabel.shape)
# (40479, 17)

# ================================================================================
# @ Feature Extraction

IMG=cv2.imread('../Data/train-jpg/train_1.jpg')

img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)
# print("img.shape",img.shape)
# (256, 256, 3)

# print("train_label.values[1]",train_label.values[1])

def visualize_RGB_img():
  plt.imshow(img)
  plt.grid(False)
  plt.colorbar()
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:15:18.png
# visualize_RGB_img()

# ================================================================================
IMG=cv2.imread('../Data/train-jpg/train_1.jpg')
img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)

def visualize_B_layer_of_img():
  Blue=img
  Blue[:,:,0]=0
  Blue[:,:,1]=0
  # print("Blue",Blue.shape)
  # (256, 256, 3)
  plt.imshow(Blue)
  plt.grid(False)
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:21:41.png
# visualize_B_layer_of_img()

# ================================================================================
IMG=cv2.imread('../Data/train-jpg/train_1.jpg')
img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)

def visualize_G_layer_of_img():
  Green = img
  Green[:,:,0]=0
  Green[:,:,2]=0
  plt.imshow(Green)
  plt.grid(False)
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:21:58.png
# visualize_G_layer_of_img()

# ================================================================================
IMG=cv2.imread('../Data/train-jpg/train_1.jpg')
img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)

def visualize_R_layer_of_img():
  Red=img
  Red[:,:,1]=0
  Red[:,:,2]=0
  plt.imshow(Red)
  plt.grid(False)
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:22:11.png
# visualize_R_layer_of_img()

# ================================================================================
# @ RGB Distributions --- Bimodel

IMG=cv2.imread('../Data/train-jpg/train_1.jpg')
img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)

def RGB_features(img):
  
  # @ R channal
  R=img[:,:,0].flatten()
  # print("R",R.shape)
  # (65536,)
  # print("R",R)
  # [ 48  52  57 ...  87  94 100]

  r=img[:,:,0].flatten().mean()
  # print("r",r)
  # 66.63986206054688

  r_median = statistics.median(R) # Media Of Red Channal
  # print("r_median",r_median)
  # 60.0

  # Bi-model
  r_leftmodel=R[R<r]
  r_rightmodel=R[R>r]

  # find mode
  r_leftmode=statistics.mode(r_leftmodel)
  # print("r_leftmode",r_leftmode)
  # 56

  r_rightmode=statistics.mode(r_rightmodel)
  # print("r_rightmode",r_rightmode)
  # 90
  
  # Difference 
  r_difmode=abs(r_rightmode-r_leftmode)
  # print("r_difmode",r_difmode)
  # 34
  
  # ================================================================================
  print("The mean and median of the red distribution is {} and {}".format(r.round(2), r_median.round(2)))
  print("The bimodel has two modes and difference are {} and {} and {}".format(r_leftmode, r_rightmode, r_difmode))
  def visual_r_dist():
    plt.hist(R,color='red',bins=100)
    plt.axvline(r,color='brown',linewidth=3)
    plt.axvline(r_median,color="gray",linewidth=3)
    plt.axvline(r_leftmode,color='orange',linewidth=2)
    plt.axvline(r_rightmode,color='orange',linewidth=2)
    plt.show()
    # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:36:33.png
  # visual_r_dist()
  
  # ================================================================================
  # @ G channal

  G=img[:,:,1].flatten()
  g=img[:,:,1].flatten().mean()
  g_median=statistics.median(G) # Media Of Red Channal

  # Bi-model
  g_leftmodel=G[G<g]
  g_rightmodel=G[G>g]

  # find mode 
  g_leftmode=statistics.mode(g_leftmodel)
  g_rightmode=statistics.mode(g_rightmodel)
  
  # Difference 
  g_difmode=abs(g_rightmode-g_leftmode)
  
  print("The mean and median of the Green distribution is {} and {}".format(g.round(2), g_median.round(2)))
  print("The bimodel has two modes and difference are {} and {}  and {}".format(g_leftmode, g_rightmode, g_difmode))

  def visual_g_dist():
    plt.hist(G,color='green',bins=100)
    plt.axvline(g,color='brown',linewidth=3)
    plt.axvline(g_median,color="gray",linewidth=3)
    plt.axvline(g_leftmode,color='orange',linewidth=2)
    plt.axvline(g_rightmode,color='orange',linewidth=2)
    plt.show()
    # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:37:55.png
  # visual_g_dist()
  
  # ================================================================================
  # @ B channal

  B=img[:,:,2].flatten()
  b=img[:,:,2].flatten().mean()
  b_median=statistics.median(B) # Media Of Red Channal

  # Bi-model
  b_leftmodel=B[B<b]
  b_rightmodel=B[B>b]

  # find mode 
  b_leftmode=statistics.mode(b_leftmodel)
  b_rightmode=statistics.mode(b_rightmodel)
  
  # Difference 
  b_difmode=abs(b_rightmode-b_leftmode)
  
  print("The mean and median of the Blue distribution is {} and {}".format(b.round(2), b_median.round(2) ))
  print("The bimodel has two modes and difference are {} and {} and {}".format(b_leftmode, b_rightmode, b_difmode))

  def visual_b_dist():
    plt.hist(G,color='blue',bins=100)
    plt.axvline(g,color='brown',linewidth=3)
    plt.axvline(g_median,color="gray",linewidth=3)
    plt.axvline(g_leftmode,color='orange',linewidth=2)
    plt.axvline(g_rightmode,color='orange',linewidth=2)
    plt.show()
    # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:41:14.png
  
  # visual_b_dist()

# RGB_features(img)

# ================================================================================
# RGB Feature Extraction Function

def RBG(img):
  R=img[:,:,0].flatten()
  r=img[:,:,0].flatten().mean()
  r_median=statistics.median(R) # Media Of Red Channal
  data=Counter(R)
  # print("data",data)
  # Counter({92: 6458, 93: 6357, 91: 6198,

  r_mode=data.most_common(1)[0][0] # Mode  Of Red Channal
  r_std=np.std(R) #Standard Deviation Of Red Channal
  r_max=np.max(R) #Maximum Of Red Channal
  r_min=np.min(R) #Minmum Of Red Channal
  r_kurtosis=scipy.stats.kurtosis(R) #kurtosis Of Red Channal
  r_skew=scipy.stats.skew(R) #skew Of Red Channal

  # ================================================================================
  # Bi-model
  r_leftmodel=R[R<r]
  r_rightmodel=R[R>r]

  # find mode
  data=Counter(r_leftmodel)
  r_leftmode=data.most_common(1)[0][0]
  data=Counter(r_rightmodel)
  r_rightmode=data.most_common(1)[0][0]

  # Difference
  r_difmode=abs(r_rightmode-r_leftmode)
  
  # ================================================================================
  G=img[:,:,1].flatten()
  g=img[:,:,1].flatten().mean() # mean  Of Green Channal
  g_median=statistics.median(G) # Media  Of Green Channal
  data=Counter(G)
  g_mode=data.most_common(1)[0][0] # Mode  Of Green Channal
  g_std=np.std(G) #Standard Deviation Of Green Channal
  g_max=np.max(G) #Maximum Of Green Channal
  g_min=np.min(G) #Minmum Of Green Channal
  g_kurtosis=scipy.stats.kurtosis(G) #kurtosis Of Green Channal
  g_skew=scipy.stats.skew(G) #skew Of Green Channal
  
  # Bi-model
  g_leftmodel=G[G<g]
  g_rightmodel=G[G>g]

  # two mode 
  data=Counter(g_leftmodel)
  g_leftmode=data.most_common(1)[0][0]
  
  data=Counter(g_rightmodel)
  g_rightmode=data.most_common(1)[0][0]

  # Difference 
  g_difmode=abs(g_rightmode-g_leftmode)
  
  # ================================================================================
  B=img[:,:,2].flatten()
  b=img[:,:,2].flatten().mean() # mean of Blue channal
  b_median=statistics.median(B) # Media Of Blue Channal
  data=Counter(B)
  b_mode=data.most_common(1)[0][0] # Mode  Of Blue Channal
  b_std=np.std(B) #Standard Deviation Of Blue Channal
  b_max=np.max(B) #Maximum Of Blue Channal
  b_min=np.min(B) #Minmum  Of Blue Channal
  b_kurtosis=scipy.stats.kurtosis(B) #kurtosis  Of Blue Channal
  b_skew=scipy.stats.skew(B) #skew  Of Blue Channal
  
  # Bi-model
  b_leftmodel=B[B<b]
  b_rightmodel=B[B>b]

  # find mode 
  data=Counter(b_leftmodel)
  b_leftmode=data.most_common(1)[0][0]
  data=Counter(b_rightmodel)
  b_rightmode=data.most_common(1)[0][0]

  # Difference 
  b_difmode=abs(b_rightmode-b_leftmode)
  
  data=pd.Series(
    {"red_mean":r,"red_median":r_median,"red_mode":r_mode,"red_std":r_std,"red_max":r_max,
     "red_min":r_min,"red_kurtosis":r_kurtosis,
     "red_r_skew":r_skew,"red_leftmode":r_leftmode,
     "red_rightmode":r_rightmode,"red_difmode":r_difmode,
     "green_mean":g,"green_median":g_median,"green_mode":g_mode,"green_std":g_std,
     "green_max":g_max,"green_min":g_min,"green_kurtosis":g_kurtosis,
     "green_skew":g_skew,"green_leftmode":g_leftmode,"green_rightmode":g_rightmode,
     "green_difmode":g_difmode,
     "blue_mean":b,"blue_median":b_median,"blue_mode":b_mode,"blue_std":b_std,
     "blue_max":b_max,"blue_min":b_min,"blue_kurtosis":b_kurtosis,
     "blue_skew":b_skew,"blue_leftmode":b_leftmode,"blue_rightmode":b_rightmode,
     "blue_difmode":b_difmode})
  
  return data

# ================================================================================
# @ Sobel Edge Detection test

IMG=cv2.imread('../Data/train-jpg/train_1.jpg')
img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print("gray",gray.shape)
# (256, 256)

# ================================================================================
def visualize_gray():
  plt.imshow(gray,cmap='gray')
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:54:28.png
# visualize_gray()

# ================================================================================
# remove noise
img=cv2.GaussianBlur(gray,(3,3),0)
def visualize_gray_blurred():
  plt.imshow(gray,cmap='gray')
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:55:11.png
# visualize_gray_blurred()

# ================================================================================
# c sobelx: x axis gradient image
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

# c sobelx: y axis gradient image
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

def visualize_sobel_gradient():
  plt.subplot(1,3,1),plt.imshow(img,cmap='gray')
  plt.title('Original'),plt.xticks([]),plt.yticks([])
  plt.subplot(1,3,2),plt.imshow(sobelx,cmap='gray')
  plt.title('Sobel X axis gradient'),plt.xticks([]),plt.yticks([])
  plt.subplot(1,3,3),plt.imshow(sobely,cmap='gray')
  plt.title('Sobel Y axis gradient'),plt.xticks([]),plt.yticks([])
  plt.show()
  # https://raw.githubusercontent.com/youngminpark2559/Multi-label-Image-Classification-/master/pics/2019_04_15_13:58:01.png
# visualize_sobel_gradient()

# ================================================================================
# Sobel feature data

# mag,ang=cv2.cartToPolar(sobelx,sobely)
# # print("mag",mag.shape)
# # print("ang",ang.shape)
# # (256, 256)
# # (256, 256)

# bins=np.int32(16*ang/(2*np.pi))
# bin_cells=bins[:10,:10],bins[10:,:10],bins[:10,10:],bins[10:,10:]
# mag_cells=mag[:10,:10],mag[10:,:10],mag[:10,10:],mag[10:,10:]

# hists=[np.bincount(b.ravel(),m.ravel(),16) for b,m in zip(bin_cells,mag_cells)]
# hist=np.hstack(hists)

# ================================================================================
# Sobel feature extraction function implementation

def sob_det(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    # Sobel Feature
    mag,ang=cv2.cartToPolar(sobelx,sobely)
    bins=np.int32(16*ang/(2*np.pi))
    bin_cells=bins[:10,:10],bins[10:,:10],bins[:10,10:],bins[10:,10:]
    mag_cells=mag[:10,:10],mag[10:,:10],mag[:10,10:],mag[10:,10:]
    hists=[np.bincount(b.ravel(),m.ravel(),16) for b,m in zip(bin_cells,mag_cells)]
    hist=np.hstack(hists)
    
    # ================================================================================
    sob=np.mean(hist) # mean of Blue channal
    sob_median=statistics.median(hist) # Media Of Blue Channal
    sob_std=np.std(hist) #Standard Deviation Of Blue Channal
    sob_max=np.max(hist) #Maximum Of Blue Channal
    sob_min=np.min(hist) #Minmum  Of Blue Channal
    sob_kurtosis=scipy.stats.kurtosis(hist) #kurtosis  Of Blue Channal
    sob_skew=scipy.stats.skew(hist) #skew  Of Blue Channal

    # ================================================================================
    sobel_feature_data=pd.Series(
      {"sob_mean":sob,"sob_median":sob_median,"sob_std":sob_std,"sob_max":sob_max,
       "sob_min":sob_min,"sob_kurtosis":sob_kurtosis,"sob_skew":sob_skew})
    
    return sobel_feature_data

# ================================================================================
# Overall Feature Extraction(RGB+Sobel)

# t=time.time()

# def feature_extranction(img_set):
#   Feature=pd.DataFrame([])
#   for i in tqdm(img_set):
#     names='../Data/train-jpg/train_'
#     extion=".jpg"
#     names+=i+extion

#     IMG=cv2.imread(names)
#     img=cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB)

#     rbg=RBG(img)
#     sob=sob_det(img)

#     samp=rbg.append(sob)
#     Feature=Feature.append(samp,ignore_index=True)
#   return (Feature)

# orders=range(0,40479)
# img_orders=[str(i) for i in orders]
# feature_extranction(img_orders)

# elapsed=time.time()-t
# print("elapsed",elapsed)

# ================================================================================
# Trasformation from Raw Pixel to Feature Values

# orders=range(0,40479)
# img_orders=[str(i) for i in orders]
# X=feature_extranction(img_orders)

# Xfinal = X.round(2)
# Xfinal.shape
# Xfinal[0:1]
# Xfinal.head()

# ================================================================================
# @ Save XGBoost feature data into CSV file

# Xfinal.to_csv("X_DF.csv",index=False, encoding='utf-8',header = True)

# ================================================================================
# @ Load saved data

Xfinal=pd.read_csv("../Data/XGB_FeatureValue.csv",encoding='utf8')
# print("Xfinal",Xfinal.shape)
# (40479, 40)

# print("Xfinal",Xfinal.columns)
# ['blue_difmode', 'blue_kurtosis', 'blue_leftmode', 'blue_max',
#  'blue_mean', 'blue_median', 'blue_min', 'blue_mode', 'blue_rightmode',
#  'blue_skew', 'blue_std', 'green_difmode', 'green_kurtosis',
#  'green_leftmode', 'green_max', 'green_mean', 'green_median',
#  'green_min', 'green_mode', 'green_rightmode', 'green_skew', 'green_std',
#  'red_difmode', 'red_kurtosis', 'red_leftmode', 'red_max', 'red_mean',
#  'red_median', 'red_min', 'red_mode', 'red_r_skew', 'red_rightmode',
#  'red_std', 'sob_kurtosis', 'sob_max', 'sob_mean', 'sob_median',
#  'sob_min', 'sob_skew', 'sob_std']

# print("Xfinal.head()",Xfinal.head())
#    blue_difmode  blue_kurtosis     ...      sob_skew     sob_std
# 0           1.0           0.58     ...          1.39   823701.35
# 1           6.0           2.62     ...          1.74  1622973.77

# ================================================================================
# @ Split Training and Testing

thre=0.6
threhold=int(np.floor(thre*len(train_label)))
# print("threhold",threhold)
# 24287

# ================================================================================
trainx=Xfinal[:threhold] 
# print("trainx",trainx.shape)
# (24287, 40)

trainy=ylabel[:threhold]
# print("trainy",trainy.shape)
# (24287, 17)

validationx=Xfinal[threhold:]
# print("validationx",validationx.shape)
# (16192, 40)

validationy=ylabel[threhold:]
# print("validationy",validationy.shape)
# (16192, 17)

# ================================================================================
xxx=trainx[:1000]
yyy=trainy[:1000]

# ================================================================================
# @ XGBoost Training model 1

xgb_clf=XGBClassifier(
  max_depth=5,learning_rate=0.1,n_estimators=100,silent=True,objective='binary:logistic',
  nthread=-1,reg_alpha=0,reg_lambda=1,base_score=0.5,missing=None)

clf_multilabel=OneVsRestClassifier(xgb_clf)

fit=clf_multilabel.fit(trainx,trainy)

# ================================================================================
# OneVsRestClassifier(pred)
# score(xxx,yyy,sample_weight=None)

# ================================================================================
# Score + predictioin model 1

clf_multilabel.score(trainx,trainy)

clf_multilabel.score(validationx,validationy)

XGBy_pred=clf_multilabel.predict_proba(validationx)

# ================================================================================
# XGBoost Training Model 2

xgb_clf=XGBClassifier(
  max_depth=10,learning_rate=0.1,n_estimators=100,silent=True,objective='binary:logistic',
  nthread=-1,reg_alpha=0,reg_lambda=1,base_score=0.5,missing=None)

clf_multilabel2=OneVsRestClassifier(xgb_clf)

fit=clf_multilabel2.fit(trainx,trainy)

# ================================================================================
# Score + Precition model 2

clf_multilabel2.score(trainx,trainy)

clf_multilabel2.score(validationx,validationy)

# ================================================================================
# XGBoost Traiing Model 3

t = time.time()

xgb_clf=XGBClassifier(
  max_depth=3,learning_rate=0.1,n_estimators=100,silent=True,objective='binary:logistic',
  nthread=-1,reg_alpha=0,reg_lambda=1,base_score=0.5,missing=None)

clf_multilabel3=OneVsRestClassifier()

fit=clf_multilabel3.fit(trainx,trainy)

elapsed=time.time()-t

print("elapsed",elapsed)

# ================================================================================
# Score + Prediction  model 3

clf_multilabel3.score(trainx,trainy)

clf_multilabel3.score(validationx,validationy)

# ================================================================================
# XGBoost Training Model 4

t=time.time()

xgb_clf=XGBClassifier(
  max_depth=7,learning_rate=0.1,n_estimators=100,silent=True,objective='binary:logistic',
  nthread=-1,reg_alpha=0,reg_lambda=1,base_score=0.5,missing=None)

clf_multilabel4=OneVsRestClassifier()

fit=clf_multilabel4.fit(trainx,trainy)

elapsed=time.time()-t
print("elapsed",elapsed)

# ================================================================================
# Score + Prediction model 4

clf_multilabel4.score(trainx,trainy)

clf_multilabel4.score(validationx,validationy)

# ================================================================================
XGBy_pred7=clf_multilabel4.predict_proba(validationx)

# ================================================================================
# Threshold adjustment Section

print(XGBy_pred[0])
print(validationy[0])
print("-------")
Threshold=0.65

newtestvali=[]
for i in XGBy_pred7:
  i=np.array(i)
  i[i>=Threshold]=1
  i[i<=Threshold]=0
  newtestvali.append(i)
    
# Examples
print(newtestvali[0])
print(validationy[0])

# ================================================================================
# Evaluation

# ================================================================================
# 1. Exact Match Ratio (MR)

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


    Exact_Match_Ratio=1-sum(msurelist)/len(msurelist)

    print("Exact Match Ratio is:{}".format(Exact_Match_Ratio))

MR_Measure(validationy,newtestvali)

# ================================================================================
# 2. Hamming Loss (HL)

def HL_measure(target,predict):
    target_flat = [item for sublist in target for item in sublist]
    predict_flat = [item for sublist in predict for item in sublist]

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
# 3. Godbole et Measure (Considering the partical correct)

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
