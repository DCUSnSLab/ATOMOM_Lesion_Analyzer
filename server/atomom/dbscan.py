
# -*- coding: utf-8 -*-
import os, sys
from tqdm import tqdm
from PIL import Image
import cv2


import numpy as np
import django

import matplotlib.pyplot as plt
import matplotlib

import matplotlib.pyplot as plt
import matplotlib

import cv2
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from kneed import KneeLocator

curPath=os.getcwd()
path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath("__file__"))))))
print(path)
# print(os.listdir(path))
path=os.path.join(path,'atoOCR')
print(path)
sys.path.append(path)
import demo_modifed_for_one_image_processing as ocr

os.chdir(path)
craftModel, model, opt = ocr.setModel()
os.chdir(curPath)
import winsound as ws


def beepsound(freq=5000, ms=5000):
    ws.Beep(freq, ms)  # winsound.Beep(frequency, duration)


def getOcrResult(imgPath):
    img, points = ocr.craftOperation(imgPath, craftModel, dirPath=opt.image_folder)
    texts = ocr.demo(opt, model)
    return img, points, texts
def customDBscan_vis_True(img,data,rows,cols,eps,minPts,name):
    db = DBSCAN(eps=eps, min_samples=minPts).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]



    # fig = plt.figure(figsize=(18, 8))
    fig = plt.figure(figsize=(18, 3))

    ax2 = plt.subplot(131)
    plt.title('src', fontsize=20)
#     ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='auto')
#     ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='auto')
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(132)
    plt.plot(data[:, 0], data[:, 1], 'ko')
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.title('bboxes', fontsize=20)

    plt.subplot(133)


    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='none', markersize=7)

        # xy = data[class_member_mask & ~core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #           markeredgecolor='none',markersize=3)
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.title('DBSCAN: %d clusters found' % n_clusters, fontsize=20)
    fig.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])



    path='./plt/'

    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path+name,dpi=300)
    plt.close(fig)
def customDBscan_vis_False(img,data,rows,cols,eps,minPts):
    pass

def customDBscan(img,data,rows,cols,eps,minPts,vis,name):
    if(vis==True):
        customDBscan_vis_True(img=img,data=np.array(data), rows=rows, cols=cols, eps=eps, minPts=minPts,name=name)
    else:
        customDBscan_vis_False(img=img,data=np.array(data), rows=rows, cols=cols, eps=eps, minPts=minPts)


def run(img,bboxs,eps,minPts,vis=False,name='default'):
    rows,cols,_=img.shape
    zeros = np.zeros((rows, cols), dtype=np.uint8)
    for i in bboxs:
        y1,x1,y2,x2=i
        cv2.rectangle(zeros, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cum = []
    x, y = np.where(zeros == 255)
    for i, data in enumerate(x):
        zeros[x[i],y[i]]=255
        cum.append([y[i],x[i]])
    customDBscan(img=img,data=np.array(cum), rows=rows, cols=cols, eps=eps, minPts=minPts,vis=vis,name=name)


if __name__ == '__main__':
    path = './plt/'
    print(os.path.join(path,'ddd'))
    imgName="test11.jpg"
    imgPath='C:/Users/dgdgk/Desktop/atomom_product_server/demo_image/'+imgName
    img, points, texts = getOcrResult(imgPath)
    # print(texts)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    run(img=img, bboxs=points, eps=45, minPts=2, vis=True, name=imgName)
#     #     base_path='C:/Users/dgdgk/Desktop/atomom_product_server/demo_image'
#     base_path = 'C:/Users/dgdgk/Desktop/atomom_product_server/cosmetic_demo_image'
#     dirlist = []
#     for filename in os.listdir(base_path):
#         if os.path.isfile(os.path.join(base_path, filename)) == True:
#             dirlist.append(os.path.join(base_path, filename))
#             print(filename)
#
# #     print(beepsound())
#         cnt = 0
#         for i in tqdm(range(len(dirlist))):
#             imgPath = dirlist[i]
#             img, points, texts = getOcrResult(imgPath)
#             rows, cols, _ = img.shape
#             eps = 25
#             minPts = 2
#
#             filename = os.path.basename(imgPath)
#             filename, extension = os.path.splitext(filename)
#             filename = "filename_" + filename + "_eps=" + str(eps) + "_minPts" + str(minPts) + extension
#             #         print(filename)
#             cv2.imwrite('./plt/' + filename, img)
#         #         cv2.namedWindow("img",cv2.WINDOW_NORMAL)
#         #         cv2.imshow("img",img)
#         #         cv2.waitKey(0)
#         #         run(img=img,bboxs=points,eps=eps,minPts=minPts,vis=True,name=filename)
#         import ctypes  # An included library with Python install.
#
#         ctypes.windll.user32.MessageBoxW(0, "Your text", "Your title", 1)



