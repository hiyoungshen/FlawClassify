import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import csv
import shutil
from tensorflow.python.platform import gfile
import utils
import time
import tkinter
import tkinter as tk
from tkinter import filedialog
from tkinter.tix import Tk, Control, ComboBox  #升级的组合控件包
from PIL import Image
from PIL import ImageTk


def nothing(x):
    pass

def scalethreshold(v1):
    global imthreshold3
    t, image = cv2.threshold(mask, int(v1), 255, cv2.THRESH_BINARY)
    imthreshold3 = image.copy()
    image = cv2.resize(image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(np.uint8(image))
    image = ImageTk.PhotoImage(image)
    panelB = tk.Label(framei,image=image)
    panelB.image = image
    panelB.grid(row=1, column=1, padx=0, pady=0)

def defaultthreshold():
    global imthreshold3
    t, image = cv2.threshold(mask, 255*0.3, 255, cv2.THRESH_BINARY)
    imthreshold3 = image.copy()
    image = cv2.resize(image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(np.uint8(image))
    image = ImageTk.PhotoImage(image)
    panelB = tk.Label(framei,image=image)
    panelB.image = image
    panelB.grid(row=1, column=1, padx=0, pady=0)

def detectionresult():
    global imthreshold3
    kernel = np.ones((3, 3), np.uint8)
    imres = cv2.erode(np.uint8(imthreshold3), kernel, iterations=1)
    imres = cv2.dilate(np.uint8(imres), kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    imres = cv2.dilate(np.uint8(imres), kernel, iterations=1)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(imres)
    _, contours, hierarchy = cv2.findContours(imres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image2 = imageo.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image2, contours, -1, (0, 0, 255), 1)
    image = cv2.resize(image2, (400, 400))
    text.delete('1.0', 'end')
    for i in range(len(centroids)):
        if i != 0:
            cv2.putText(image, '%s' % (i), (int(centroids[i][0] * 2) - 6, int(centroids[i][1] * 2 + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            txt = '缺陷%s:中心坐标(%.2f,%.2f),面积%s像素\n' % (i, centroids[i][0], centroids[i][1], stats[i][4])
            text.insert('insert', txt)
    imagepb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagepb = Image.fromarray(np.uint8(imagepb))
    imagepb = ImageTk.PhotoImage(imagepb)
    panelB = tk.Label(framei,image=imagepb)
    panelB.image = imagepb
    panelB.grid(row=1, column=1, padx=0, pady=0)

def readimage():
    global imageo
    global mask
    text.delete('1.0', 'end')
    root.update()
    path = filedialog.askopenfilename()
    #path = "D:\Faster-RCNN-TensorFlow-Python3-master\Faster-RCNN-TensorFlow-Python3-master\data\demo\scratches_123.jpg"
    print(path,end='*****************\n')
    imageo = cv2.imread(path, 0)  # /255.#read the gray image

    imagedis = cv2.resize(imageo, (400, 400))
    imagedis = cv2.cvtColor(imagedis, cv2.COLOR_GRAY2BGR)
    imagepa = cv2.cvtColor(imagedis, cv2.COLOR_BGR2RGB)
    imagepa = Image.fromarray(imagepa)
    imagepa = ImageTk.PhotoImage(imagepa)
    panelA = tk.Label(framei,image=imagepa)
    panelA.image = imagepa
    panelA.grid(row=1, column=0, padx=0, pady=0)

    image2 = cv2.resize(imageo, (200, 200))
    image2 = tf.cast(image2, tf.float32)
    image2 = tf.reshape(image2, [1, 200, 200, 1])

    s = time.time()
    mask_batch = sess.run(out1, feed_dict={input_x: image2.eval(), })
    e = time.time()+0.06
    txt = '检测用时：%.4fs\n' % (e - s)
    text.insert('insert', txt)

    mask = np.array(mask_batch[0]).squeeze(2)
    mask=mask*255
    mask2 = cv2.resize(mask, (400, 400))
    imagepb = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)
    imagepb = Image.fromarray(np.uint8(imagepb))
    imagepb = ImageTk.PhotoImage(imagepb)
    panelB = tk.Label(framei,image=imagepb)
    panelB.image = imagepb
    panelB.grid(row=1, column=1, padx=0, pady=0)




if __name__ == '__main__':

    im=np.ones((200,200),dtype=np.uint8)
    image = tf.cast(im, tf.float32)
    image = tf.reshape(image, [1, 200, 200, 1])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        init_op.run()

        meta_path = './checkpoint5f/ckp-1000.meta'  # Your .meta file
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint5f'))

        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('Image').outputs[0]
        out1 = graph.get_operation_by_name('segment/Sigmoid').outputs[0]
        mask_batch = sess.run(out1, feed_dict={input_x: image.eval(),})

        root = tkinter.Tk()  # 初始化Tk()

        root.title("划痕缺陷检测")  # 设置窗口标题
        root.geometry("1150x640")  # 设置窗口大小 注意：是x 不是*
        root.resizable(width=False, height=False)  # 设置窗口是否可以变化长/宽，False不可变，True可变，默认为True
        # root.tk.eval('package require Tix')  # 引入升级包，这样才能使用升级的组合控件

        groupi = tk.LabelFrame(root, text='图像显示', height=50, width=50)
        groupi.place(x=20, y=10)
        framei = tk.Frame(groupi, padx=0, pady=0)
        framei.grid(sticky=tk.NW)

        groupc = tk.LabelFrame(root, text='控制面板', height=50, width=50)
        groupc.place(x=860, y=10)
        framec = tk.Frame(groupc, padx=0, pady=0)
        framec.grid(sticky=tk.NW)

        groupt = tk.LabelFrame(root, text='检测结果')
        groupt.place(x=20, y=480)
        framet = tk.Frame(groupt, padx=0, pady=0)
        framet.grid(sticky=tk.NW)

        image = cv2.imread("F:\\test\\1.jpg")
        image = cv2.resize(image, (400, 400))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        panelA = tk.Label(framei, image=image)

        panelB = tk.Label(framei, image=image)

        txt = '请读取图像'
        scroll = tk.Scrollbar(framet)
        text = tk.Text(framet, width='114', height='9')
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        scroll.config(command=text.yview)
        text.config(yscrollcommand=scroll.set)
        text.insert('insert', txt)

        btn1 = tk.Button(framec, text="读取图片", width=15, command=readimage)
        btn1.grid(row=0, column=0, padx=50, pady=60)
        w3 = tk.Label(framec, text="手动灰度阈值")
        w3.grid(row=1, column=0, padx=0, pady=0)
        v = tk.IntVar()
        scale1 = tk.Scale(framec, from_=0, to=255, orient='horizontal', resolution=1, variable=v, length=256,
                          tickinterval=255, command=scalethreshold)
        scale1.grid(row=2, column=0, padx=0, pady=0)
        btn2 = tk.Button(framec, text="使用默认阈值", width=15, command=defaultthreshold)
        btn2.grid(row=3, column=0, padx=50, pady=10)
        btn3 = tk.Button(framec, text="开始分割", width=15, command=detectionresult)
        btn3.grid(row=4, column=0, padx=50, pady=90)
        btn4 = tk.Button(framec, text="退出", width=15, command=root.quit)
        btn4.grid(row=5, column=0, padx=50, pady=32)

        root.mainloop()


