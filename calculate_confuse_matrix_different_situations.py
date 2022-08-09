from getopt import gnu_getopt
from threading import main_thread
from tkinter.tix import MAIN
from pip import main
from builtins import print
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import confusion_matrix 
import itertools  
import string

def trans_label_acc(num):
    if num in [1.0,2.0,3.0]:
        res=1.0
    if num in [4.0,5.0,6.0]:
        res=2.0
    if num in [7.0,8.0,9.0]:
        res=3.0
    return res

def trans_label_course(num):
    if num in [1.0,4.0,7.0]:
        res=1.0
    if num in [2.0,5.0,8.0]:
        res=2.0
    if num in [3.0,6.0,9.0]:
        res=3.0
    return res

def get_encounter_label():
    path=r"/Users/jia/Documents/intent_inference/data/reshape_savedata3_raw_label"
    files=os.listdir(path)
    s=[]
    for file in files:
        if not os.path.isdir(file):
            f=path+"/"+file
            s.append(f)
            
    print(s)
    label_dic={}

    for i in range (len(s)):
        xx=s[i]
        # print(xx)
        file=xx.split("/")[-1].split(".")[0]

        # print(file)
        fr=open(xx,"r")
        status=-1
        for line in fr:
            line=line.strip().split(',')[0]
            line=float(line)
    
            if line==1:
                label_dic[file]="head-on"
                status=1
            if line==2:
                label_dic[file]="crossing"
                status=1
            if line==3:
                label_dic[file]="overtaking"
                status=1
        if status==-1:
            label_dic[file]="No encounter"
    print(label_dic)
    with open("./encounter_label.pickle", "wb") as fp:   #Pickling
        pickle.dump(label_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)     
def confuse_matrix_for_situations(encounter_mode):
    path=r"/Users/jia/Documents/intent_inference/codes/res7_smoothing/"
    files=os.listdir(path)
    s=[]
    for file in files:
        if not os.path.isdir(file):
            f=path+"/"+file
            s.append(f)
    # print(s)
    pre=[]
    gd=[]

    pre_a=[]
    pre_c=[]
    gd_a=[]
    gd_c=[]

    with open("./encounter_label.pickle", "rb") as fp:   #Pickling
        label_dic = pickle.load(fp)   
    cunt=0
    for i in range (len(s)):
        xx=s[i]
        # print(xx)
        file=xx.split("/")[-1].split(".")[0]
        file=StrA = "_".join(file.split("_")[2:])
        try:
            if label_dic[file]==encounter_mode:
                # print("牛逼哄哄")
                fr=open(xx,"r")
                for line in fr:
                    line=line.strip().split(',')
                    pre.append(float(line[0]))
                    gd.append(float(line[1]))
                    pre_a.append(trans_label_acc(float(line[0])))
                    gd_a.append(trans_label_acc(float(line[1])))
                    pre_c.append(trans_label_course(float(line[0])))
                    gd_c.append(trans_label_course(float(line[1])))


        except:
            cunt+=1
    print(cunt)
    cout=0
    for i in range(len(pre)):
        if float(pre[i])==float(gd[i]):
            cout+=1
    acc=cout/len(pre)
    print(acc)
    cm = confusion_matrix(gd, pre)
    cm_a = confusion_matrix(gd_a, pre_a)
    cm_c = confusion_matrix(gd_c, pre_c)
    return [encounter_mode,cm,cm_a,cm_c]

def plot_confuse_matrix(cm):
    encounter_mode,cm=cm[:2]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    if encounter_mode=="head-on":
        cmap=plt.cm.Oranges
    if encounter_mode=="overtaking":
        cmap=plt.cm.Greens
    if encounter_mode=="crossing":
        cmap=plt.cm.Purples
    if encounter_mode=="No encounter":
        cmap=plt.cm.Oranges
    
    classes=list(string.ascii_uppercase)[:9]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./confuse_matrix_res/%s.jpg'%encounter_mode,dpi=600)
    
    plt.show()

def sub_confuse_matrix_for_situations(cm,sub_label):
    if sub_label=="acc":
        encounter_mode,cm=cm[0],cm[2]
        classes=["Acc","Keep","Dec"]
    if sub_label=="course":
        encounter_mode,cm=cm[0],cm[3]
        classes=["Right","Straight","Left"]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    if encounter_mode=="head-on":
        cmap=plt.cm.Oranges
    if encounter_mode=="overtaking":
        cmap=plt.cm.Greens
    if encounter_mode=="crossing":
        cmap=plt.cm.Purples
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.savefig('./confuse_matrix_res/sub_%s_%s.jpg'%(encounter_mode,sub_label),dpi=600)
    
    plt.show()
    


# def sub_confuse_plot():

if __name__ == '__main__':
    get_encounter_label()
    cm=confuse_matrix_for_situations("overtaking")
    plot_confuse_matrix(cm)
    # sub_confuse_matrix_for_situations(cm,"course")

