
from builtins import float, print
import numpy as np
import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def judge_course(lg,lp):
    r=0
    if lg in [1.0,4.0,7.0] and lp in [1.0,4.0,7.0]:
       r=1
    if lg in [2.0,8.0] and lp in [2.0,8.0]:
        r=1
    if lg in [3.0,6.0,9.0] and lp in [3.0,6.0,9.0]:
        r=1
    return r
def judge_acc(lg,lp):
    r=0
    if lg in [1.0,2.0,3.0] and lp in [1.0,2.0,3.0]:
       r=1
    if lg in [4.0,6.0] and lp in [4.0,6.0]:
       r=1
    if lg in [7.0,8.0,9.0] and lp in [7.0,8.0,9.0]:
        r=1
    return r
def first_detect():
    #找到第一次避让的时间（预测/真实）
    #输出为res是【提前量，第一次真实值转向，意图标签】，res_d是以会遇标签为key，res为value的字典
    with open("./encounter_label.pickle", "rb") as fp:   #Pickling
        label_dic = pickle.load(fp)   
    path=r"/Users/jia/Documents/intent_inference/codes/res/"
    files=os.listdir(path)
    res=[]
    res_d={}
    s=[]
    for file in files:
        if not os.path.isdir(file):
            f=path+"/"+file
            s.append(f)
    for i in range (len(s)):
        
        xx=s[i]
        file=xx.split("/")[-1].split(".")[0]
        file=StrA = "_".join(file.split("_")[2:])
        try:
            encounter_label=label_dic[file]
        except:
            encounter_label=-1
        fr=open(xx,"r")
        cnt=0
        cnt2=0
        first_gd=8000
        for line in fr:
            cnt+=1
            line=line.strip().split(',')
            # print(line)
            gd=float(line[1])
            if gd!=5.0:
                first_gd=cnt
                first_label=gd
                break
        if first_gd!=8000:
            for line in fr:
                cnt2+=1
                if cnt2>=first_gd-5 and cnt2<=first_gd+5 and encounter_label!=-1:
                # if cnt2>=first_gd-10 and cnt2<=first_gd+10:
                    line=line.strip().split(',')
                    pre=float(line[1])
                    r=judge_course(first_label,pre)
                    r2=judge_acc(first_label,pre)
                    # if pre==first_label and first_gd>5:
                    # if r==1:
                    if r2==1:
                        first_pre=cnt2
                        res.append([cnt2-first_gd,first_gd,first_label])
                        if encounter_label not in res_d.keys():
                            print(res_d.keys())
                            res_d[encounter_label]=[[first_pre-first_gd,first_gd,first_label]]
                        else:
                            res_d[encounter_label].append([first_pre-first_gd,first_gd,first_label])
                            break

    return res,res_d

def try_sta(res):
    #计算平均提前量
    res=np.array(res)
    m=np.mean(res[:,0])
    print(m)
                        
def plot_ahead_time(res_d):
    if isinstance(res_d,dict):
        value=np.array(res_d["overtaking"])
    if isinstance(res_d,list):
        value=np.array(res_d)
    # print(value)
    pal = ["red", "blue"]
    max_time=np.max(value[:,1])+5
    plt.plot([0,max_time],[0,max_time],c="black",linestyle=":")
    x=[0,max_time]
    y1=[0,max_time]
    y2=[max_time,0]
    plt.stackplot(x,y1,y2, labels=['Inffered in advance','Inffered delayed'],colors=pal, alpha=0.2 )
    
    # plt.scatter(value[:,1],value[:,1]+value[:,0],marker="*",s=30,c="green")
    # plt.scatter(value[:,1],value[:,1]+value[:,0],marker="+",s=30,c="green")
    plt.scatter(value[:,1],value[:,1]+value[:,0],marker=">",s=30,c="green")
    plt.xlim(0, max_time)
    plt.ylim(0, max_time)
    plt.xlabel('The time step of avoid intent in groud truth')
    plt.ylabel('The time step of avoid intent inferred')

    plt.legend(loc='upper left')
    # plt.savefig('./confuse_matrix_res/delay_all.jpg',dpi=600)
    # plt.savefig('./confuse_matrix_res/delay_course.jpg',dpi=600)
    plt.savefig('./confuse_matrix_res/delay_acc.jpg',dpi=600)
    plt.show()

    return value
    
def for_test():
    with open("./encounter_label.pickle", "rb") as fp:   #Pickling
        label_dic = pickle.load(fp)
    print(label_dic)
    v=label_dic.values()
      
    result = Counter(v)
    print(result) 




if __name__ == '__main__':
    res,res_d=first_detect()
    # l=plot_ahead_time(res_d)
    l=plot_ahead_time(res)
    # print(np.shape(l))
    # print(len(res))
    try_sta(res)
    # for_test()