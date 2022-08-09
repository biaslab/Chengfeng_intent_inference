
from builtins import len, print
from email import header
from socket import if_nameindex
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib import font_manager
from geopy import distance
from math import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import OrderedDict
def get_time_interactive(num1,inter):
    a=10.0
    res=[a]
    while a<num1:
        a=a+1
        if a%inter==0:
            res.append(a)
    return res

def plot_predicted_trajectory():
    
    # file=r"/Users/jia/Documents/intent_inference/data/test_data/1543593600_1543597185_413408630_413505980.csv"
    # file="/Users/jia/Documents/intent_inference/data/test_data3//1543607100_1543610685_412373480_413405160.csv"
    # file="/Users/jia/Documents/intent_inference/data/test_data3//1543617300_1543620885_413442560_413918000.csv"
    # file="/Users/jia/Documents/intent_inference/data/test_data3//1543604100_1543607685_412350330_412413848.csv"
    file="/Users/jia/Documents/intent_inference/data/test_data3//1543610700_1543614285_412425459_414181000.csv"
    

    data=np.loadtxt(file,delimiter=",")
    tx1=data[1:,0]
    ty1=data[1:,1]
    tx2=data[1:,6]
    ty2=data[1:,7]
    
    #背景部分
    x = [122.552956, 122.514842, 122.494897, 122.494897, 122.514431, 122.552667, 122.552956]
    y = [30.970414, 30.978219, 30.991589, 31.008742, 31.013256, 31.005278, 30.970414]

    fig, ax=plt.subplots(1, 1)
    # plt.figure(figsize=(6, 5))
    # a = plt.axes(facecolor='lightskyblue')
    plt.grid(c='white')
    # plt.plot(x, y, color='#FF00FF', linestyle='-.', linewidth=0.8)
    #填色
    plt.fill(x, y, color=(237 / 255, 237 / 255, 237 / 255),zorder=-1)
    plt.fill([122.552667, 122.625, 122.625, 122.552667], [31.005278, 30.99, 31.10, 31.10],
             color=(212 / 255, 231 / 255, 245 / 255),zorder=-1)
    plt.fill([122.552667, 122.625, 122.625, 122.552667], [30.978219, 30.9551, 30.90, 30.90],
             color=(212 / 255, 231 / 255, 245 / 255),zorder=-1)
    plt.fill([122.514431, 122.552956, 122.552956, 122.514431], [31.013256, 31.005278, 31.10, 31.10],
             color=(237 / 255, 237 / 255, 237 / 255),zorder=-1)
    plt.fill([122.514431, 122.552956, 122.552956, 122.514431], [30.90, 30.90, 30.970414, 30.978219],
             color=(237 / 255, 237 / 255, 237 / 255),zorder=-1)
    plt.fill([122.552956, 122.625, 122.625, 122.552667], [31.005278, 30.991589, 30.9551, 30.970414],
             color=(237 / 255, 237 / 255, 237 / 255),zorder=-1)
    x3 = [122.494897, 122.450]
    y3 = [31.008742, 31.02]
    # plt.plot(x3, y3, color='black', linestyle='-.', linewidth=0.8)
    plt.fill([122.494897, 122.450, 122.450, 122.494897, 122.514431, 122.514431],
             [31.008742, 31.02, 31.1, 31.1, 31.1, 31.013256], color=(212 / 255, 231 / 255, 245 / 255),zorder=-1)
    plt.fill([122.494897, 122.450, 122.450, 122.494897, 122.514431, 122.514431],
             [30.9916, 31.013256, 30.90, 30.90, 30.90, 30.978], color=(212 / 255, 231 / 255, 245 / 255),zorder=-1)
    plt.fill([122.494897, 122.450, 122.450, 122.494897], [31.008742, 31.02, 31.003, 30.991589, ],
             color=(237 / 255, 237 / 255, 237 / 255),zorder=-1)
    xx=[30.9841,31.006]
    yy=[122.494897,122.494897]
  
    # plt.grid('off')
    ax.set_xlim(122.450, 122.625)
    ax.set_ylim(30.915, 31.06)
    plt.xlabel('Lon ($\circ$)')
    plt.ylabel('Lat ($\circ$)')
    index_=get_time_interactive(len(tx2)-1,10)

    plt.plot(tx1,ty1,c="red",label="Own ship")
    plt.plot(tx2,ty2,c="blue",label="Target ship")
    
    for i in index_:
        i=int(i)
        plt.plot([tx1[i],tx2[i]],[ty1[i],ty2[i]],c="gray",ls = '--',linewidth = 0.5,alpha = 0.95)
    
    #case1
    # index_list=[27,32]
    #case2
    # index_list=[119,127]
    #case3
    # index_list=[62,63,105,132]
    #case4
    index_list=[106,146]
    ii=0
    for index in index_list:
        if ii%2==0:
            plt.scatter(tx2[index],ty2[index],  marker='*',c="orange" , s=80,zorder=1,label="Avoidance intent")
            
        else:
             plt.scatter(tx2[index],ty2[index],  marker='o',c="orange" , s=80,zorder=1,label="Keep speed and course")
        ii+=1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # print(handles)
    # plt.legend(by_label.values(), by_label.keys())


    #局部放大
    #case1
    # axins =  ax.inset_axes((0.65, 0.08, 0.3, 0.3))
    #case2
    # axins =  ax.inset_axes((0.65, 0.08, 0.3, 0.3))
    #case3
    axins =  ax.inset_axes((0.65, 0.08, 0.3, 0.3))

    axins.plot(tx2[1:],ty2[1:],c="blue",label="Target ship",zorder=1)
    axins.plot(tx1[1:],ty1[1:],c="red",label="Own ship",zorder=1)
    #case2
    # xlim_left,xlim_right=122.5049, 122.5295
    # ylim_bottom,ylim_top=30.972, 30.9967
    #case3
    # xlim_left,xlim_right=122.4798, 122.499
    # ylim_bottom,ylim_top=30.982, 31.0067
    #case4
    xlim_left,xlim_right=122.53, 122.5527
    ylim_bottom,ylim_top=30.978, 31.0067

    axins.set_xlim(xlim_left,xlim_right)
    axins.set_ylim(ylim_bottom,ylim_top)
    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black",linewidth=1)
    ii=0
    for index in index_list[:2]:
        if ii==0:
            axins.scatter(tx2[index],ty2[index],  marker='*',c="orange" , s=80,zorder=1,label="Avoidance intent")
            ii=1
        else:
             axins.scatter(tx2[index],ty2[index],  marker='o',c="orange" , s=80,zorder=1)
    # axins.get_xaxis().get_major_formatter().set_scientific(False)
    axins.set_xticks([])
    axins.set_yticks([])

    # plt.legend()
    
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.savefig('./case_results/case2_trajectory.jpg',dpi=600)
    # plt.savefig('./case_results/case3_trajectory.jpg',dpi=600)
    # plt.savefig('./case_results/case4_trajectory.jpg',dpi=600)
    #数据部分   
    plt.show()
def PiRad(x):
    return x/180*pi
def DoRad(x):
    return x/pi*180
def fwAngle(Long0,Lati0,Long1,Lati1,):#本船经度，本船纬度，来船经度，来船纬度
    a=(Lati1-Lati0)*60
    g=(Long1-Long0)*cos(PiRad(Lati1))*60
    d=(a**2+g**2)**0.5
    TB=DoRad(acos(a/d))
    if Long1-Long0<0:
        TB=360-TB
    return TB,d
def dcpa_tcpa_method1(x1, y1, v1, c1, x2, y2, v2,
                      c2):  # (x1,y1)本船经纬度，(c1,v1)航向和航速 ，（y1，y2）他船经纬度，(c2,v2)航向和航速 ，return是节和分钟
    TB, d = fwAngle(x1, y1, x2, y2)
    tcpa_zero = ((v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) ** 2 + (v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1))) ** 2)

    if tcpa_zero == 0:
        TCPA = 0
        DCPA = (d ** 2 + ((v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) ** 2 + (
                    v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1))) ** 2) * TCPA ** 2 + d * (
                            sin(PiRad(TB)) * (v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) + cos(PiRad(TB)) * (
                                v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1)))) * 2 * TCPA) ** 0.5
        return  DCPA, 0 * 60

    TCPA = -(d * (sin(PiRad(TB)) * (v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) + cos(PiRad(TB)) * (
                v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1)))) / ((v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) ** 2 + (
                v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1))) ** 2))
    DCPA = (d ** 2 + ((v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) ** 2 + (
                v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1))) ** 2) * TCPA ** 2 + d * (
                        sin(PiRad(TB)) * (v2 * sin(PiRad(c2)) - v1 * sin(PiRad(c1))) + cos(PiRad(TB)) * (
                            v2 * cos(PiRad(c2)) - v1 * cos(PiRad(c1)))) * 2 * TCPA) ** 0.5
    # print("method 1 方位：%f"%(TB),DCPA,TCPA)

    # return d, TB, DCPA, TCPA * 60
    return  DCPA, TCPA * 60

def basic_DCPA():
    # file=r"/Users/jia/Documents/intent_inference/data/test_data/1543593600_1543597185_413408630_413505980.csv"
    # file="/Users/jia/Documents/intent_inference/data/test_data3//1543607100_1543610685_412373480_413405160.csv"
    # file="/Users/jia/Documents/intent_inference/data/test_data3//1543617300_1543620885_413442560_413918000.csv"
    file="/Users/jia/Documents/intent_inference/data/test_data3//1543604100_1543607685_412350330_412413848.csv"

    data=np.loadtxt(file,delimiter=",")
    # print(data)
    d_list=[]
    t_list=[]
    dis_list=[]
    for i in range(1,len(data)):
        x1=data[i,0]
        y1=data[i,1]
        v1=data[i,2]
        c1=data[i,3]

        x2=data[i,6]
        y2=data[i,7]
        v2=data[i,8]
        c2=data[i,9]

        dcpa,tcpa=dcpa_tcpa_method1(x1, y1, v1, c1, x2, y2, v2,c2)
        dis=distance.distance([y1,x1], [y2,x2]).km
        d_list.append(dcpa)
        t_list.append(tcpa)
        dis_list.append(dis)
    return d_list,t_list,dis_list

def plot_collision_risk():
    #case1
    # points=[27,32]
    #case2
    points=[116,124]
    #case3
    # points=[62,63]
    # points2=[105,132]
    #case4
    points=[106,146]

    d_list,t_list,dis=basic_DCPA()
    plt.figure(31)
    plt.subplot(311)
    x=range(len(d_list))
    plt.plot(x, d_list, 'r--',label="DCPA")
    plt.xticks([])
    plt.xlim((0,150))
    plt.legend()
    plt.ylabel('DCPA (NM)')
    foo = max(plt.yticks()[0])
    foo2 = min(plt.yticks()[0])
    plt.fill_between(points,[foo2,foo2], [foo,foo],facecolor="gold")
    # plt.fill_between(points2,[foo2,foo2], [foo,foo],facecolor="gold")
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
 
    plt.subplot(312)
    plt.plot(x, t_list, '-.',c="green",label="TCPA")
    plt.xticks([])
    plt.xlim((0,150))
    plt.ylabel('TCPA (Min)')
    foo = max(plt.yticks()[0])
    foo2 = min(plt.yticks()[0])
    plt.fill_between(points,[foo2,foo2], [foo,foo],facecolor="gold")
    # plt.fill_between(points2,[foo2,foo2], [foo,foo],facecolor="gold")

    plt.legend()
 
    plt.subplot(313)
    plt.plot(x,dis, ':',c="blue",label="Distance")
    plt.xlim((0,150))
    
    plt.xlabel('TIme step')
    plt.ylabel('Distance (Km)')
    foo = max(plt.yticks()[0])
    foo2 = min(plt.yticks()[0])
    
    plt.fill_between(points,[foo2,foo2], [foo,foo],facecolor="gold")
    # plt.fill_between(points2,[foo2,foo2], [foo,foo],facecolor="gold")
    plt.scatter(points[0],foo2,color='orange',s=60,marker='*')
    plt.scatter(points[1],foo2,color='orange',s=60,marker='o')
    # plt.scatter(points2[0],foo2,color='orange',s=60,marker='o')
    # plt.scatter(points2[1],foo2,color='orange',s=60,marker='o')
    plt.legend()
    # plt.savefig('./case_results/case1_risk.jpg',dpi=600)
    # plt.savefig('./case_results/case2_risk.jpg',dpi=600)
    # plt.savefig('./case_results/case3_risk.jpg',dpi=600)
    # plt.savefig('./case_results/case4_risk.jpg',dpi=600)
    plt.show()
    

def plot_intent_result():
    file=r"/Users/jia/Documents/intent_inference/codes/res6_smoothing/"
    # file=r"/Users/jia/Documents/intent_inference/codes/res7_smoothing/"
    name="own_res_1543593600_1543597185_413408630_413505980.csv"
    # name="own_res_1543607100_1543610685_412373480_413405160.csv"
    # name="own_res_1543617300_1543620885_413442560_413918000.csv"
   
    # name="own_res_1543604100_1543607685_412350330_412413848.csv"

    f=open(file+name,"r")
    pre=[]
    gd=[]
    for line in f:

        line=line.strip().split(",")
        pre.append(float(line[0]))
        gd.append(float(line[1]))
    print(pre)
    print(gd)
    x=range(len(gd))
    # fig, ax=plt.subplots(figsize = (7, 4))
    fig, ax=plt.subplots()

    index_list=[27]
    # index_list=[116]
    # index_list=[62]
    # index_list=[106]
    for index in index_list:
        plt.scatter(x[index],pre[index],color='orange',marker='*',s=90)

    index_list=[33]
    # index_list=[124]
    # index_list=[146]
    for index in index_list:
        plt.scatter(x[index],pre[index],color='orange',marker='o')

    # #case3
    # index_list=[104]
    # for index in index_list:
    #     plt.scatter(x[index],pre[index],color='orange',marker='*',s=90)

    # # index_list=[33]
    # # index_list=[124]
    # index_list=[132]
    # for index in index_list:
    #     plt.scatter(x[index],pre[index],color='orange',marker='o')

    plt.plot(x,gd,"deepskyblue",linestyle='--',label="Ground truth intent")
    plt.plot(x,pre,"lightcoral",label="Predicted intent")
    


    ax.set_ylim(1, 9)
    ax.set_xlim(0, 150)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.yticks(range(1,10), ["A", "B","C","D","E","F","G","H","I"])
    plt.xlabel('Time step')
    plt.ylabel('Intent')
    plt.legend()
    plt.savefig('./case_results/case1_intents.jpg',dpi=600)
    # plt.savefig('./case_results/case2_intents.jpg',dpi=600)
    # plt.savefig('./case_results/case3_intents.jpg',dpi=600)
    # plt.savefig('./case_results/case4_intents.jpg',dpi=600)
    plt.show()



if __name__ == '__main__':
    # plot_predicted_trajectory()
    # plot_collision_risk()
    plot_intent_result()