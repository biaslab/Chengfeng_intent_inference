import numpy as np
import os
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import itertools  
import string
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

for i in range (len(s)):
    xx=s[i]
    print(xx)
    fr=open(xx,"r")
    for line in fr:
        line=line.strip().split(',')
        print(line)
        pre.append(float(line[0]))
        gd.append(float(line[1]))
# print(pre)
# print(gd)
cout=0
for i in range(len(pre)):
    if float(pre[i])==gd[i]:
        cout+=1
acc=cout/len(pre)
print(acc)
cm = confusion_matrix(gd, pre)

# cm[8,8]=1
# cm[8,6]=3
# cm[7,8]=7
# cm[7,7]=1
# cm[2,2]=8
# cm[2,7]=6
# cm[2,4]=1
# cm[2,3]=3
# cm[5,5]=6
# cm[5,8]=7
# cm[5,2]=4
# cm[5,2]=9
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm)
cmap=plt.cm.Blues
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
plt.savefig('./confuse_matrix_res/all.jpg',dpi=600)
plt.show()

#现在在228个会遇数据中，避让意图检测准确率是79%。具体的混淆矩阵情况是图中这样，表现好的方面是只要是stand-on的意图，都能较准确的推测（红色行）。表现不好的方面是减速右转，加速左转这两类避让意图效果特别差。为了找到原因，我用gif的形式画出避让轨迹，看是哪一步出问题。发现现在的意图推断算法其实比较准了，反倒是离线打的标签不准。

#实验结果：训练集个数128对会遇轨迹，测试集2418对。准确率现在为76%，这个里面只考虑转向避让的话准确率能超过80%。之前的错误在于（1）标签标记有误、（2）参考的文献里船舶运动模型有问题。这两点改过来后实验准确率变高了。
#从有限的例子看，转弯意图平均可以提前15-30秒推断出来，但是加减速没有提前，甚至还有很多延后的例子。这里我不准备继续提高准确率了。（1）用表格和混淆矩阵展现总体推断的统计结果，（2）用时间分布图表明ground truth第一次出现避让动作和推断结果时间提前量的对比。（3）找到明显避让的例子，用概率变化图展示避让意图是如何随着会遇态势的改变而累积的。
#这三天我会把实验的所有可视化结果都做完