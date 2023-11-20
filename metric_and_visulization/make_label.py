from turtle import heading
import numpy as np
import pandas as pd
import os
def translabel(num1,num2):
    #Num1 is turing,num2 is acceleration
    if num1 in [2,4] and num2>0:
        l=1
    if num1==0 and num2>0:
        l=2
    if num1 in [1,3] and num2>0:
        l=3
    if num1 in [2,4] and  num2==0:
        l=4
    if num1==0 and num2==0:
        l=5
    if num1 in [1,3] and num2==0:
        l=6
    if num1 in [2,4] and num2<0:
        l=7
    if num1==0 and num2<0:
        l=8
    if num1 in [1,3] and num2<0:
        l=9
        # print(num1)
    return l
path=r"/Users/jia/Documents/intent_inference/data/merge_data/"
files=os.listdir(path)
s=[]
for file in files:
    if not os.path.isdir(file):
        f=path+"/"+file
        s.append(f)
print(s)
save_path=r"/Users/jia/Documents/intent_inference/data/test_data/"
for i in range (len(s)):
    xx=s[i]
    # print(xx)
    file_name=xx.split("/")[-1]
    # print(file_name)
    x=np.loadtxt(xx,delimiter=",")
    # print(x)
    res_a=[]
    res_b=[]
    for i in range(len(x)):
        line=x[i]
        la=translabel(line[4],line[5])
        lb=translabel(line[10],line[11])
        res_a.append(la)
        res_b.append(lb)
    res=np.array([res_a,res_b])
    res=res.transpose(1,0)
    data=np.hstack((x,res))
    

    pd.DataFrame(data).to_csv(save_path+file_name,index=False)
        # # np.savetxt("new.csv", data, delimiter=',')

# res_a=[]
# res_b=[]
# for i in range(len(x)):
#     line=x[i]
#     la=translabel(line[4],line[5])
#     lb=translabel(line[10],line[11])
#     res_a.append(la)

#     res_b.append(lb)
# res=np.array([res_a,res_b])
# res=res.transpose(1,0)
# data=np.hstack((x,res))
# pd.DataFrame(data).to_csv('sample.csv',index=False)
# # np.savetxt("new.csv", data, delimiter=',')
