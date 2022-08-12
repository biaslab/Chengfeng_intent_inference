from email import header
import numpy as np
import os 

def one_hot(y):
    res=[0.0]*9
    res[int(y)-1]=1.0
    return res

def get_data():
    path=r"/Users/jia/Documents/intent_inference/data/train_data"
    # path=r"/Users/jia/Documents/intent_inference/data/test_data3"
    files=os.listdir(path)
    s=[]
    for file in files:
        if not os.path.isdir(file):
            f=path+"/"+file
            s.append(f)
    train_data=[]
    train_label=[]
    look_back=5

    for name in s:
        csv_array = np.loadtxt(name,delimiter=',',skiprows=1)
        # print(csv_array.shape)
        data=csv_array[:,[12,13,2,3,14,15,8,9]]
        label=csv_array[:,17]
        # print(len(data))
        for i in range(len(data)):
            # print(i)
            if i>=look_back:
                train_data.append(data[i-look_back:i,:])
                train_label.append(one_hot(label[i-1]))
                # train_label.append(label[i-1])
        # break
    return np.array(train_data),np.array(train_label)

def gettest_data():
    # path=r"/Users/jia/Documents/intent_inference/data/train_data"

    path=r"/Users/jia/Documents/intent_inference/data/test_data3"
    files=os.listdir(path)
    s=[]
    for file in files:
        if not os.path.isdir(file):
            f=path+"/"+file
            s.append(f)
    
    look_back=5

    dic={}

    for name in s:
        train_data=[]
        train_label=[]
        key_name=name.strip().split("/")[-1].split(".")[0]
        csv_array = np.loadtxt(name,delimiter=',',skiprows=1)
        # print(csv_array.shape)
        data=csv_array[:,[12,13,2,3,14,15,8,9]]
        label=csv_array[:,17]
        # print(len(data))
        for i in range(len(data)):
            # print(i)
            if i>=look_back:
                train_data.append(data[i-look_back:i,:])
                train_label.append(one_hot(label[i-1])) #for lstm
                # train_label.append(label[i-1]) #for svm
        dic[key_name]={}
        dic[key_name]["data"]=train_data
        dic[key_name]["label"]=train_label
        # break
    return dic
if __name__ == "__main__":
    # a=one_hot(7)
    # print(a)
    # data,label=get_data()
    dic=gettest_data()
    print(dic["1543928700_1543932285_412153000_413361570"])
    # print(data.shape)
    # print(label[1])