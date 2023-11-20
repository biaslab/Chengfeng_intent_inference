from get_data import get_data,gettest_data
import numpy as np
from sklearn.svm import SVC
import joblib
from create_key import get_keys
from sklearn.ensemble import RandomForestClassifier

def train_rf():
    data,label=get_data()
    data=np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
    print(data.shape)
    print(label.shape)

    rfc = RandomForestClassifier(n_estimators=100)                      #实例化
    rfc = rfc.fit(data,label)                      #用训练集数据训练模型
    # clf.fit(data, label) 
    # joblib.dump(clf, "/Users/jia/Documents/intent_inference/compare_model/model_svm/svm_model.m")
    joblib.dump(rfc, "/Users/jia/Documents/intent_inference/compare_model/model_rf/rf_model2.m")

def test_rf():
    clf = joblib.load("/Users/jia/Documents/intent_inference/compare_model/model_rf/rf_model3.m")
    path=r"/Users/jia/Documents/intent_inference/compare_model/rf_res3/"
    dic=gettest_data()
    key_list=get_keys()
    for key in dic.keys():
        if key not in key_list:
            continue
        data=dic[key]["data"]
        data=np.array(data)
        data=np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
        label=dic[key]["label"]
        fw=open(path+"own_res_%s.csv"%(key),"w")
        for li in range(len(data)): 
            dd=data[li]
            ll=label[li]
            pre=clf.predict([dd])[0]
            # print(pre,ll)
            fw.write(str(pre)+","+str(ll)+"\n")

            
        # break
            # data=np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
        
        # fw=open(path+"own_res_%s.csv"%(key),"w")

    # data=np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
    # for i in range(len(data)):
    #     dd=[data[i]]
    #     ll=[label[i]]
    #     pre=clf.predict(dd)
    #     print(pre,ll)

    # print(clf.predict(data))
    # print(clf.score(data,label))

if __name__ == "__main__":
    # train_rf() 
    test_rf()