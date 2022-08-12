import os
from create_key import get_keys
import pandas as pd
from collections import Counter


def merge_data():
    key_list=get_keys()
    path=r"/Users/jia/Documents/intent_inference/compare_model/elstm_res/"
    for name in key_list:
        all_files=[]
        for i in range(1,6):
            name_file=path+str(i)+"/"+"own_res_"+name+".csv"
            all_files.append(name_file)
        df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
        df_merged   = pd.concat(df_from_each_file, ignore_index=True,axis=1)
        df_merged.to_csv(path+"all"+"/own_res_"+name+".csv",index=None,header=None)


def emsemble():
    path=r"/Users/jia/Documents/intent_inference/compare_model/elstm_res/all/"
    path2=r"/Users/jia/Documents/intent_inference/compare_model/elstm_res/all_merge/"
    files=os.listdir(path)
    index_=[0,2,4,6,8]
    s=[]
    for file in files:
        if not os.path.isdir(file):
            f=path+"/"+file
            s.append(f)
    for name in s:
        f=open(name,"r")
        n=name.split("/")[-1]
        fw=open(path2+n,"w")
        for line in f:
            line=line.strip().split(",")
            list_=[line[i] for i in index_]
            gd=line[1]
            
            # if list_[0]!="5":
            #     print(list_)
            #     print(gd)
            gd=line[9]
            result = Counter(list_)
            # print(set(result.keys()))
            if result.keys()=={"5"}:
                pre="5"
            else:
                del result['5']
                # print(result)
                pre = max(result, key=result.get)

            
            # print(pre,gd)
            fw.write(str(pre)+","+str(gd)+"\n")
            # result=pd.value_counts(list_)
            # print(result.keys())

        # break


    # print(s)
if __name__ == "__main__":
    emsemble()