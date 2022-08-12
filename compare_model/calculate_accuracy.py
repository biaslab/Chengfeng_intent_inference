import numpy as np
import os
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import itertools  
import string
from sklearn.metrics import classification_report
# path=r"/Users/jia/Documents/intent_inference/codes/res7_smoothing/"
# path=r"/Users/jia/Documents/intent_inference/compare_model/lstm_res/"
# path=r"/Users/jia/Documents/intent_inference/compare_model/elstm_res/all_merge/"
path=r"/Users/jia/Documents/intent_inference/compare_model/svm_res/"
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
    # print(xx)
    fr=open(xx,"r")
    for line in fr:
        line=line.strip().split(',')
        # print(line)
        pre.append(float(line[0]))
        gd.append(float(line[1]))
# print(pre)
# print(gd)
classes=list(string.ascii_uppercase)[:9]

t = classification_report(gd, pre, target_names=classes)
print(t)

# Factor graph
"""
              precision    recall  f1-score   support

           A       0.22      0.19      0.20      8051
           B       0.18      0.20      0.19      9308
           C       0.16      0.17      0.17      3953
           D       0.20      0.35      0.25     60904
           E       0.89      0.77      0.83    682832
           F       0.14      0.28      0.19     46969
           G       0.34      0.41      0.37      7707
           H       0.38      0.30      0.34     12915
           I       0.09      0.13      0.11      4771

    accuracy                           0.68    837410
   macro avg       0.29      0.31      0.29    837410
weighted avg       0.77      0.68      0.72    837410
"""

#LSTM
"""
              precision    recall  f1-score   support

           A       0.00      0.00      0.00      8043
           B       0.00      0.00      0.00      9170
           C       0.00      0.00      0.00      3929
           D       0.07      0.09      0.08     59918
           E       0.82      0.82      0.82    673425
           F       0.00      0.00      0.00     45886
           G       0.00      0.00      0.00      7683
           H       0.01      0.08      0.03     12769
           I       0.00      0.00      0.00      4755

    accuracy                           0.68    825578
   macro avg       0.10      0.11      0.10    825578
weighted avg       0.68      0.68      0.68    825578
"""
#elstm
"""
             precision    recall  f1-score   support

           A       0.00      0.00      0.00      8038
           B       0.00      0.00      0.00      9104
           C       0.00      0.00      0.00      3909
           D       0.21      0.12      0.15     59553
           E       0.83      0.95      0.89    670383
           F       0.00      0.00      0.00     45520
           G       0.02      0.01      0.01      7679
           H       0.01      0.00      0.01     12697
           I       0.00      0.00      0.00      4751

    accuracy                           0.79    821634
   macro avg       0.12      0.12      0.12    821634
weighted avg       0.69      0.79      0.73    821634

"""

#svm
"""
              precision    recall  f1-score   support

           A       0.09      0.00      0.00      8043
           B       0.00      0.00      0.00      9170
           C       0.00      0.00      0.00      3929
           D       0.36      0.14      0.20     59918
           E       0.83      0.99      0.90    673425
           F       0.00      0.00      0.00     45886
           G       0.00      0.00      0.00      7683
           H       0.01      0.00      0.00     12769
           I       0.00      0.00      0.00      4755

    accuracy                           0.82    825578
   macro avg       0.14      0.13      0.12    825578
weighted avg       0.70      0.82      0.75    825578
"""