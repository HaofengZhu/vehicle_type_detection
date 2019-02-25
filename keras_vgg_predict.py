from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

#加载模型和label对应的编号
model = load_model('my_model.h5')

d=pd.read_csv('labels_match.csv')
labels_dict={}
for i in d.index:
    labels_dict[d['label'][i]]=d['no'][i]

x_test=[]
y_test=[]
## predict test data
predictions = model.predict(x_test)

# get labels
predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in labels_dict.items()}
y_pre = [rev_y[k] for k in predictions]

print(y_test)
print(y_pre)
loss_and_metrics = model.evaluate(x_test, y_test)
print(classification_report(y_test,y_pre))
