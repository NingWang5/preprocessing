import pandas as pd
import numpy as np
import os


################## Configuration ####################
percent = 60
times = 10
folder_path = '/root/wangning/data'
label_path = '/root/wangning/label.csv'
output_path = '/root/wangning/data.csv'
#####################################################

label = pd.read_csv(label_path)
files = os.listdir(folder_path)
ls = []

for file in files:

    ls.append(pd.DataFrame(columns=['file', 'x', 'y', 'z', 'a']))
    data = ls[-1]

    data = pd.read_table(os.path.join(folder_path, file), header=None, skiprows=1)
    data = data.applymap(lambda x: float(str(x).replace(',', '.')))
    data.columns = ['file', 'x', 'y', 'z', 'a', 'label']
    data['file'] = file
    
    temp = data[['x', 'y', 'z']].applymap(lambda x: abs(x))
    x_mean = temp['x'].mean() * percent / 100
    y_mean = temp['y'].mean() * percent / 100
    z_mean = temp['z'].mean() * percent / 100

    drop_list = []
    for idx, row in data.iterrows():
        if abs(row['x']) > x_mean:
            break
        drop_list.append(idx)
    data = data.drop(index=drop_list)
    data = data.reset_index(drop=True)

    num = 0
    for idx, row in data.iterrows():
        if abs(row['x']) < x_mean:
            num += 1
        else:
            num = 0
        if num > times:
            num = idx - times
            break
    length = len(data)
    num = length if num == 0 else num
    data = data.drop(index=list(range(num, length)))
    data = data.reset_index(drop=True)

    t = -1
    for idx, row in label.iterrows():
        if row['name'] in file and row['category'] == file[0]:
            t = row['time']
            break

    if t == -1:
        print(file)
        continue
        # raise "Not Find Time"
    
    length = len(data)
    labels = np.linspace(t, 0, length)
    data['label'] = labels

    ls[-1] = data
    print(data)
    
 
data = pd.concat(ls, ignore_index=True)
print(data)
data.to_csv(output_path, index=None)