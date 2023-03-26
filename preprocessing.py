import pandas as pd
import numpy as np
import os


################## Configuration ####################
percent = 0.1
times = 10
folder_path = '/root/wangning/data'
label_path = '/root/wangning/preprocessing/label.csv'
output_path = '/root/wangning/preprocessing/data.csv'
#####################################################

label = pd.read_csv(label_path)
files = os.listdir(folder_path)

files2 = [i for i in files if 'xr' in i]
files1 = [i for i in files if i not in files2]
files1 = sorted(files1)
files2 = sorted(files2)

ls1 = []

for file in files1:

    ls1.append(pd.DataFrame(columns=['file', 'x', 'y', 'z', 'a']))
    data = ls1[-1]

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
    drop_list = list(data[data['x'] <= x_mean].index)
    for i in range(1, len(drop_list)):
        if drop_list[i] == drop_list[i-1] + 1:
            num += 1
        else:
            if num >= times:
                data = data.drop(index=list(range(drop_list[i-1]-num, drop_list[i-1])))
            num = 0
    data = data.reset_index(drop=True)

    t = -1
    for idx, row in label.iterrows():
        if row['name'] in file and row['category'] == file[0]:
            t = row['time']
            break

    if t == -1:
        print(file, "is not found")
        continue
        # raise "Not Find Time"
    
    start = ls1[-2].iloc[-1].at['label'] if len(ls1) > 1 else 0

    length = len(data)
    interval = (t - start) / length if len(ls1) > 1 else 0
    labels = np.linspace(start+interval, t, length)
    data['label'] = labels

    ls1[-1] = data
    print(data)


ls2 = []

for file in files2:

    ls2.append(pd.DataFrame(columns=['file', 'x', 'y', 'z', 'a']))
    data = ls2[-1]

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
    drop_list = list(data[data['x'] <= x_mean].index)
    for i in range(1, len(drop_list)):
        if drop_list[i] == drop_list[i-1] + 1:
            num += 1
        else:
            if num >= times:
                data = data.drop(index=list(range(drop_list[i-1]-num, drop_list[i-1])))
            num = 0
    data = data.reset_index(drop=True)

    t = -1
    for idx, row in label.iterrows():
        if row['name'] in file and row['category'] == file[0]:
            t = row['time']
            break

    if t == -1:
        print(file, "is not found")
        continue
        # raise "Not Find Time"
    
    start = ls2[-2].iloc[-1].at['label'] if len(ls2) > 1 else 0

    length = len(data)
    interval = (t - start) / length if len(ls2) > 1 else 0
    labels = np.linspace(start+interval, t, length)
    data['label'] = labels

    ls2[-1] = data
    print(data)

data1 = pd.concat(ls1, ignore_index=True)
data1.label = data1.label / data1.label.max()
data2 = pd.concat(ls2, ignore_index=True)
data2.label = data2.label / data2.label.max()
data = pd.concat([data1, data2], ignore_index=True)
print(data)
data.to_csv(output_path, index=None)