import numpy as np
import os
import pandas as pd

if not os.path.exists('./data/extracted_annotations/floorholder/'):
    os.makedirs('./data/extracted_annotations/floorholder/')

print('Line 9')

file_list_path = './data/splits/complete.txt'
va_annotations_path = './data/extracted_annotations/voice_activity/'
fl_annotations_path = './data/extracted_annotations/floorholder/'

print('Line 15')

test_file_list = list(pd.read_csv(file_list_path,header=None)[0])
annotations_dict = {}
frame_times_dict = {}

length_past_window = 10

print('Line 23')

for file_name in test_file_list:
    data_f = pd.read_csv(va_annotations_path+'/'+file_name+'.f.csv')
    data_g = pd.read_csv(va_annotations_path+'/'+file_name+'.g.csv')
    annotations = np.column_stack([np.array(data_g)[:, 1].astype(bool), np.array(data_f)[:, 1].astype(bool)])
    annotations_dict[file_name] = annotations
    frame_times_dict[file_name] = np.array(data_g)[:, 0]

    g_csv = fl_annotations_path + file_name + '.g.csv'
    f_csv = fl_annotations_path + file_name + '.f.csv'

    g_floorholder = []
    f_floorholder = []

    print(file_name, "defined")

    for line in range(len(annotations_dict[file_name])):
        if line <= length_past_window:
            g_floorholder.append(0)
            f_floorholder.append(0)
        # print("Zeros appended.")
        if line > length_past_window:
            if sum(annotations[line - length_past_window:line + 1, 0]) > sum(
                    annotations[line - length_past_window:line + 1, 1]):
                g_floorholder.append(1)
                f_floorholder.append(0)
            elif sum(annotations[line - length_past_window:line + 1, 0]) == sum(
                    annotations[line - length_past_window:line + 1, 1]):
                g = g_floorholder[-1]
                f = f_floorholder[-1]
                g_floorholder.append(g)
                f_floorholder.append(f)
            else:
                g_floorholder.append(0)
                f_floorholder.append(1)

    print("Data lists created.")

    g_d = {'frameTimes':frame_times_dict[file_name], 'val':g_floorholder}
    f_d = {'frameTimes':frame_times_dict[file_name], 'val':f_floorholder}

    g_df = pd.DataFrame(data=g_d)
    f_df = pd.DataFrame(data=f_d)

    print("Dataframes created")

    g_df.to_csv(g_csv, columns=['frameTimes', 'val'],index=False)
    f_df.to_csv(f_csv, columns=['frameTimes', 'val'],index=False)

    print("CSVs created")