import os

import numpy as np
import pickle
import pandas as pd

dir_path = './data/signals/gemaps_features_10ms'
file_list_path = './data/splits/'
gemaps_features_list = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                        'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                        'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                        'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                        'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                        'mfcc4']

data = []
counts = np.array([])
# row = np.zeros((1,len(gemaps_features_list)))
# df = pd.DataFrame(row, columns=gemaps_features_list)

with open('{}complete.txt'.format(file_list_path), 'r') as conv_list:
    for line in conv_list:
        conv_name = line.strip()
        print(conv_name)
        data_temp_g = pd.read_csv('{}/{}.{}.csv'.format(dir_path,conv_name,'g'), usecols=gemaps_features_list)
        data_temp_f = pd.read_csv('{}/{}.{}.csv'.format(dir_path,conv_name,'f'), usecols=gemaps_features_list)
        data_temp = data_temp_g.append(data_temp_f)
        print("GF appended.")
        if type(data) == np.ndarray:
            data = pd.DataFrame(data, columns=gemaps_features_list)
            print("Data made df.")
        # print(len(data_temp), len(data_temp[0]))
        counts_temp = [1]*data_temp.shape[0]
        print("Counts_temp: ", len(counts_temp))
        # data = list(data)
        counts = list(counts)
        if len(data) == 0:
            data = data_temp
            counts = counts_temp
        else:
            if data.shape[-1] == data_temp.shape[-1]:
                print("Equal feature nums.")
                print("Num frames: ", data.count())
                if len(data) != len(data_temp):
                    if len(data) > len(data_temp):
                        diff = abs(data_temp.shape[0]-data.shape[0])
                        print("Data")
                        counts_temp += [0]*diff
                        print("Counts_temp: ", len(counts_temp))
                        data_zeros = np.zeros((diff, data.shape[-1]))
                        df = pd.DataFrame(data_zeros, columns=gemaps_features_list)
                        # for i in range(data.shape[0]-data_temp.shape[0]):
                        data_temp = data_temp.append(df)
                        print(data_temp.shape, data.shape)
                    else:
                        diff = abs(data_temp.shape[0]-data.shape[0])
                        print("Data_temp")
                        counts += [0]*diff
                        print("Counts: ", len(counts))
                        print(data_temp.shape[0],data.shape[0])
                        data_zeros = np.zeros((diff, data.shape[-1]))
                        df = pd.DataFrame(data_zeros, columns=gemaps_features_list)
                        # for i in range(data_temp.shape[0]-data.shape[0]):
                        data = data.append(df)
                        print(data.shape[0])
            # data, data_temp = np.array(data), np.array(data_temp)
            counts, counts_temp = np.array(counts), np.array(counts_temp)
            print(counts[:10])
            print(counts_temp[:10])
            data, data_temp = data.to_numpy(), data_temp.to_numpy()
            if data.shape[0] == data_temp.shape[0]:
                print("Shapes match.")
                data = np.add(data, data_temp)
                print("Data added.")
                counts = np.add(counts, counts_temp)
                print("Counts added.")
        print(counts[:10])

counts = list(counts)
counts_mat = []
for i in counts:
    counts_mat.append([i]*data.shape[-1])
counts_mat = np.array(counts_mat)
counts_df = pd.DataFrame(counts_mat, columns=gemaps_features_list)
if counts_df.shape == data.shape:
    print("Counts shapes match.")
    avg_mat = data/counts_df
else:
    print("Counts shape error.")

np.save('avg_feats_array.npy', avg_mat)
avg_mat.to_csv('avg_feats_array.csv')

# counts = np.reshape(counts,(counts.shape[1],counts.shape[0]))


# # features_per_speaker = []
#
# feat_conv_dict = {}
# with open('{}complete.txt'.format(file_list_path), 'r') as conv_list:
#     for line in conv_list:
#         conv_name = line.strip()
#         feat_conv_dict[conv_name] = []
#         for gf in ['g','f']:
#             try:
#                 with open('{}/{}.{}.csv'.format(dir_path,conv_name,gf), newline='') as csv_file:
#                     data = list(csv.reader(csv_file))
#                     # print(type(data))
#                     # print(len(data), len(data[0]))
#                     # features_per_speaker.append(data)
#                     print("Processed: ",conv_name, gf)
#                     feat_conv_dict[conv_name].extend(data)
#                     print(conv_name, len(feat_conv_dict[conv_name]))
#             except FileNotFoundError:
#                 print(conv_name,gf)
#
# with open('feat_dict.p', 'wb') as feat_dict_doc:
#     pickle.dump(feat_conv_dict, feat_dict_doc)
#
# print('Files processed.')
#
# feature_lengths = []
# for i in feat_conv_dict.values():
#     feature_lengths.append(len(i))
#
# num_frames = min(feature_lengths)
# print(num_frames)
#
# for key in feat_conv_dict.keys():
#     feat_conv_dict[key] = feat_conv_dict[key][:num_frames]
# print(len(feat_conv_dict.keys()))

# with open('feat_dict.p','rb') as pickle_doc:
#     feat_conv_dict = pickle.load(pickle_doc)
#     print("Data loaded.")
#
# x = np.array(list(feat_conv_dict.values()))
# print(x.size)
# avg_x = np.mean(x, axis=2, dtype=np.float64)
# print('X averaged')
# print(len(avg_x), len(avg_x[0]))
#
# np.save('features_complete_array1.npy', x)
#
# print(x.size)