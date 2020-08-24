import torch
import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap
from os import listdir
from os import walk
from os.path import isfile, join
import pandas as pd

mypath = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/hidden/'
features = '_Ffeatures'
person = 'q5ec2f'
cos = torch.nn.CosineSimilarity(dim = 0)

folders = []
for (dirpath, dirnames, filenames) in walk(mypath):
    if 'hidden' not in dirpath[-7:]:
        if person in dirpath:
            if features in dirpath:
                folders.append(dirpath)
# print(len(folders))
folders = sorted(folders)
print(folders)

gen_str1 = '-1.pt'
gen_strFG = '-2.pt'
acous_str = '-acous.pt'
vis_str = '-visual.pt'
labels, gen_files, acous_files, ling_files = [], [], [], []
for filepath in folders:
    labels.append(filepath[-13:].strip('_'))
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]

    count = max([int(i[:2].strip('-')) for i in files])
    if '-2' in files[0]:
        gen_files.append(str(count) + gen_strFG)
    else:
        gen_files.append(str(count) + gen_str1)
    acous_files.append(str(count) + acous_str)
    ling_files.append(str(count) + vis_str)
# print(gen_files[:5])
# print(acous_files[:5])
# print(ling_files[:5])

# print(len(gen_files)==len(acous_files))
# print(len(acous_files)==len(ling_files))
# print(len(gen_files) == len(labels))
# print(len(labels) == len(folders))

### Create heatmap
size = len(gen_files)
size_up = size + int(size/3)

rand_embed = torch.randn([50])


def draw_heatmap(mod_list):
    heat_mat_a = np.zeros((size, size))
    heat_mat_b = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            embedding_path1 = folders[i] + '/' + mod_list[i]
            embedding_path2 = folders[j] + '/' + mod_list[j]

            embedding1 = torch.load(embedding_path1, map_location=torch.device('cpu'))
            embedding2 = torch.load(embedding_path2, map_location=torch.device('cpu'))

            embedding1 = torch.squeeze(embedding1, 0)  # [0]
            embedding2 = torch.squeeze(embedding2, 0)  # [0]

            if embedding1.shape[0] != 1:
                print(embedding1.shape)
                print('1')
                embedding1 = np.mean(embedding1.detach().numpy(), axis=0)
                embedding1 = torch.tensor(embedding1)
                print(embedding1.shape)
            else:
                embedding1 = embedding1[0]
            if embedding2.shape[0] != 1:
                print(embedding2.shape)
                print('2')
                embedding2 = np.mean(embedding2.detach().numpy(), axis=0)
                embedding2 = torch.tensor(embedding2)
                print(embedding2.shape)
            else:
                embedding2 = embedding2[0]

            # embedding1 = torch.squeeze(embedding1, 0)  # [0]
            # embedding2 = torch.squeeze(embedding2, 0)  # [0]

            # print(embedding1[0].shape)
            # print(embedding2[0].shape)

            heat_mat_a[i][j] = cos(embedding2, embedding1)

    print(heat_mat_a)
    # print(heat_mat_b)

    df = pd.DataFrame(data=heat_mat_a, columns=labels)

    heat_map = heatmap(df, vmin=-1, vmax=1, annot=True, yticklabels=labels, xticklabels=True)
    plt.title(person)
    plt.show()

    # df = pd.DataFrame(data=heat_mat_b, columns=labels)
    #
    # heat_map = heatmap(df, annot=True)
    # plt.show()

draw_heatmap(ling_files)


# def create_heatmap(mod_list):
#     heat_mat = np.zeros((size_up, size_up))
#     print(len(folders))
#     print(len(mod_list))
#     i_count = 0
#     print(size)
#     for i in range(size):
#         j_count = 0
#         for j in range(size):
#
#             print('Trying i ', i)
#             embedding_path1 = folders[i] + '/' + mod_list[i]
#             print('Trying j ', j)
#             embedding_path2 = folders[j] + '/' + mod_list[j]
#
#             embedding1 = torch.load(embedding_path1, map_location=torch.device('cpu'))
#             embedding2 = torch.load(embedding_path2, map_location=torch.device('cpu'))
#
#             embedding1 = torch.squeeze(embedding1, 0)  # [0]
#             embedding2 = torch.squeeze(embedding2, 0)  # [0]
#
#             if embedding1.shape[0] == 2:
#                 print('a')
#                 if embedding2.shape[0] == 2:
#                     print('a')
#                     # print(embedding1.shape)
#                     v11 = embedding1[0]
#                     v12 = embedding1[1]
#                     v21 = embedding2[0]
#                     v22 = embedding2[1]
#
#                     print('i_count ', i_count)
#                     i_index = i + i_count
#                     print('j_count ', j_count)
#                     j_index = j + j_count
#
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(v11, v21, 0)
#                     j_index += 1
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(v11, v22, 0)
#                     i_index += 1
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(v12, v22, 0)
#                     j_index -= 1
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(v12, v21, 0)
#
#                     print("a add")
#                     i_count += 1
#                     j_count += 1
#
#                 else:
#                     print('b')
#                     # print(embedding1.shape)
#
#                     v1 = embedding1[0]
#                     v2 = embedding1[1]
#                     # print(v1.shape)
#                     # print(v2.shape)
#                     embedding2 = embedding2.squeeze()
#
#                     i_index = i + i_count
#                     j_index = j + j_count
#
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(v1, embedding2, 0)
#                     j_index += 1
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(v2, embedding2, 0)
#
#                     print("b add")
#                     j_count += 1
#
#             elif embedding2.shape[0] == 2:
#                 print('c')
#                 # print(mod_list[j])
#                 v1 = embedding2[0]
#                 v2 = embedding2[1]
#                 # print(v1.shape)
#                 # print(v2.shape)
#                 # v1 = torch.squeeze(v1, 0)  # [0]
#                 # v2 = torch.squeeze(v2, 0)  # [0]
#                 embedding1 = embedding1.squeeze()
#
#                 i_index = i + i_count
#                 j_index = j + j_count
#
#                 heat_mat[i_index][j_index] = torch.cosine_similarity(v1, embedding1, 0)
#                 i_index += 1
#                 heat_mat[i_index][j_index] = torch.cosine_similarity(v2, embedding1, 0)
#
#                 print("c add")
#                 i_count += 1
#
#             elif embedding1.shape[0] == 1:
#                 print('d')
#                 if embedding2.shape[0] == 1:
#                     print('d')
#                     # print(embedding1.shape)
#                     # print(embedding2.shape)
#                     # print(torch.cosine_similarity(embedding2, embedding1, 0))
#
#                     print("d add")
#                     i_index = i + i_count
#                     j_index = j + j_count
#
#                     heat_mat[i_index][j_index] = torch.cosine_similarity(embedding2, embedding1, 1)
#
#             else:
#                 print('Something went wrong')
#
#     heat_map = heatmap(heat_mat)
#


##### Do not use below vvv #####


# def my_heatmap(mod_list):
#     heat_mat_a = np.zeros((size, size))
#
#     embedding_list = []
#
#     for i in range(size):
#         embedding_path1 = folders[i] + '/' + mod_list[i]
#
#         embedding1 = torch.load(embedding_path1, map_location=torch.device('cpu'))
#         embedding1 = torch.squeeze(embedding1, 0)  # [0]
#
#         print(embedding1.shape)
#
#         if embedding1.shape[0] != 1:
#             embedding_list.append(embedding1[0])
#             embedding_list.append(embedding1[1])
#         else:
#             embedding_list.append(embedding1)
#
#     embedding_tensor = torch.stack(embedding_list,0)
#     embedding_tensor = embedding_tensor.transpose(0,1)
#     print(embedding_tensor.shape)
#
#     # df = pd.DataFrame(data=embedding_tensor, columns=labels)
#
#     heat_map = heatmap(embedding_tensor.detach().numpy(), vmin=-1, vmax=1, annot=True, yticklabels=labels, xticklabels=labels)
#     plt.title(person)
#     plt.show()
#
# my_heatmap(gen_files)