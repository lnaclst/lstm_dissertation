import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

embedding_path2 = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/hidden/train_q4ec1g_60pred_Gfeatures_mods__hidden/ing_50ms_q4ec1g_0_g_60/20-1.pt'
embedding_path3 = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/hidden/train_q4ec1f_60pred_Ffeatures_mods__hidden/ing_50ms_q4ec1f_0_f_60/20-1.pt'
# /Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/train_q1ec2g_60pred_Ffeatures_mods/2_Acous_10ms_Ling_50ms/16-acous.pt

embedding1 = torch.load(embedding_path2, map_location=torch.device('cpu'))
embedding2 = torch.load(embedding_path3, map_location=torch.device('cpu'))

print('1', embedding1.size())
print('2', embedding2.size())

embedding1 = torch.squeeze(embedding1, 0)#[0]
embedding2 = torch.squeeze(embedding2, 0)#[0]

# embedding1 = embedding1.view(1,100)
# embedding2 = embedding2.view(1,100)

print('3', embedding1.size())
print('4', embedding2.size())

# embedding1 = embedding1[torch.randperm(600)]
# embedding2 = embedding1[torch.randperm(128)]
# embedding1 = torch.randn([600,128,50])
# embedding2 = torch.randn([128,50])

cov_mat = np.cov(embedding2.detach().numpy())
# mean_mat = np.mean(embedding2.detach().numpy(),0)
# # print(mean_mat)
# # mean = np.mean(mean_mat)
# # varc = np.var(mean_mat)

cos = torch.cosine_similarity(embedding2, embedding1, 1)
try:
    num_embeds = cos.shape[0]*cos.shape[1]
except IndexError:
    num_embeds = cos.shape[0]
print(cos.shape)
print("Max: ", torch.max(cos))
print("Min: ", torch.min(cos))
range = 0
low = 0
high = 0
for j in cos:
    # for j in i:
    if -0.3<=j<=0.3:
        range += 1
    if -0.3>j:
        low += 1
    if j>0.3:
        high += 1
print(range, low, high, num_embeds)
print("Range -0.3-0.3: {}%, Low <-0.3: {}%, High >0.3: {}%".format(range/num_embeds, low/num_embeds, high/num_embeds))
print(cos.size()[0] == 2)
print(np.mean(cos.detach().numpy()))
h_plot1 = plt.hist(cos.detach().numpy(), label="Cosine Hist")


plt.legend()
plt.show()




# sim = 0
# dissim = 0
# for i in cos:
#     if i < 0:
#         dissim += 1
#     elif i > 0.6:
#         sim += 1
# print(sim)
# print(dissim)
# print(128-sim-dissim)


# embedding_for_size = torch.load(embedding_path1, map_location=torch.device('cpu'))

# map = torch.zeros(128)
# a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# for i in a:
#     for j in a:
#         print(i,j)
#         embedding_path1 = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/train_roleidprediction_hidden-master/{}-128.pt'.format(int(i))
#         embedding_path2 = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/train_roleidprediction_hidden-master/{}-128.pt'.format(int(j))
#
#         embedding1 = torch.load(embedding_path1, map_location=torch.device('cpu'))
#         embedding2 = torch.load(embedding_path2, map_location=torch.device('cpu'))
#
#         embedding1 = torch.squeeze(embedding1, 0)  # [1]
#         embedding2 = torch.squeeze(embedding2, 0)  # [1]
#
#         print('3', embedding1.size())
#         print('4', embedding2.size())
#
#         cos = torch.cosine_similarity(embedding2, embedding1, 1)
#         print(cos)
#
#         map = map.add(cos)
#
#         # sim = 0
#         # dissim = 0
#         # for i in cos:
#         #     if i < 0:
#         #         dissim += 1
#         #     elif i > 0.6:
#         #         sim += 1
#         # print(sim)
#         # print(dissim)
#         # print(128-sim-dissim)
#
# print(map)
# map_final = map/(16**2)
# print(map_final)

# output = cos(embedding1,embedding2)

# x = 'q1nc2q8ec7q6ec7q8nc7q8nc1q1nc4q3nc8q6nc8q5ec8q8ec2q2ec8q2ec2q1nc4q1nc8q8ec4q5nc8q3ec2q5nc2q6nc3q8nc5q2nc3q2nc8q1nc8q2ec4q1ec3q5nc7q3ec3q1nc6q3ec2q3nc8q6nc5q3nc5q8nc7q8ec4q3ec3q3ec7q2ec8q5nc7q1nc6q5nc4q3nc4q5nc7q6nc2q1nc3q8nc1q1ec4q6nc2q1nc4q6ec1q2nc1q3nc7q3nc7q8nc8q1ec3q2nc7q1nc7q8ec1q5nc2q1nc3q8ec6q2nc5q6ec1q2nc3q1nc2q6ec4q2ec6q1ec3q5nc2q1nc4q2nc3q1nc4q8nc1q2ec6q2ec1q3ec6q6nc5q3ec7q1nc6q6nc4q8nc8q8ec6q5ec4q1ec7q3nc7q1nc7q6nc8q8nc8q6nc1q5nc6q8nc5q2nc3q3nc6q3ec1q6nc1q8nc3q8ec6q5nc7q3ec7q8ec6q5ec3q8ec4q6nc6q6nc1q2nc2q6nc4q8nc8q6ec5q1ec2q2nc8q6ec8q6ec1q1ec8q8nc4q5ec3q2nc2q1nc2q2nc3q1ec2q8ec2q1nc2q8nc5q2ec3q5nc3q6nc6q8ec4q1nc3q6ec8q1nc2'
# print(len(x)/5)

# train_1
# ff tensor([-0.0017], grad_fn=<DivBackward0>)

# Note: more negative is more different

# train_20
# ff tensor([0.0297], grad_fn=<DivBackward0>)
# fg tensor([-0.0906], grad_fn=<DivBackward0>)
# fg tensor([-0.0054], grad_fn=<DivBackward0>)
# gg tensor([-0.0361], grad_fn=<DivBackward0>)




# tensor([[-0.0718,  0.0170]], grad_fn=<DivBackward0>)
# tensor([[0.0687, 0.1003]], grad_fn=<DivBackward0>)

# After applying view - combines f and g, so this really just tells me if the similar ones are more similar, or if the different ones are more different
# ff tensor([0.0775], grad_fn=<DivBackward0>)
# fg tensor([-0.0791], grad_fn=<DivBackward0>)
# fg tensor([0.1340], grad_fn=<DivBackward0>)
# fg tensor([0.0961], grad_fn=<DivBackward0>)
# fg tensor([-0.1065], grad_fn=<DivBackward0>)
# fg tensor([0.1508], grad_fn=<DivBackward0>)


# Dist
# ff tensor(2.4506, grad_fn=<DistBackward>)
# fg tensor(3.4724, grad_fn=<DistBackward>)


# These 2 are more similar for some reason
# embedding_path1 = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/two_subnets_train_1-frame_f_embeddings_17-7-4-14-f.pt'
# embedding_path2 = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings_temp/two_subnets_train_1-frame_g_embedding_17-7-22-8-g.pt'

# is there a general trend towards fs being more similar and gs being less similar?


# 1g embedding[0] is strongly correlated to the baseline[0] embedding
# same for the idx[1] embeddings ^^
# similarly, the [0] and [1] indexes from the different embeddings are negatively correlated to each other

# the indexes in the embeddings trained together are correlated, but not as strongly as the equal indexes in the differently trained embeddings
# differently trained embeddings: 0
# tensor(0.2477, grad_fn=<DivBackward0>)
# differently trained embeddings: 1
# tensor(0.3836, grad_fn=<DivBackward0>)

# jointly trained: baseline
# tensor(0.1790, grad_fn=<DivBackward0>)
# jointly trained: two_subnets_train_1-frame_g_embedding_17-7-23-12-g.pt
# tensor(0.1123, grad_fn=<DivBackward0>)

# totally opposite have negative similarity
# baseline[0]: tensor(-0.0859, grad_fn=<DivBackward0>)
# two_subnets_train_1-frame_g_embedding_17-7-23-12-g.pt[0]: tensor(-0.1455, grad_fn=<DivBackward0>)


# Embeddings trained on baseline are all about equally correlated, both between the [0] and [1] indexes and the different models
# Actually seems sort of random how they are correlated


# G from one conversation to the other is pretty different. Are the maps identical or are they using 2 different maps and sets of steps?