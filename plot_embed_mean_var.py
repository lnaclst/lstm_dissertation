from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


gen = torch.load('/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/train_q1ec2f_60pred_g/16-2.pt',map_location=torch.device('cpu'))#.detach().numpy()
gen_f = torch.load('/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/train_q1ec1g_60pred_g/8-2.pt',map_location=torch.device('cpu'))#.detach().numpy()
# acous = torch.load('/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/train_baseline_split3/19-acous.pt',map_location=torch.device('cpu')).detach().numpy()
# ling = torch.load('/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/embeddings/Pre-LSTM-out/train_baseline_split3/19-visual.pt',map_location=torch.device('cpu')).detach().numpy()

gen1 = gen.squeeze()[0]
gen_f1 = gen_f.squeeze()[1]
# gen2 = gen.squeeze()[1]
# gen_f2 = gen_f.squeeze()[1]
print(gen1.shape)
print(gen_f1.shape)

sim = torch.cosine_similarity(gen1,gen_f1,0).detach().numpy()
print(sim)
h_plot1 = plt.hist(sim, label="Cos Sim Hist")


# print(gen1)
# print(gen2)

# acous = acous.squeeze()
# ling = ling.squeeze()

# pca1 = PCA(1)    # Do I need to define a PCA function for each embedding I want to fit?
# pca2 = PCA(1)
# pca3 = PCA(1)
# pca1.fit(gen)
# pca2.fit(gen_f)
# # pca2.fit(acous)
# # pca3.fit(ling)
#
# pca_data1 = np.array(pca1.transform(gen))
# pca_data2 = np.array(pca2.transform(gen_f))

# pca_data2 = np.array(pca2.transform(acous))
# pca_data3 = np.array(pca3.transform(ling))


# h_plot1 = plt.hist(gen1, label="Flipped1")
# h_plot2 = plt.hist(gen_f1, label='q1ec1F1')
# h_plot3 = plt.hist(gen2, label="Flipped2")
# h_plot4 = plt.hist(gen_f2, label='q1ec1F2')
# h_plot3 = plt.hist(pca_data3, label='Ling')

plt.legend()
plt.show()


