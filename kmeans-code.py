import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import math
import write_hold_shift_processing as whp


if whp.hold_shift_id == 0:
    hs = 'hold'
elif whp.hold_shift_id == 1:
    hs = 'shift'
else:
    hs = 'hs'
csv_file = 'avg_{}_{}_1f.csv'.format(hs, whp.conv)

gemaps_features_list = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency', 'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness','shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex','spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0','F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3','mfcc4']
gemaps_features_list_role = ['role','frameIdx','F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                        'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                        'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                        'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                        'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                        'mfcc4']

# features = pd.read_csv('avg_feats_array.csv', usecols=gemaps_features_list, encoding='utf-8')
# features = pd.read_csv(csv_file, usecols=gemaps_features_list, encoding='utf-8')
features = pd.read_csv(csv_file, usecols=gemaps_features_list, encoding='utf-8')
features_role = pd.read_csv(csv_file, usecols=gemaps_features_list_role, encoding='utf-8')

# features = features_role.loc[['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency', 'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness','shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex','spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0','F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3','mfcc4']]

clusters = 2

kmeans = KMeans(n_clusters=clusters).fit(features)
print('Kmeans')
pca = PCA(2)
pca.fit(features)
print('Fit')
pca_data = pd.DataFrame(pca.transform(features))
print(pca_data.head())

colors = list(zip(*sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).items())))[1]
print('Colors')

# number of steps to taken generate n(clusters) colors
skips = math.floor(len(colors[5 : -5])/clusters)
cluster_colors = colors[5 : -5 : skips]
print('Skips')

fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
ax = fig.add_subplot(111)
# ax.scatter(pca_data[0], pca_data[1], pca_data[2], c = list(map(lambda label : cluster_colors[label],
#                                             kmeans.labels_)))
ax.scatter(pca_data[0], pca_data[1], c = list(map(lambda label : cluster_colors[label],
                                            kmeans.labels_)))
print('Figure')

str_labels_frameIdx = list(map(lambda label: '% s' % label, features_role['frameIdx']))
str_labels_role = list(map(lambda label: '% s' % label, features_role['role']))
# kmeans.labels_))
# str_labels_frameIdx = [i[:5] for i in str_labels_frameIdx]
str_labels = []
for i in range(len(str_labels_frameIdx)):
    str_labels.append(str_labels_frameIdx[i][:5] + str_labels_role[i])
print(str_labels[0])

# list(map(lambda data1, data2, data3, str_label:
#          ax.text(data1, data2, data3, s=str_label, size=16.5,
#                  zorder=20, color='k'), pca_data[0], pca_data[1],
         # pca_data[2], str_labels))

list(map(lambda data1, data2, str_label:
         ax.text(data1, data2, s=str_label, size=16.5,
                 zorder=20, color='k'), pca_data[0], pca_data[1], str_labels))

# print('List')
plt.savefig('{}.png'.format(csv_file[:-4]))
plt.show()

# df = pd.DataFrame(features, columns=gemaps_features_list)



# avg_by_spkr = np.mean(features, axis=0)

# print(len(avg_by_spkr), len(avg_by_spkr[0]))



