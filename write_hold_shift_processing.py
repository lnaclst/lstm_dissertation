import h5py
import numpy as np
import pandas as pd


seconds = 1
frames = 1
frame_diff = seconds*10
hold_shift_pause_length = 250
frame_start_diff = hold_shift_pause_length*0.1
split_file = 'q1.txt'
hold_shift = h5py.File('./data/datasets/hold_shift.hdf5','r')
# role = 'g'
hold_shift_id = 0     # 0 for hold, 1 for shift
# feature_labels = pd.read_csv('./data/extracted_annotations/voice_activity/q1ec1.g.csv')
features_list = ['conv','role','frameIdx', 'F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                        'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                        'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                        'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                        'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                        'mfcc4']
data = False
# zeros = np.zeros((1,len(features_list)))
# df = pd.DataFrame(zeros, columns=features_list)


# with open('./data/splits/{}'.format(split_file), 'r') as split:
#     for line in split:
for role in ['g','f']:
    # conv = line.strip()
    conv = 'q5ec6'
    ft = pd.read_csv('./data/signals/gemaps_features_processed_10ms/znormalized/{}.{}.csv'.format(conv,role))
    print(conv,role)
    print('Features loaded.')
    # print(ft['frame_time'])
    dset = hold_shift['250ms']['hold_shift'][conv][role]
    arr = np.zeros(dset.shape)
    # print(dset.shape)
    dset.read_direct(arr)
    print("Hold-shift values array filled.")
    # for i in arr:
    #     print(i[0],i[1])
    for i in arr:
        # print(i[0])
        if i[0] >= 0:
            # print('Greater than 20.')
            if i[1] == hold_shift_id:
                print('Hold at {}'.format(i))
                row = [conv,role]
                frame = int(i[0])-frame_start_diff
                frame_prior = int(frame-frame_diff)
                mat = ft.loc[frame_prior:frame, :]
                mat = mat.mean(0)
                # print("Mat shape = ", mat.shape)
                mat_to_list = list(mat)
                row.extend(mat)
                print(row)
                # new_mat = np.array(row)
                # # mat = list_to_mat.transpose()
                # print("Mat: ", new_mat)
                # # print("Row shape =", len(row))
                # print(len(row))
                # print(len(features_list))
                df_temp = pd.DataFrame([row], columns=features_list)
                if data == False:
                    df = df_temp
                else:
                    df = df.append(df_temp)

                data = True

if hold_shift_id == 0:
    hs = 'hold'
elif hold_shift_id == 1:
    hs = 'shift'
else:
    hs = 'hs'
df.to_csv('avg_{}_{}_1f.csv'.format(hs, conv))


