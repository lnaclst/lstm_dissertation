# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import os


#%% Load file
file_list_path = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/data/splits/complete.txt'
va_annotations_path = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/data/extracted_annotations/voice_activity/'

# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'file_hold_shift' in locals():
    if isinstance(file_hold_shift,h5py.File):   # Just HDF5 files
        try:
            file_hold_shift.close()
        except:
            pass # Was already closed
            
if 'file_floorholder' in locals():
    if isinstance(file_floorholder,h5py.File):   # Just HDF5 files
        try:
            file_floorholder.close()
        except:
            pass # Was already closed

if not(os.path.exists('./data/datasets/')):
    os.makedirs('./data/datasets/')
file_hold_shift = h5py.File('./data/datasets/hold_shift.hdf5','w')
file_floorholder = h5py.File('./data/datasets/floorholder.hdf5','w')

pause_str_list = ['50ms','250ms','500ms']
pause_length_list = [1,5,10]
length_of_future_window = 20 # (one second) find instances where only one person speaks after the pause within this future window
### Lena exp. 1
length_past_window = 20 #use the same length as the window Roddy uses to determine who speaks next in future speaker predictions 

#%% get Annotations
# Load list of test files
test_file_list = list(pd.read_csv(file_list_path,header=None)[0])
annotations_dict = {}
frame_times_dict = {}
for file_name in test_file_list:
    data_f = pd.read_csv(va_annotations_path+'/'+file_name+'.f.csv')
    data_g = pd.read_csv(va_annotations_path+'/'+file_name+'.g.csv')
    annotations = np.column_stack([np.array(data_g)[:,1].astype(bool),np.array(data_f)[:,1].astype(bool)])
    annotations_dict[file_name] = annotations
    frame_times_dict[file_name] = np.array(data_g)[:,0]


#%% loop through pause lengths

for pause_str, pause_length in zip(pause_str_list,pause_length_list):
    pause_seq_list = list()
    g_turns_list = list()
    f_turns_list = list()
    g_hold_shift_list = list()
    f_hold_shift_list = list()
    #Lena exp. 1
    floorholder_list, g_floorholder_list, f_floorholder_list = [], [], []
    
    turns_sum = 0
    list_sum = 0
    uncertain_sum = 0
    num_holds = 0
    num_shifts = 0
    
    for seq in test_file_list:
        annotations = annotations_dict[seq]

#         print(annotations)
        #%% find pauses
        ### Creates a new list 
        pauses = ~(annotations[:,0] | annotations[:,1]) 
        # False is when someone is speaking
#         print(pauses)
#         quit()
        
        pause_list = list()
        g_hold_shift = list()
        f_hold_shift = list()
        uncertain = list()
        
        ### Lena, exp. 1: Create lists of where g is speaking and where f is speaking before a pause
        g_floorholder, f_floorholder = [], []
        
        # pause_length = 10 # 500 ms
        for i in range(pause_length,len(pauses)):    
            if all(pauses[i-pause_length:i]==True) & (pauses[i-pause_length-1]==False):
                # check if last speech frame had speech by both speakers
                if (annotations[i-pause_length-1,0] & annotations[i-pause_length-1,1]):
                    uncertain.append(i)
                else:
                    pause_list.append(i) # start calculating who takes turn from i (inclusive)
                        
        pause_seq_list.append(pause_list)
        list_sum +=len(pause_list)
        uncertain_sum += len(uncertain)
        
        ### Edit to change from marking future speech to marking past speech activity
        for i in pause_list:                                    # where all the pauses are for a given pause length
            if annotations[i-pause_length-1,0]:      # first value being the index of the frame right before the pause started, second value being the speaker (0 == g)
                if (any(annotations[i:i+length_of_future_window,0]) and not(any(annotations[i:i+length_of_future_window,1]))):
                    g_hold_shift.append([i,0]) # [index of pause, hold(0) or shift(1)]
                    num_holds += 1
                    turns_sum += 1
                if any(annotations[i:i+length_of_future_window,1]) and not(any(annotations[i:i+length_of_future_window,0])):
                    g_hold_shift.append([i,1]) # [index of pause, hold(0) or shift(1)]
                    num_shifts += 1
                    turns_sum += 1
            elif annotations[i-pause_length-1,1]:
                if any(annotations[i:i+length_of_future_window,1]) and not(any(annotations[i:i+length_of_future_window,0])):
                    f_hold_shift.append([i,0]) # [index of pause, hold(0) or shift(1)]
                    num_holds += 1
                    turns_sum += 1                
                if any(annotations[i:i+length_of_future_window,0]) and not(any(annotations[i:i+length_of_future_window,1])):
                    f_hold_shift.append([i,1]) # [index of pause, hold(0) or shift(1)]
                    num_shifts += 1
                    turns_sum += 1
            else:
                raise('pause list indexing problem')
                
            #Lena exp. 1
            if (sum(annotations[i-pause_length-length_past_window:i,0]) > sum(annotations[i-pause_length-length_past_window:i,1])):
                g_floorholder.append([i,1])
                f_floorholder.append([i,0])

            else: 
                g_floorholder.append([i,0])   
                f_floorholder.append([i,1])

        # Lena exp. 1
        g_floorholder_list.append(g_floorholder)
        f_floorholder_list.append(f_floorholder)
        
        g_hold_shift_list.append(g_hold_shift)
        f_hold_shift_list.append(f_hold_shift)

    print(pause_str)
    # print("Floorholder, g:", g_floorholder_list[2])
#     print('num of instances: {}'.format(list_sum))  
#     print('num Uncertain: {}'.format(uncertain_sum))
#     print('num of instances where only one speaker continued: {}'.format(turns_sum))
#     print('num holds: {}'.format(num_holds))
#     print('num shifts: {}'.format(num_shifts))
    #%% Write to file
    # structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
    dset_stats = file_hold_shift[pause_str+'/stats/num_holds'] = num_holds
    dset_stats = file_hold_shift[pause_str+'/stats/num_shifts'] = num_shifts
    dset_stats = file_hold_shift[pause_str+'/stats/num_instances_total'] = list_sum
    dset_stats = file_hold_shift[pause_str+'/stats/num_instances_one_speaker_continues'] = turns_sum
    
    for seq,indx in zip(test_file_list,range(len(test_file_list))):
        file_hold_shift.create_dataset(pause_str+'/hold_shift/'+seq+'/'+'g',data=np.array(g_hold_shift_list[indx]))
        file_hold_shift.create_dataset(pause_str+'/hold_shift/'+seq+'/'+'f',data=np.array(f_hold_shift_list[indx]))
    for seq,indx in zip(test_file_list,range(len(test_file_list))):
        file_floorholder.create_dataset(pause_str+'/floorholder/'+seq+'/'+'g',data=np.array(g_floorholder_list[indx]))
        file_floorholder.create_dataset(pause_str+'/floorholder/'+seq+'/'+'f',data=np.array(f_floorholder_list[indx]))

print(file_floorholder.keys())
file_hold_shift.close()
file_floorholder.close()
        
