from copy import deepcopy


# Need: test_file_list,


true_vals = list()
    predicted_class = list()
    for conv_key in test_file_list:        # conv_key = conversation
        for g_f_key in list(floorholder[pause_str + '/floorholder' + '/' + conv_key].keys()):      # 'f' and 'g' distinguish datasets from each other (datasets created in find_pauses.py)
            g_f_key_not = deepcopy(data_select_dict[data_set_select])
            g_f_key_not.remove(g_f_key)
            for frame_indx, true_val in floorholder[pause_str + '/floorholder' + '/' + conv_key + '/' + g_f_key]:
                # make sure the index is not out of bounds
                # print(frame_indx, true_val)
                if frame_indx < len(results_dict[conv_key + '/' + g_f_key]):
                    #Lena exp. 1 vvv
                    #Not looking ahead or behind for evaluation. Just looking at the floorholder prediction for the current frame.
    #                         if frame_idx >= length_of_future_window:
                    true_vals.append(true_val)
                    # Lena exp. 1 vvv PROBLEM AREA
                    # Shouldn't need to sum this because I'm only creating a binary prediction. Sum is
                    # because Roddy's predictions are 60 items long and we need to determine which is the chosen class
                    # Need a different results dict with size
                    if np.sum(
                            results_dict[conv_key + '/' + g_f_key][frame_indx, 0]) > np.sum(
                            results_dict[conv_key + '/' + g_f_key_not[0]][frame_indx, 0]):         #compares the predictions for f and g over the whole past window
                        predicted_class.append(0)      # 0 == g is floorholder
                    else:
                         predicted_class.append(1)      # 1 == f is floorholder
    f_score = f1_score(true_vals, predicted_class, average='weighted')
    results_save['f_scores_' + pause_str].append(f_score)
    # set up confusion matrix to get f-scores
    tn, fp, fn, tp = confusion_matrix(true_vals, predicted_class).ravel()
    results_save['tn_' + pause_str].append(tn)
    results_save['fp_' + pause_str].append(fp)
    results_save['fn_' + pause_str].append(fn)
    results_save['tp_' + pause_str].append(tp)
    print('majority vote f-score(' + pause_str + '):' + str(
        f1_score(true_vals, np.zeros([len(predicted_class)]).tolist(), average='weighted')))
