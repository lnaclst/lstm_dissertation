# -*- coding: utf-8 -*-
import torch
import pickle
from torch.autograd import Variable
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.multiprocessing as mp
import numpy as np
# import asynch_lstm
# from datetime import datetime
#
# now = datetime.now()

use_cuda = torch.cuda.is_available()

if use_cuda:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True

# # Glove embeddings
# glove_embed_table =


# %% LSTM Class

# lstm axes: [sequence,minibatch,features]
class LSTMPredictor(nn.Module):

    def __init__(self, lstm_settings_dict, feature_size_dict={'acous': 0, 'visual': 0},
                 batch_size=32, seq_length=200, prediction_length=60, embedding_info=[],  speaker_embed = False, embedding_for_size = torch.Tensor(), use_embeds_init=False):
        super(LSTMPredictor, self).__init__()

        # General model settings
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.feature_size_dict = feature_size_dict                 # self.num_feat_for_lstm in data_loader_s.py
        self.prediction_length = prediction_length

        # lstm_settings_dict
        self.lstm_settings_dict = lstm_settings_dict
        self.feature_size_dict['master'] = 0
        if self.lstm_settings_dict['no_subnets']:
            for act_mod in self.lstm_settings_dict['active_modalities']:            # active_modalities gets created in the init function of TurnPredictionDataset (under <for feature_dict in self.feature_dict_list>)
                # from the modalities in the feature_vars dicts that are called  <- (icmi) <- feature-dict-list <- exp-features-lists <- feat-dicts <- (feature_vars.py) <- gemaps-dict-lists
                self.feature_size_dict['master'] += self.feature_size_dict[act_mod]
        else:
            for act_mod in self.lstm_settings_dict['active_modalities']:
                self.feature_size_dict['master'] += self.lstm_settings_dict['hidden_dims'][act_mod]

        if speaker_embed:
                self.feature_size_dict['master'] += embedding_for_size.size()[-1]  # !!!!!!!!

        self.num_layers = lstm_settings_dict['layers']

        print("Master: ", self.feature_size_dict['master'])

        # embedding settings
        self.embedding_info = embedding_info
        self.embeddings = {'acous': [], 'visual': []}
        self.embedding_indices = {'acous': [], 'visual': []}
        self.embed_delete_index_list = {'acous': [], 'visual': []}
        self.embed_data_types = {'acous': [], 'visual': []}
        self.len_output_of_embeddings = {'acous': 0, 'visual': 0}
        self.embedding_flags = {}

        # my embedding settings
        self.speaker_embed = speaker_embed
        self.use_embeds_init = use_embeds_init

        for modality in self.embedding_info.keys():
            self.embedding_flags[modality] = bool(len(self.embedding_info[modality]))
            if self.embedding_flags[modality]:
                for embedding in self.embedding_info[modality]:
                    self.len_output_of_embeddings[modality] += 2 * embedding['embedding_out_dim']
                for emb_func_indx in range(len(self.embedding_info[modality])):
                    if self.embedding_info[modality][emb_func_indx]['embedding_use_func']:
                        self.embeddings[modality].append( nn.Embedding(
                            self.embedding_info[modality][emb_func_indx]['embedding_num'],
                            self.embedding_info[modality][emb_func_indx]['embedding_out_dim']
                            ).type(dtype))
                        self.embedding_func = self.embeddings[modality][-1]
                        self.embed_data_types[modality].append(dtype_long)
                    elif self.embedding_info[modality][emb_func_indx]['use_glove']:
                        embed_tab_path = self.embedding_info[modality][emb_func_indx]['glove_embed_table']
                        glove_embed_table = pickle.load(open(embed_tab_path, 'rb'))
                        glove_embed_table[0] = np.random.normal(0,1e5,300) # need this to deal with BCE error
                        self.embeddings[modality].append( nn.Embedding.from_pretrained(
                            torch.FloatTensor(glove_embed_table).type(dtype),
                            freeze=self.lstm_settings_dict['freeze_glove']))
                        self.embedding_func = self.embeddings[modality][-1]
                        self.embed_data_types[modality].append(dtype_long)
                        print('using glove embeddings')
                    else:
                        self.embeddings[modality].append(
                            nn.Linear(self.embedding_info[modality][emb_func_indx]['embedding_num'],
                                      self.embedding_info[modality][emb_func_indx]['embedding_out_dim'],
                                      bias=True).type(dtype))
                        self.embedding_linear = self.embeddings[modality][-1]
                        self.embed_data_types[modality].append(dtype)
                    self.embedding_indices[modality].append(
                        self.embedding_info[modality][emb_func_indx]['emb_indices'])  # two tuples for start and end
                for emb_func_indx in range(len(self.embedding_info[modality])):
                    self.embed_delete_index_list[modality] += list(
                        range(self.embedding_indices[modality][emb_func_indx][0][0],
                              self.embedding_indices[modality][emb_func_indx][0][1]))
                    self.embed_delete_index_list[modality] += list(
                        range(self.embedding_indices[modality][emb_func_indx][1][0],
                              self.embedding_indices[modality][emb_func_indx][1][1]))

        # Initialize LSTMs
        self.lstm_dict= {}
        ### Embeddings LSTM ###
        # if embeddings == True and embeddings_lstm == True:
        #     self.lstm_dict['embedding'] = nn.LSTM()

        if self.lstm_settings_dict['no_subnets']:
            if not (len(self.lstm_settings_dict['active_modalities']) == 1):
                raise ValueError('Can only have one modality if no subnets')
            else:
                self.lstm_settings_dict['is_irregular']['master'] = self.lstm_settings_dict['is_irregular'][
                    self.lstm_settings_dict['active_modalities'][0]]
                if self.lstm_settings_dict['is_irregular']['master']:
                    # self.lstm_dict['master'] = nn.LSTMCell(self.feature_size_dict['master'],
                    #                                        self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
                    self.lstm_dict['master'] = nn.LSTMCell(self.feature_size_dict['master'],
                                                           self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
                    self.lstm_master = self.lstm_dict['master']
                else:
                    self.lstm_dict['master'] = nn.LSTM(self.feature_size_dict['master'],
                                                       self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
                    self.lstm_master = self.lstm_dict['master']
        else:
            self.lstm_settings_dict['is_irregular']['master'] = False                                       # Master LSTM is regular (meaning 50
            self.lstm_dict['master'] = nn.LSTM(self.feature_size_dict['master'],
                                               self.lstm_settings_dict['hidden_dims']['master']).type(dtype)       # Multi-layer LSTM with one layer per element of input
            # print('Dims1: ', self.feature_size_dict['master'])
            # print('Dims2: ', self.lstm_settings_dict['hidden_dims']['master'])
            self.lstm_master = self.lstm_dict['master']
            for lstm in self.lstm_settings_dict['active_modalities']:
                if self.lstm_settings_dict['is_irregular'][lstm]:
                    # self.lstm_dict[lstm] = nn.LSTMCell(self.feature_size_dict[lstm],
                    #                                    self.lstm_settings_dict['hidden_dims'][lstm]).type(dtype)
                    self.lstm_dict[lstm] = nn.LSTMCell(self.feature_size_dict[lstm],
                                                       self.lstm_settings_dict['hidden_dims'][lstm]).type(dtype)    # Single-layer LSTM applied to the input
                    if lstm == 'acous':
                        self.lstm_cell_acous = self.lstm_dict[lstm]
                    else:
                        self.lstm_cell_visual = self.lstm_dict[lstm]
                else:
                    self.lstm_dict[lstm] = nn.LSTM(self.feature_size_dict[lstm],
                                                   self.lstm_settings_dict['hidden_dims'][lstm]).type(dtype)
                    if lstm == 'acous':
                        self.lstm_acous = self.lstm_dict[lstm]
                    else:
                        self.lstm_visual = self.lstm_dict[lstm]

        # init dropout layers
        self.dropout_dict = {}
        for drop_key,drop_val in self.lstm_settings_dict['dropout'].items():
            self.dropout_dict[drop_key] = nn.Dropout(drop_val)
            setattr(self,'dropout_'+str(drop_key),self.dropout_dict[drop_key])


        ### Defined at end of forward() so shouldn't need this here
        # if speaker_embed:
        #     self.out = nn.Linear(2*self.lstm_settings_dict['hidden_dims']['master'], prediction_length).type(dtype)
        # else:
        self.out = nn.Linear(self.lstm_settings_dict['hidden_dims']['master'], prediction_length).type(dtype)

        self.embedding_transformer = nn.Linear(self.lstm_settings_dict['hidden_dims']['master'],20).type(dtype)

        self.init_hidden()

    def init_hidden(self):
        self.hidden_dict = {}

        if not self.use_embeds_init:
            for lstm in self.lstm_dict.keys():     # per layer?
                if self.lstm_settings_dict['is_irregular'][lstm]:        # ONLY ONE LAYER
                    self.hidden_dict[lstm] = (
                    Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype),
                    Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
                else:
                    self.hidden_dict[lstm] = (Variable(
                        torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(
                        dtype), Variable(torch.zeros(
                            self.num_layers, self.batch_size,self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
        else:
            #### Code copied from just above, replace with code to load embeddings as starting hidden layers
            for lstm in self.lstm_dict.keys():     # per layer?
                if self.lstm_settings_dict['is_irregular'][lstm]:        # ONLY ONE LAYER
                    self.hidden_dict[lstm] = (
                    Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype),
                    Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
                else:
                    self.hidden_dict[lstm] = (Variable(
                        torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(
                        dtype), Variable(torch.zeros(
                            self.num_layers, self.batch_size,self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
            # print(self.hidden_dict[lstm][0].size())

    def change_batch_size_reset_states(self, batch_size):
        self.batch_size = int(batch_size)
        for lstm in self.lstm_dict.keys():
            if self.lstm_settings_dict['is_irregular'][lstm]:
                self.hidden_dict[lstm] = (
                Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype),
                Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
            else:
                self.hidden_dict[lstm] = (Variable(
                    torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(
                    dtype), Variable(torch.zeros(
                        self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))

    def change_batch_size_no_reset(self, new_batch_size):
        for lstm in self.lstm_dict.keys():
            if self.lstm_settings_dict['is_irregular'][lstm]:
                self.hidden_dict[lstm] = (
                Variable(self.hidden_dict[lstm][0][:new_batch_size, :].data.contiguous().type(dtype)),
                Variable(self.hidden_dict[lstm][1][:new_batch_size, :].data.contiguous().type(dtype)))
            else:
                self.hidden_dict[lstm] = (
                Variable(self.hidden_dict[lstm][0][:, :new_batch_size, :].data.contiguous().type(dtype)),
                Variable(self.hidden_dict[lstm][1][:, :new_batch_size, :].data.contiguous().type(dtype)))
        self.batch_size = new_batch_size

    def weights_init(self, init_std):
        # init bias to zero recommended in http://proceedings.mlr.press/v37/jozefowicz15.pdf
        nn.init.normal(self.out.weight.data, 0, init_std)
        nn.init.constant(self.out.bias, 0)

        ##### Init embedding transformer here
        nn.init.normal(self.embedding_transformer.weight, 0, init_std)
        nn.init.constant(self.embedding_transformer.bias, 0)

        for lstm in self.lstm_dict.keys():                     # Assign values to the keys
            if self.lstm_settings_dict['is_irregular'][lstm]:
                nn.init.normal(self.lstm_dict[lstm].weight_hh, 0, init_std)
                nn.init.normal(self.lstm_dict[lstm].weight_ih, 0, init_std)
                nn.init.constant(self.lstm_dict[lstm].bias_hh, 0)
                nn.init.constant(self.lstm_dict[lstm].bias_ih, 0)

            else:
                nn.init.normal(self.lstm_dict[lstm].weight_hh_l0, 0, init_std)
                nn.init.normal(self.lstm_dict[lstm].weight_ih_l0, 0, init_std)
                nn.init.constant(self.lstm_dict[lstm].bias_hh_l0, 0)
                nn.init.constant(self.lstm_dict[lstm].bias_ih_l0, 0)

    def embedding_helper(self, in_data, modality):
        embeds_one = []
        embeds_two = []

        for emb_func_indx in range(len(self.embeddings[modality])):
            debug1 = self.embeddings[modality][emb_func_indx](
                Variable(in_data[:, :, self.embedding_indices[modality][emb_func_indx][0][0]:
                                       self.embedding_indices[modality][emb_func_indx][0][1]] \
                         .data.type(self.embed_data_types[modality][emb_func_indx])))
            embeds_one_tmp = self.embeddings[modality][emb_func_indx](
                Variable(in_data[:, :, self.embedding_indices[modality][emb_func_indx][0][0]:
                                       self.embedding_indices[modality][emb_func_indx][0][1]] \
                         .data.type(self.embed_data_types[modality][emb_func_indx]).squeeze(dim=2)))
            embeds_two_tmp = self.embeddings[modality][emb_func_indx](
                Variable(in_data[:, :, self.embedding_indices[modality][emb_func_indx][1][0]:
                                       self.embedding_indices[modality][emb_func_indx][1][1]] \
                         .data.type(self.embed_data_types[modality][emb_func_indx]).squeeze(dim=2)))

            if not (self.lstm_settings_dict['uses_master_time_rate'][modality]) and self.lstm_settings_dict['is_irregular'][modality]:
                embeds_one_tmp = embeds_one_tmp.transpose(2,3)
                embeds_two_tmp = embeds_two_tmp.transpose(2,3)

            embeds_one.append(embeds_one_tmp)
            embeds_two.append(embeds_two_tmp)

        non_embeddings = list(set(list(range(in_data.shape[2]))).difference(
            set(self.embed_delete_index_list[modality])))  # !!! is shape[2] correct?
        
        if len(non_embeddings) != 0:
            in_data = in_data[:, :, non_embeddings]
            for emb_one, emb_two in zip(embeds_one, embeds_two):
                in_data = torch.cat((in_data, emb_one), 2)
                in_data = torch.cat((in_data, emb_two), 2)
        else:
            for emb_one, emb_two in zip(embeds_one, embeds_two):
                in_data = torch.cat((in_data, emb_one), 2)
                in_data = torch.cat((in_data, emb_two), 2)
            embed_keep = list(set(list(range(in_data.shape[2]))).difference(set(self.embed_delete_index_list[modality])))
            in_data = in_data[:, :, embed_keep]
        return in_data

    def forward(self, in_data, embedding=None):          # Is this a single forward step, or is this all of the forward steps?
        if embedding != None:
            print("Embedding loaded.")

        print("Batch size: ", self.batch_size)
        x, i, h = {}, {}, {}
        h_list = []
        hidden_list = []
        x['acous'], i['acous'], x['visual'], i['visual'] = in_data

        ##### WHAT HAPPENS IF I CONCATENATE THE EMBEDDINGS TO THE MASTER LSTM HERE?
        # print('Hidden dict:', self.hidden_dict['master'][0].size())
        # if self.speaker_embed:
        #     hidden_dict_master_temp = torch.cat([self.hidden_dict['master'][0],embedding[:1]],2)
        #     cell_temp = self.hidden_dict['master'][1]
        #     self.hidden_dict['master'] = (hidden_dict_master_temp,cell_temp)

        if not (self.lstm_settings_dict['no_subnets']):
            # If saving embedding
            mod_list = []
            for mod in self.lstm_settings_dict['active_modalities']:
                if self.embedding_flags[mod]:
                    x[mod] = self.embedding_helper(x[mod], mod)

                # Apply dropout input layers
                x[mod] = self.dropout_dict[mod + '_in'](x[mod])

                # print("Test", self.hidden_dict[mod][0].size())

                cell_out_list = []
                if not(self.lstm_settings_dict['is_irregular'][mod]) and self.lstm_settings_dict['uses_master_time_rate'][mod]:  # not is_irregular: Not asynch, ling&acous = 50ms; uses_master_time_rate: 50ms
                    h[mod], self.hidden_dict[mod] = self.lstm_dict[mod](x[mod], self.hidden_dict[mod])                           # call the values of lstm_dict
                    print('1a')

                elif not (self.lstm_settings_dict['is_irregular'][mod]) and not(self.lstm_settings_dict['uses_master_time_rate'][mod]):  # not is_irregular: Not asynch, ling&acous = 10ms; not uses_master_time_rate: 10ms
                    h_acous_temp, self.hidden_dict[mod] = self.lstm_dict[mod](x[mod], self.hidden_dict[mod])
                    # h[mod] = h_acous_temp[0::self.lstm_settings_dict['time_step_size'][mod]]
                    h[mod] = h_acous_temp[self.lstm_settings_dict['time_step_size'][mod]-1::self.lstm_settings_dict['time_step_size'][mod]]
                    print('2b')
                # Think it's using this one vvv

                elif self.lstm_settings_dict['is_irregular'][mod] and self.lstm_settings_dict['uses_master_time_rate'][mod]:  # is_irregular: asynch; uses_master_time_rate: acous = 50ms
                    print('3c')
                    for seq_indx in range(self.seq_length):
                        changed_indices = np.where(i[mod][seq_indx])[0].tolist()
                        if len(changed_indices) > 0:
                            h_l, c_l = self.lstm_dict[mod](x[mod][seq_indx][changed_indices], (self.hidden_dict[mod][0][changed_indices], self.hidden_dict[mod][1][changed_indices]))
                            h_l_copy = self.hidden_dict[mod][0].clone()
                            c_l_copy = self.hidden_dict[mod][1].clone()
                            h_l_copy[changed_indices] = h_l
                            c_l_copy[changed_indices] = c_l
                            self.hidden_dict[mod] = (h_l_copy,c_l_copy)
                            # self.hidden_dict[mod][0][changed_indices] = h_l
                            # self.hidden_dict[mod][1][changed_indices] = c_l
                        cell_out_list.append(self.hidden_dict[mod][0])
                    h[mod] = torch.stack(cell_out_list)

                elif bool(self.lstm_settings_dict['is_irregular'][mod]) and not (
                self.lstm_settings_dict['uses_master_time_rate'][mod]):  # is_irregular: asynch; not uses_master_time_rate: 10ms   # for ling and visual data
                    print('4d')
                    for seq_indx in range(self.seq_length):
                        for step_indx in range(x[mod].shape[-1]):
                            changed_indices = np.where(i[mod][seq_indx, :, step_indx])[0].tolist()
                            if len(changed_indices) > 0:
                                h_l, c_l = self.lstm_dict[mod](x[mod][seq_indx][:][changed_indices][:, :, step_indx], (self.hidden_dict[mod][0][changed_indices], self.hidden_dict[mod][1][changed_indices]))
                                h_l_copy = self.hidden_dict[mod][0].clone()
                                c_l_copy = self.hidden_dict[mod][1].clone()
                                h_l_copy[changed_indices] = h_l
                                c_l_copy[changed_indices] = c_l
                                self.hidden_dict[mod] = (h_l_copy, c_l_copy)
                                # self.hidden_dict[mod][0][changed_indices] = h_l
                                # self.hidden_dict[mod][1][changed_indices] = c_l
                        cell_out_list.append(self.hidden_dict[mod][0])         # Do NOT append embeddings to this. This is because there are multiple subnet cells between the master cells and this accumulates them
                    h[mod] = torch.stack(cell_out_list)
                else:
                    raise ValueError('problem in forward pass')

                # apply dropout
                h[mod] = self.dropout_dict[str(mod)+'_out'](h[mod])                # h[mod] contains all of the hidden states for the given modality

                h_list.append(h[mod])

                print("H", h[mod].size())

                hidden_list.append(self.hidden_dict[mod][0])
                print("Hidden: ", hidden_list[-1].size())

                # self.mod_embeddings.append(self.hidden_dict[mod][0])
                mod_list.append(mod)
                self.mod_list = mod_list

            self.mod_embeddings = hidden_list

            print("Num mod embeddings: ", len(self.mod_embeddings))

            if self.speaker_embed:
                try:
                    if embedding.size()[1] != h_list[0].size()[1]:
                        embedding = embedding.transpose(0,1)
                        # embedding = embedding[:h_list[0].size()[1]]    # Implemented this in run_json
                        embedding = embedding.transpose(0,1)
                except AttributeError:
                    pass
                h_list.append(embedding)
                cat = torch.cat(h_list, 2)
            else:
                cat = torch.cat(h_list, 2)
                print("No embedding in model.")

            ############################

            print('Hidden dict pre-lstm master:', self.hidden_dict['master'][0].size(),self.hidden_dict['master'][1].size())

            lstm_out, self.hidden_dict['master'] = self.lstm_dict['master'](cat,self.hidden_dict['master'])        # Puts the outputs of the sub-LSTM through the master LSTM, h_list being the hidden states from each of the modalities and hidden_dict['master'] being the hidden state from the previous master LSTM

            # print(embedding.size())
            print("Hidden dict transformed: ", self.hidden_dict['master'][0].size())
            #### Try attaching the embededing to self.hidden_dict['master'][0] here ^^^ (without having done the 600 loop in run_json.py

            lstm_out = self.dropout_dict['master_out'](lstm_out)
            print("LSTM_out: ", lstm_out.size())


        else:  # For no subnets...
            print("No subnets")
            if not (len(self.lstm_settings_dict['active_modalities']) == 1):
                raise ValueError('need to have only one modality when there are no subnets')

            mod = self.lstm_settings_dict['active_modalities'][0]
            if self.embedding_flags[mod]:
                x[mod] = self.embedding_helper(x[mod], mod)

            x[mod] = self.dropout_dict['master_in'](x[mod])
            if self.speaker_embed:
                x[mod] = [x[mod],embedding]
                x[mod] = torch.cat(x[mod],2)

            cell_out_list = []
            # get outputs of lstm['acous']
            # print("LSTM, ", lstm_out.size())
            # print("Hidden, ", self.hidden_dict['master'][0].size(), self.hidden_dict['master'][1].size())
            if not (self.lstm_settings_dict['is_irregular'][mod]) and \
                    self.lstm_settings_dict['uses_master_time_rate'][mod]:
                lstm_out, self.hidden_dict['master'] = self.lstm_dict['master'](x[mod], self.hidden_dict['master'])        # self.hidden_dict['master'] on the right is a tuple of the input hidden and cell states; on the right is the output hidden and cell states

            elif not (self.lstm_settings_dict['is_irregular'][mod]) and not (self.lstm_settings_dict['uses_master_time_rate'][mod]):
                h_acous_temp, self.hidden_dict['master'] = self.lstm_dict['master'](x[mod],self.hidden_dict['master'])
                # lstm_out = h_acous_temp[0::self.lstm_settings_dict['time_step_size'][mod]] # <- example of bad index
                lstm_out = h_acous_temp[self.lstm_settings_dict['time_step_size'][mod]-1::self.lstm_settings_dict['time_step_size'][mod]] # <-correct indexing

            elif self.lstm_settings_dict['is_irregular'][mod] and self.lstm_settings_dict['uses_master_time_rate'][mod]:
                for seq_indx in range(self.seq_length):
                    changed_indices = np.where(i[mod][seq_indx])[0].tolist()
                    if len(changed_indices) > 0:
                        h_l, c_l = self.lstm_dict['master'](
                                                            x[mod][seq_indx][changed_indices],
                                                            (
                                                                self.hidden_dict['master'][0][changed_indices],
                                                                self.hidden_dict['master'][1][changed_indices]
                                                            )
                                                            )
                        h_l_copy = self.hidden_dict['master'][0].clone()
                        c_l_copy = self.hidden_dict['master'][1].clone()
                        h_l_copy[changed_indices] = h_l
                        c_l_copy[changed_indices] = c_l
                        self.hidden_dict['master'] = (h_l_copy,c_l_copy)
                        # self.hidden_dict['master'][0][changed_indices] = h_l
                        # self.hidden_dict['master'][1][changed_indices] = c_l
                    cell_out_list.append(self.hidden_dict['master'][0])
                lstm_out = torch.stack(cell_out_list)

            elif bool(self.lstm_settings_dict['is_irregular'][mod]) and not (self.lstm_settings_dict['uses_master_time_rate'][mod]):  # for ling and visual data
                for seq_indx in range(self.seq_length):
                    for step_indx in range(x[mod].shape[-1]):
                        changed_indices = np.where(i[mod][seq_indx])[0].tolist()
                        if len(changed_indices) > 0:
                            h_l, c_l = self.lstm_dict['master'](
                                                                x[mod][seq_indx][:][changed_indices][:, :, step_indx],
                                                                (
                                                                    self.hidden_dict['master'][0][changed_indices],
                                                                    self.hidden_dict['master'][1][changed_indices]
                                                                )
                                                                )
                            h_l_copy = self.hidden_dict['master'][0].clone()
                            c_l_copy = self.hidden_dict['master'][1].clone()
                            h_l_copy[changed_indices] = h_l
                            c_l_copy[changed_indices] = c_l
                            self.hidden_dict['master'] = (h_l_copy,c_l_copy)        # hidden states/cell states; need to get the right index to call the hidden state?
                            # self.hidden_dict['master'][0][changed_indices] = h_l
                            # self.hidden_dict['master'][1][changed_indices] = c_l
                    cell_out_list.append(self.hidden_dict['master'][0])
                lstm_out = torch.stack(cell_out_list)                               # do I want lstm_out or do I want self.hidden_dict['master'][0]???
            else:
                raise ValueError('problem in forward pass')

            # hidden_return = lstm_out[-1]

            lstm_out = self.dropout_dict[str(mod)+'_out'](lstm_out)

        ## Concatenating embeddings

        # sigmoid_out = F.sigmoid(self.out(lstm_out))
        #Take stacked, dropout-ed hidden layers and run them through a linear layer

        if not self.speaker_embed:
            self.embedding = self.hidden_dict['master'][0]
            # self.embedding = lstm_out
            # self.embedding = self.embedding_transformer(self.hidden_dict['master'][0])

            # Add a step that trains the embedding transformer: need to change the size expected by self.out, then
            # perform self.out on self.embedding to get the prediction

            print("Embedding size: ", self.embedding.size())

        sigmoid_out = self.out(lstm_out)

        return sigmoid_out











### All of this used to be between Concatenate embeddings and sigmoid_out, but I wasn't using it and was annoying me

# if self.speaker_embed and type(lstm_out) == torch.Tensor:
        #     try:
        #         lstm_out = torch.cat((lstm_out, embedding),2)  # Treats each embedding as extra features ........ # a frame in tbptt; could also concatenate the lists created above along the 2nd dimension (1) and input the embeddings at each frame
        #     except TypeError:
        #         self.speaker_embed = False

        # if self.speaker_embed:
        #     self.out = nn.Linear(2 * self.lstm_settings_dict['hidden_dims']['master'], self.prediction_length).type(
        #         dtype)
        #     print("Embeddings attached to lstm_out.")
        # else:
        #     self.out = nn.Linear(self.lstm_settings_dict['hidden_dims']['master'], self.prediction_length).type(dtype)
        #
        #     print("Output size with embedding: ", lstm_out.size())