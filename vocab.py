from os import walk
import pandas as pd
import pickle

"""
Script creates 3 dictionaries: conv_dict, which contains word count by conversation, total_dict, which contains
total word count in the data set, and conv_vecs, which contains vectors with the counts of each word by index in each
conversation. Pickles all 3 dictionaries.
Important note: If using the averaged annotations, some words will come out with high frequency, even if they are not
frequent, because their indices in ix_to_word correspond to common averages. "Jaggedy" is one such example. If you want 
to be able to use the actual word counts, you need to use *raw*. 
"""

# Total vocab count
# Vocab count for role
# Vocab count by conversation by role

base_path = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/data/extracted_annotations/'

words_folder = base_path + 'words_advanced_50ms_raw/'

ix_to_word_file = open(base_path + 'ix_to_word.p','rb')
ix_to_word = pickle.load(ix_to_word_file)

print(len(ix_to_word))

conv_vecs = {}
for word in ix_to_word.values():
    conv_vecs[word] = 0

conv_dict = {}
total_dict = {}
f_wordcount = 0
g_wordcount = 0
f_words = {}
g_words = {}
for (dirpath, dirnames, filenames) in walk(words_folder):
    for filename in filenames:
        if '.csv' in filename:
            if filename not in conv_dict:
                conv_dict[filename] = {}
            csv = pd.read_csv(words_folder + filename, usecols = ['0'])

            if filename not in conv_vecs:
                conv_vecs[filename] = [0]*len(ix_to_word)

            for x in csv['0']:
                idx = x - 1
                if x != 0:
                    conv_vecs[filename][int(idx)] += 1

                    if '.f.' in filename:
                        f_wordcount += 1
                        if ix_to_word[x] not in f_words:
                            f_words[ix_to_word[x]] = 0
                        else:
                            f_words[ix_to_word[x]] += 1
                    if '.g.' in filename:
                        g_wordcount += 1
                        if ix_to_word[x] not in g_words:
                            g_words[ix_to_word[x]] = 0
                        else:
                            g_words[ix_to_word[x]] += 1

                    if ix_to_word[x] == 'jaggedy':
                        print(filename)

                    if ix_to_word[x] not in total_dict:
                        total_dict[ix_to_word[x]] = 1
                    else:
                        total_dict[ix_to_word[x]] += 1
                    if ix_to_word[x] not in conv_dict[filename]:
                        conv_dict[filename][ix_to_word[x]] = 1
                    else:
                        conv_dict[filename][ix_to_word[x]] += 1

# print(total_dict)
print(f_wordcount)
# print(g_wordcount)
print(f_words)
print(g_words)

# print(conv_vecs)

# pickle.dump(conv_vecs, open(base_path + 'conv_vectors_raw.p','wb'))
# pickle.dump(conv_dict, open(base_path + 'conv_count_dict_raw.p','wb'))
# pickle.dump(total_dict, open(base_path + 'total_count_dict_raw.p','wb'))

# print(conv_dict)
# print(total_dict)