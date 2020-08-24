import h5py

path = './data/datasets/'
file = 'hold_shift.hdf5'

full_path = path + file

d = h5py.File(full_path, 'r')

print()


