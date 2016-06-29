import numpy as np


def dump_nparray(array, filename):
    '''

    Saves numpy array in binary format in the specified file

    Parameters
    ----------
    array : numpy array
        Array to be saved

    filename : str
        Output file name


    '''
    array_file = open(filename, 'wb')
    np.uint32(array.ndim).tofile(array_file)
    for d in range(array.ndim):
        np.uint32(array.shape[d]).tofile(array_file)
    array.tofile(array_file)
    array_file.close()


def load_nparray_from_bin_file(filename, array_dtype):
    '''

    Loads numpy array saved with dump_nparray

    Parameters
    ----------
    filename : str
        Path to the binary file containing the desired array

    array_dtype : dtype
        Type of array elements
        (used to decide how many bytes will be read from the file)

    Returns
    -------
    array: numpy array
        Retrieved array


    '''
    with open(filename, 'rb') as array_file:
        ndim = np.fromfile(array_file, dtype=np.uint32, count=1)[0]
        shape = []
        for d in range(ndim):
            shape.append(np.fromfile(array_file, dtype=np.uint32, count=1)[0])
        array_data = np.fromfile(array_file, dtype=array_dtype)
        return np.reshape(array_data, shape)
