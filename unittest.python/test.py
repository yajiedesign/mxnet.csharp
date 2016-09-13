import numpy as np

if __name__ == '__main__':
    array1 = np.array(range(0, 3 * 3 * 3))
    array1.shape = (3, 3, 3)
    array2 = np.array(range(0, 5 * 4 * 3 * 2))
    array2.shape = (5, 4, 3, 2)
    t = array1.mean()
    print(t.T.shape)
    print(t)
    data_tmp = np.ones((3, 4))
    data_tmp[:] = 0.5
    print(data_tmp * 1 / (1 - data_tmp ** 2) ** (1 / 2) * 2)
