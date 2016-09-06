import numpy as np

if __name__ == '__main__':
    array1 = np.array(range(0, 3 * 3 * 3))
    array1.shape = (3, 3, 3)
    array2 = np.array(range(0, 10 * 4))
    array2.shape = (10, 4)
    t = array1[1]
    print(t.shape)
    print(array2[[1, 3, 5, 7], [0, 2, 0, 1]])
