import numpy as np

if __name__ == '__main__':
    array1 = np.array(range(0, 3 * 3 * 3))
    array1.shape = (3, 3, 3)
    array2 = np.array(range(0, 10 * 4 * 5 * 5))
    array2.shape = (10, 4, 5, 5)
    t = array1.dot(array1)
    print(t.shape)
    print( t)
