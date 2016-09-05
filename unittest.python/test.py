import numpy as np

if __name__ == '__main__':
    array1 = np.array(range(0, 10 * 4 * 5 * 5))
    array1.shape = (10, 5, 4, 5)
    array2 = np.array(range(0, 10 * 4 * 5 * 5))
    array2.shape = (10, 4, 5, 5)

    print(np.argmax(array2, axis=3))
    print(np.argmax(array2, axis=3).shape)