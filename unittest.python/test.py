import numpy as np

if __name__ == '__main__':
    array = np.array(range(0, 10 * 4 * 5 * 5))
    array.shape = (10, 4, 5, 5)
    t = array[-1:-1:3,:1:-1]
    print(t)
