import numpy as np
import pickle
import os
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        ret = pickle.load(fo, encoding='bytes')
    return ret

if __name__ == '__main__':
    train_imgs, train_labels = [], []
    for i in range(1, 6):
        path = 'cifar-10-batches-py/data_batch_'+str(i)
        d = unpickle(path)
        train_imgs.extend(d[b'data'])
        train_labels.extend(d[b'labels'])
    train_imgs = np.array(train_imgs).reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    train_labels = np.array(train_labels)
    print(train_imgs.shape, train_imgs.dtype)
    print(train_labels.shape, train_labels.dtype)
    path = 'cifar-10-batches-py/test_batch'
    d = unpickle(path)
    test_imgs, test_labels = np.array(d[b'data']).reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1]), np.array(d[b'labels'])
    print(test_imgs.shape, test_imgs.dtype)
    print(test_labels.shape, test_labels.dtype)
    
    if not os.path.exists('train'):
        os.makedirs('train')
    np.save('train/imgs.npy', train_imgs)
    np.save('train/labels.npy', train_labels)
    if not os.path.exists('test'):
        os.makedirs('test')
    np.save('test/imgs.npy', test_imgs)
    np.save('test/labels.npy', test_labels)
    print(train_imgs.mean(axis=(0,1,2))/255)
    cv2.imwrite('example.png', train_imgs[0])
