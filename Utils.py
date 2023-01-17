
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from models.regularization import EarlyStoppingCheckPoint
from models.cnn import ConvolutionLayer, ReluActivationLayer, SimpleCNN,BuildCnnModels



def benchmark_test(X_train_left, X_train_right, Y_train, X_test_left, X_test_right, Y_test):
    # convert labels from [-1, 1] to [0, 1]
    Y_test = (Y_test + 1) / 2
    Y_train = (Y_train + 1) / 2

    enc = OneHotEncoder()
    Y_test = enc.fit_transform(Y_test).toarray()
    Y_train = enc.fit_transform(Y_train).toarray()

    print("Y_test shape", Y_test.shape)
    print("Y_test", type(Y_test))
    # print("Y_test", Y_test)

    tf.reset_default_graph()

    conv_layer_1_for_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_2_A = ReluActivationLayer()
    conv_layer_3_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_4_A = ReluActivationLayer()

    simpleCNN = SimpleCNN(1)
    simpleCNN.add_layer(conv_layer_1_for_A)
    simpleCNN.add_layer(activation_layer_2_A)
    simpleCNN.add_layer(conv_layer_3_A)
    simpleCNN.add_layer(activation_layer_4_A)
    simpleCNN.build(input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=0.001)

    show_fig = True
    batch_size = 256
    N, D = Ytrain_b.shape
    residual = N % batch_size
    if residual == 0:
        n_batches = N // batch_size
    else:
        n_batches = N // batch_size + 1

    epochs = 5
    earlyStoppingCheckPoint = EarlyStoppingCheckPoint("acc", 100)
    earlyStoppingCheckPoint.set_model(simpleCNN)

    # merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        simpleCNN.set_session(sess)

        sess.run(init)
        loss_list = []
        acc_list = []
        global_step = 0

        earlyStoppingCheckPoint.on_train_begin()
        for ep in range(epochs):
            for i in range(n_batches):
                global_step += 1
                # X_left = X_train_left[i * batch_size: i * batch_size + batch_size]
                X = X_train_right[i * batch_size: i * batch_size + batch_size]
                Y = Y_train[i * batch_size: i * batch_size + batch_size]
                loss, summary = simpleCNN.train(X, Y)

                if i % 1 == 0:
                    loss_list.append(loss)
                    # y_preds = federatedLearning.predict(X_test_b_left, X_test_b_right)
                    acc, summary = simpleCNN.evaluate(X_test_right, Y_test)
                    acc_list.append(acc)
                    print(ep, "batch", i, "loss:", loss, "acc", acc)

                    metrics = {"acc": acc}
                    earlyStoppingCheckPoint.on_iteration_end(ep, i, metrics)

                if simpleCNN.is_stop_training():
                    break

            if simpleCNN.is_stop_training():
                break

        if show_fig:
            plt.subplot(121)
            plt.plot(loss_list)
            plt.xlabel("loss")
            plt.subplot(122)
            plt.plot(acc_list)
            plt.xlabel("acc")
            plt.show()

        print("loss_list:", loss_list)
        print("acc_list:", acc_list)


def getKaggleMINST(data_dir,data_shuffle=True):
    # MNIST datasets:
    # column 0 is labels
    # column 1-785 is datasets, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv(data_dir + '/MNIST/train.csv')
    # test = pd.read_csv('../data/MINST/test.csv')
    train = train.to_numpy()
    print("[INFO] train data shape:{0}".format(train.shape))
    if data_shuffle: train = shuffle(train)

    Xtrain = train[:-7500, 1:] / 255
    Ytrain = train[:-7500, 0].astype(np.int32)
    Xtest = train[-7500:, 1:] / 255
    Ytest = train[-7500:, 0].astype(np.int32)

    return Xtrain, Ytrain, Xtest, Ytest

def get_binary_labels(X, Y, binary=(9, 8)):
    """
    Convert two specified labels to 0 and 1
    :param X:
    :param Y:
    :param binary:
    :return:
    """
    X_b = []
    Y_b = []
    for index in range(X.shape[0]):
        lbl = Y[index]
        if lbl in binary:
            X_b.append(X[index])
            Y_b.append(0 if lbl == binary[0] else 1)
    return np.array(X_b), np.array(Y_b)


def split_in_half(imgs):
    """
    split input images in half vertically
    :param imgs: input images
    :return: left part of images and right part of images
    """
    left, right = [], []
    for index in range(len(imgs)):
        img = imgs[index]
        left.append(img[:, 0:14, :])
        right.append(img[:, 14:, :])
    return np.array(left), np.array(right)

def split_in_N(imgs,n=2,fullsize=28):
    """
    split input images in half vertically
    :param imgs: input images
    :n is the number of part in which want to devide
    :return: list of part of images
    """
    assert n in (2,4)," Division not possible"
    if n==2:
        return split_in_half(imgs)
    # incr=fullsize//n
    # img_list=[imgs[:,:,i*incr:(i+1)*incr,:] for i in range(0,imgs.shape[2],incr)]
    incr=fullsize//(n//2)
    img_list=[imgs[:,i:i+incr,j:j+incr,:] for i in range(0,imgs.shape[1],incr) for j in range(0,imgs.shape[2],incr)]
    return img_list


def getData(data_shuffle=True,numberofParties=2,shapvalues=None):
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMINST("data",data_shuffle=data_shuffle)
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)

    # choose two labels from the 10 digit labels and convert the two labels to 0 and 1
    Xtrain_b, Ytrain_b = get_binary_labels(Xtrain, Ytrain, [3, 8])
    Xtest_b, Ytest_b = get_binary_labels(Xtest, Ytest, [3, 8])
    if shapvalues is not None:
        Xtrain_b=makeImageOFF(Xtrain_b,shapvalues)

    Ytrain_b = Ytrain_b.reshape(Ytrain_b.shape[0], 1)
    Ytest_b = Ytest_b.reshape(Ytest_b.shape[0], 1)
    if numberofParties==2:
        # split each image in half to simulate two-party vertical federated learning
        X_train_b_left, X_train_b_right = split_in_half(Xtrain_b)
        X_test_b_left, X_test_b_right = split_in_half(Xtest_b)
        return [X_train_b_left, X_train_b_right],Ytrain_b,[X_test_b_left, X_test_b_right],Ytest_b
    if numberofParties==4:
        # split each image in half to simulate two-party vertical federated learning
        train_list_b= split_in_N(Xtrain_b,4)
        test_list_b = split_in_N(Xtest_b,4)
        return train_list_b,Ytrain_b,test_list_b,Ytest_b



def makeImageOFF(X,shaplyvalues,elementperblock=7):
    if np.all(shaplyvalues):
        return X
    for i in range(len(shaplyvalues)):
        for j in range(len(shaplyvalues[i])):
            if not shaplyvalues[i][j]:
                X[:, i * elementperblock:(i + 1) * elementperblock,j * elementperblock:(j + 1) * elementperblock]=np.zeros((X.shape[0],elementperblock,elementperblock,1))
    return X

def compute_accuracy(y_targets, y_preds):
    corr_count = 0
    total_count = len(y_preds)
    for y_p, y_t in zip(y_preds, y_targets):
        if (y_p <= 0.5 and y_t == -1) or (y_p > 0.5 and y_t == 1):
            corr_count += 1

    acc = float(corr_count) / float(total_count)
    return acc

