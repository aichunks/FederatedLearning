#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.chdir("FedBCD-main")

import sys
import shap
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from config import data_dir
from models.cnn import ConvolutionLayer, ReluActivationLayer, SimpleCNN
from models.learning_rate_decay import sqrt_learning_rate_decay
from models.regularization import EarlyStoppingCheckPoint
from store_utils import save_experimental_results
from vnn_demo.vfl import VFLGuestModel, VFLHostModel, VerticalMultiplePartyFederatedLearning
from vnn_demo.vfl_learner import VerticalFederatedLearningLearner


# In[2]:


def getKaggleMINST(data_dir):
    # MNIST datasets:
    # column 0 is labels
    # column 1-785 is datasets, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv(data_dir + '/MNIST/train.csv')
    # test = pd.read_csv('../data/MINST/test.csv')
    train = train.to_numpy()
    print("[INFO] train data shape:{0}".format(train.shape))
    train = shuffle(train)

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


def compute_accuracy(y_targets, y_preds):
    corr_count = 0
    total_count = len(y_preds)
    for y_p, y_t in zip(y_preds, y_targets):
        if (y_p <= 0.5 and y_t == -1) or (y_p > 0.5 and y_t == 1):
            corr_count += 1

    acc = float(corr_count) / float(total_count)
    return acc



def run_experiment(train_data, test_data, output_directory_name, n_local, batch_size, learning_rate, is_parallel,
                   epochs=5, apply_proximal=False, proximal_lbda=0.1, is_debug=False, verbose=False, show_fig=True):
    
    
    print("hyper-parameters:")
    print("# of epochs: {0}".format(epochs))
    print("# of local iterations: {0}".format(n_local))
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))
    print("is async {0}".format(is_parallel))
    print("is apply_proximal {0}".format(apply_proximal))
    print("proximal_lbda {0}".format(proximal_lbda))
    print("show figure: {0}".format(show_fig))
    print("is debug: {0}".format(is_debug))
    print("verbose: {0}".format(verbose))

    X_train_b_left, X_train_b_right, Ytrain_b = train_data
    X_test_b_left, X_test_b_right, Ytest_b = test_data

    tf.reset_default_graph()

    conv_layer_1_for_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_2_A = ReluActivationLayer()
    conv_layer_3_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_4_A = ReluActivationLayer()

    simpleCNN_A = SimpleCNN(1)
    simpleCNN_A.add_layer(conv_layer_1_for_A)
    simpleCNN_A.add_layer(activation_layer_2_A)
    simpleCNN_A.add_layer(conv_layer_3_A)
    simpleCNN_A.add_layer(activation_layer_4_A)
    simpleCNN_A.build(input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,
                      proximal_lbda=proximal_lbda)

    conv_layer_1_for_B = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_2_B = ReluActivationLayer()
    conv_layer_3_B = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_4_B = ReluActivationLayer()

    simpleCNN_B = SimpleCNN(2)
    simpleCNN_B.add_layer(conv_layer_1_for_B)
    simpleCNN_B.add_layer(activation_layer_2_B)
    simpleCNN_B.add_layer(conv_layer_3_B)
    simpleCNN_B.add_layer(activation_layer_4_B)
    simpleCNN_B.build(input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,
                      proximal_lbda=proximal_lbda)

    (guest_n_local, host_n_local) = n_local
    partyA = VFLHostModel(local_model=simpleCNN_A, n_iter=guest_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)
    partyB = VFLGuestModel(local_model=simpleCNN_B, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                           apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                           verbose=verbose)

    using_learning_rate_decay = False
    if using_learning_rate_decay:
        partyA.set_learning_rate_decay_func(sqrt_learning_rate_decay)
        partyB.set_learning_rate_decay_func(sqrt_learning_rate_decay)

    party_B_id = "B"
    federated_learning = VerticalMultiplePartyFederatedLearning(partyA,verbose=verbose)
    federated_learning.add_party(id=party_B_id, party_model=partyB)

    print("################################ Train Federated Models ############################")

    train_data = {federated_learning.get_main_party_id(): {"X": X_train_b_left, "Y": Ytrain_b},
                  "party_list": {party_B_id: X_train_b_right}}

    test_data = {federated_learning.get_main_party_id(): {"X": X_test_b_left, "Y": Ytest_b},
                 "party_list": {party_B_id: X_test_b_right}}

    fl_learner = VerticalFederatedLearningLearner(federated_learning)
    experiment_result = fl_learner.fit(train_data=train_data,
                                       test_data=test_data,
                                       is_parallel=is_parallel,
                                       epochs=epoch,
                                       batch_size=batch_size,
                                       is_debug=is_debug,
                                       verbose=verbose)
    


    task_name = "vfl_cnn"
    save_experimental_results(experiment_result, output_directory_name, task_name, show_fig)
    return fl_learner


# In[ ]:


if __name__ == '__main__':

    Xtrain, Ytrain, Xtest, Ytest = getKaggleMINST("data")
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)

    # choose two labels from the 10 digit labels and convert the two labels to 0 and 1
    Xtrain_b, Ytrain_b = get_binary_labels(Xtrain, Ytrain, [3, 8])
    Xtest_b, Ytest_b = get_binary_labels(Xtest, Ytest, [3, 8])

    Ytrain_b = Ytrain_b.reshape(Ytrain_b.shape[0], 1)
    Ytest_b = Ytest_b.reshape(Ytest_b.shape[0], 1)

    # split each image in half to simulate two-party vertical federated learning
    X_train_b_left, X_train_b_right = split_in_half(Xtrain_b)
    X_test_b_left, X_test_b_right = split_in_half(Xtest_b)

    print("[INFO] X_train_b_left", X_train_b_left.shape)
    print("[INFO] X_train_b_right", X_train_b_right.shape)
    print("[INFO] X_test_b_left", X_test_b_left.shape)
    print("[INFO] X_test_b_right", X_test_b_right.shape)
    print("################################ Build benchmark Models ############################")

    # benchmark_test(X_train_b_left, X_train_b_right, Ytrain_b, X_test_b_left, X_test_b_right, Ytest_b)

    print("################################ Build Federated Models ############################")

    is_debug = True
    verbose = True
    show_fig = True

    output_dir_name = "/vnn_demo/result/cnn_two_party/"
    n_experiments = 1
    apply_proximal = True
    proximal_lbda = 0.1
    batch_size = 512
    epoch = 1

    is_parallel_list = [True]

    # lr_list = [1e-03]
    # n_local_iter_list = [(1, 1)]  # [(number_local_iterations_guest, number_local_iterations_host)]

    lr_list = [1e-04]
    n_local_iter_list = [(5, 5)]  # [(number_local_iterations_guest, number_local_iterations_host)]
    n_local_iter_list = [(1, 1)]  # [(number_local_iterations_guest, number_local_iterations_host)]

    for is_parallel in is_parallel_list:
        for lr in lr_list:
            for n_local in n_local_iter_list:
                if show_fig: show_fig = False if n_experiments > 1 else True
                for i in range(n_experiments):
                    print("[INFO] communication_efficient_experiment: {0} for is_asy {1}, lr {2}, n_local {3}"
                          .format(i, is_parallel, lr, n_local))
                    X_train_b_left, X_train_b_right, Ytrain_b = shuffle(X_train_b_left, X_train_b_right, Ytrain_b)
                    X_test_b_left, X_test_b_right, Ytest_b = shuffle(X_test_b_left, X_test_b_right, Ytest_b)
                    train = [X_train_b_left, X_train_b_right, Ytrain_b]
                    test = [X_test_b_left, X_test_b_right, Ytest_b]
                    fl_learner=run_experiment(train_data=train, test_data=test, output_directory_name=output_dir_name,
                                   n_local=n_local, batch_size=batch_size, learning_rate=lr, is_parallel=is_parallel,
                                   apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, epochs=epoch,
                                   is_debug=is_debug, verbose=verbose, show_fig=show_fig)


    # # select a set of background examples to take an expectation over
    # background = X_train_b_left[np.random.choice(X_train_b_left.shape[0], 100, replace=False)]
    #
    # # explain predictions of the model on four images
    # e = shap.DeepExplainer(fl_learner, background)
    # # ...or pass tensors directly
    # # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    # shap_values = e.shap_values(Xtest[1:5])
    #
    # # plot the feature attributions
    # shap.image_plot(shap_values, -Xtest[1:5])





