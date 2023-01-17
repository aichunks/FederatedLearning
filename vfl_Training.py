#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.chdir("FedBCD-main")

import sys
import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from config import data_dir
from Utils import getData
from models.cnn import ConvolutionLayer, ReluActivationLayer, SimpleCNN,BuildCnnModels
from models.learning_rate_decay import sqrt_learning_rate_decay
from models.regularization import EarlyStoppingCheckPoint
from store_utils import save_experimental_results
from vnn_demo.vfl import VFLGuestModel, VFLHostModel, VerticalMultiplePartyFederatedLearning
from vnn_demo.vfl_learner import VerticalFederatedLearningLearner



def run_experiment(train_data, test_data, output_directory_name,model_path, n_local, batch_size, learning_rate, is_parallel,
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

    simpleCNN_A,simpleCNN_B=BuildCnnModels(learning_rate,proximal_lbda)
    (guest_n_local, host_n_local) = n_local
    # Recievening Component
    partyA = VFLHostModel(local_model=simpleCNN_A, n_iter=guest_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)
    # Sending Components
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

    os.makedirs(model_path, exist_ok=True)
    fl_learner = VerticalFederatedLearningLearner(federated_learning)
    os.makedirs(model_path, exist_ok=True)
    experiment_result = fl_learner.fit(train_data=train_data,
                                       test_data=test_data,
                                       is_parallel=is_parallel,
                                       epochs=epoch,
                                       batch_size=batch_size,
                                       is_debug=is_debug,
                                       verbose=verbose,
                                       model_path=model_path+"CNN")

    # os.makedirs(model_path, exist_ok=True)
    # if verbose: print("Saving Models")
    # simpleCNN_A.saveModel(model_path + "CNNA")
    # simpleCNN_B.saveModel(model_path + "CNNB")

    task_name = "vfl_cnn"
    save_experimental_results(experiment_result, output_directory_name, task_name, show_fig)
    return fl_learner


# In[ ]:


if __name__ == '__main__':


    [X_train_b_left, X_train_b_right], Ytrain_b, [X_test_b_left, X_test_b_right], Ytest_b=getData()


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
    model_path="vnn_demo/models/"
    n_experiments = 1
    apply_proximal = True
    proximal_lbda = 0.1
    batch_size = 512
    epoch = 1

    is_parallel_list = [True]
    # is_parallel_list = [False]


    # lr_list = [1e-03]
    # n_local_iter_list = [(1, 1)]  # [(number_local_iterations_guest, number_local_iterations_host)]

    lr_list = [1e-04]
    n_local_iter_list = [(5, 5)]  # [(number_local_iterations_guest, number_local_iterations_host)]
    # n_local_iter_list = [(1, 1)]  # [(number_local_iterations_guest, number_local_iterations_host)]

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
                    fl_learner=run_experiment(train_data=train, test_data=test, output_directory_name=output_dir_name,model_path=model_path,
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





