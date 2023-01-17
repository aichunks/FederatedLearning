#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.chdir("FedBCD-main")

import tensorflow as tf
from Utils import getData
from models.cnn import BuildCnnModels
from models.learning_rate_decay import sqrt_learning_rate_decay
from store_utils import save_experimental_results
from vnn_demo.vfl import VFLGuestModel, VFLHostModel, VerticalMultiplePartyFederatedLearning
from vnn_demo.vfl_learner import VerticalFederatedLearningLearner


def run_Training(train_data, test_data, output_directory_name,model_path,numberofParties, n_local, batch_size, learning_rate, is_parallel,
                   epochs=5, apply_proximal=False, proximal_lbda=0.1, is_debug=False, verbose=False, show_fig=True):

    print("hyper-parameters:")
    print("# Number of parties {0}".format(numberofParties))
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

    X_train_b_list, Ytrain_b = train_data
    X_test_b_list, Ytest_b = test_data

    tf.reset_default_graph()

    simpleCNN_list=BuildCnnModels(learning_rate,proximal_lbda,n=numberofParties)

    (guest_n_local, host_n_local) = n_local
    # Recievening Component
    hostparty = VFLHostModel(local_model=simpleCNN_list[0], n_iter=guest_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,verbose=verbose)
    party_list=[]
    # Sending Components
    for i in range(1,numberofParties):
        party_list.append(VFLGuestModel(local_model=simpleCNN_list[i], n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                               apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                               verbose=verbose))

    using_learning_rate_decay = False
    if using_learning_rate_decay:
        hostparty.set_learning_rate_decay_func(sqrt_learning_rate_decay)
        for party in party_list:
            party.set_learning_rate_decay_func(sqrt_learning_rate_decay)

    federated_learning = VerticalMultiplePartyFederatedLearning(hostparty,verbose=verbose)
    for i,party in enumerate(party_list):
        federated_learning.add_party(id=chr(ord("B")+i), party_model=party)
    print("################################ Train Federated Models ############################")

    train_data = {federated_learning.get_main_party_id(): {"X": X_train_b_list[0], "Y": Ytrain_b},
                  "party_list": {party_id: X_train_b_list[i+1] for i,party_id in enumerate(federated_learning.party_dict.keys())}}

    test_data = {federated_learning.get_main_party_id(): {"X": X_test_b_list[0], "Y": Ytest_b},
                  "party_list": {party_id: X_test_b_list[i+1] for i,party_id in enumerate(federated_learning.party_dict.keys())}}

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
                                       model_path=model_path+"CNN_"+str(numberofParties)+"_parties")

    task_name = "vfl_cnn"
    save_experimental_results(experiment_result, output_directory_name, task_name, show_fig)
    return fl_learner


# In[ ]:


if __name__ == '__main__':

    numberofParties=2 # only 2 and 4 suppoted
    # X_train_b_left, X_train_b_right, Ytrain_b, X_test_b_left, X_test_b_right, Ytest_b=getData(numberofParties=numberofParties)
    X_train_b_list, Ytrain_b, X_test_b_list, Ytest_b=getData(numberofParties=numberofParties)

    for i,X_train_b in enumerate(X_train_b_list):
        print("[INFO] X_train_",i, X_train_b.shape)
    for i, X_test_b in enumerate(X_test_b_list):
        print("[INFO] X_test", i, X_test_b.shape)

    print("################################ Build benchmark Models ############################")
    # benchmark_test(X_train_b_left, X_train_b_right, Ytrain_b, X_test_b_left, X_test_b_right, Ytest_b)

    print("################################ Build Federated Models ############################")

    is_debug = True
    verbose = True
    show_fig = True

    output_dir_name = "/vnn_demo/result/cnn_n_party/"
    model_path="vnn_demo/muli_party_models/"
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
                    train = [X_train_b_list, Ytrain_b]
                    test = [X_test_b_list, Ytest_b]
                    fl_learner=run_experiment(train_data=train, test_data=test, output_directory_name=output_dir_name,model_path=model_path,
                                   numberofParties=numberofParties,n_local=n_local, batch_size=batch_size, learning_rate=lr, is_parallel=is_parallel,
                                   apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, epochs=epoch,
                                   is_debug=is_debug, verbose=verbose, show_fig=show_fig)






