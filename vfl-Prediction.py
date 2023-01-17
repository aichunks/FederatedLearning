import os
import pickle as pk
import tensorflow as tf
import numpy as np
from models.cnn import SimpleCNN
from sklearn.metrics import roc_auc_score
from vnn_demo.vfl  import VFLGuestModel,VFLHostModel
from Utils import getData,makeImageOFF
from vnn_demo.vfl import VFLGuestModel, VFLHostModel, VerticalMultiplePartyFederatedLearning
from vnn_demo.vfl_learner import compute_correct_prediction



host_n_local,apply_proximal,is_debug,verbose=1,True,True,False
learning_rate,proximal_lbda=1e-04,0.1
model_path="vnn_demo/models/CNN.pk"
model_path="vnn_demo/muli_party_models/CNN.pk"
outpath="output/"
os.makedirs(outpath, exist_ok=True)
elementperblock=7
assert 28%elementperblock==0, "elementperblock must be divisible by 28"

def makeModel(modelID,input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,proximal_lbda=proximal_lbda):
    simpleCNN = SimpleCNN(modelID)
    simpleCNN.setVariable(input_shape=input_shape, representation_dim=representation_dim, class_num=class_num, lr=lr,proximal_lbda=proximal_lbda)
    return simpleCNN

def BuildModels():
    with open(model_path, "rb") as f:
        weigths = pk.load(f)
    party_a_modelperemeter = weigths["party_a_modelperemeter"]
    party_modelperemeter_list = weigths["party_modelperemeter_list"]
    input_shape=party_a_modelperemeter["hyperparameters"]["input_shape"]
    numberofparties=len(party_modelperemeter_list)+1
    simpleCNN_list=[makeModel(i+1,input_shape) for i in range(numberofparties)]

    hostparty = VFLHostModel(local_model=simpleCNN_list[0], n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)
    partylist = [VFLGuestModel(local_model=simpleCNN_list[i], n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                           apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                           verbose=verbose) for i in range(1,numberofparties)]
    hostparty.load_weights(party_a_modelperemeter)
    for i,party in enumerate(partylist): party.load_weights(party_modelperemeter_list[i])
    # simpleCNN_A,simpleCNN_B=BuildCnnModels(learning_rate,proximal_lbda)
    federated_learning = VerticalMultiplePartyFederatedLearning(hostparty, verbose=verbose)
    for i,party in enumerate(partylist):
        federated_learning.add_party(id=chr(ord("B")+i), party_model=party)
    return hostparty,partylist,federated_learning,numberofparties


def makePredictions(shapvalues,threshold=0.5):
    # simpleCNN_A,simpleCNN_B=BuildCnnModels(learning_rate,proximal_lbda)
    hostparty, partylist, federated_learning, numberofparties = BuildModels()
    X_train_b_list, Ytrain_b, X_test_b_list, Ytest_b = getData(data_shuffle=False,numberofParties=numberofparties,shapvalues=shapvalues)
    X_list,  Y = X_test_b_list, Ytest_b
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        federated_learning.set_session(sess)
        sess.run(init)
        y_prob_preds=federated_learning.predict(X_list[0],{party_id: X_list[i+1] for i,party_id in enumerate(federated_learning.party_dict.keys())})
    y_hat_lbls, statistics = compute_correct_prediction(y_targets=Y,
                                                        y_prob_preds=y_prob_preds,
                                                        threshold=threshold)
    pred_pos_count, pred_neg_count, correct_count = statistics
    acc = correct_count / len(Y)
    auc = roc_auc_score(Y, y_prob_preds, average="weighted")
    print(f"Accuracy {acc} , AUC {auc}")

shapvalues=[[1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],]


makePredictions(shapvalues)




