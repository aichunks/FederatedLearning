import pickle as pk
import tensorflow as tf
import numpy as np
from models.cnn import SimpleCNN
from vfl_new import VFLHostModel,VFLGuestModel
from Utils import getData

host_n_local,apply_proximal,is_debug,verbose=1,True,True,True
learning_rate,proximal_lbda=1e-04,0.1
model_path="vnn_demo/models/CNN.pk"

def LoadModel(modelID,input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,proximal_lbda=proximal_lbda):
    simpleCNN = SimpleCNN(modelID)
    simpleCNN.setVariable(input_shape=input_shape, representation_dim=representation_dim, class_num=class_num, lr=lr,proximal_lbda=proximal_lbda)
    return simpleCNN

def BuildModels():
    with open(model_path, "rb") as f:
        weigths = pk.load(f)
    party_a_modelperemeter = weigths["party_a_modelperemeter"]
    party_b_modelperemeter = weigths["party_modelperemeter_list"][0]
    simpleCNN_A = LoadModel(1)
    simpleCNN_B = LoadModel(2)
    partyA = VFLGuestModel(local_model=simpleCNN_A, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                           apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                           verbose=verbose)
    partyB = VFLHostModel(local_model=simpleCNN_B, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)
    partyA.load_weights(party_a_modelperemeter)
    partyB.load_weights(party_b_modelperemeter)
    return partyA,partyB

# simpleCNN_A,simpleCNN_B=BuildCnnModels(learning_rate,proximal_lbda)
partyA,partyB=BuildModels()

[X_train_b_left, X_train_b_right], Ytrain_b, [X_test_b_left, X_test_b_right], Ytest_b=getData(data_shuffle=False)


def getModelOutput(partyB,X_train_b_right):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        partyB.set_session(sess)
        sess.run(init)
        # x=np.random.random((2,28,14,1))
        x=X_train_b_right[:2]
        partyB.set_batch(x,1)
        out=partyB.send_components()
        print([o.shape for o in out])
        for o in out:
            print(o)

getModelOutput(partyB,X_train_b_right)

