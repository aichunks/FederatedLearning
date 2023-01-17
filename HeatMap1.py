import os
import pickle as pk
import tensorflow as tf
import numpy as np
from models.cnn import SimpleCNN
from vnn_demo.vfl  import VFLGuestModel,VFLHostModel
from Utils import getData,makeImageOFF
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm



host_n_local,apply_proximal,is_debug,verbose=1,True,True,False
learning_rate,proximal_lbda=1e-04,0.1
model_path="vnn_demo/models/CNN.pk"
outpath="output/"
os.makedirs(outpath, exist_ok=True)


def makeModel(modelID,input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,proximal_lbda=proximal_lbda):
    simpleCNN = SimpleCNN(modelID)
    simpleCNN.setVariable(input_shape=input_shape, representation_dim=representation_dim, class_num=class_num, lr=lr,proximal_lbda=proximal_lbda)
    return simpleCNN

def BuildModels():
    with open(model_path, "rb") as f:
        weigths = pk.load(f)
    party_a_modelperemeter = weigths["party_a_modelperemeter"]
    party_b_modelperemeter = weigths["party_modelperemeter_list"][0]
    simpleCNN_A = makeModel(1)
    simpleCNN_B = makeModel(2)
    partyA = VFLHostModel(local_model=simpleCNN_A, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)
    partyB = VFLGuestModel(local_model=simpleCNN_B, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                           apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                           verbose=verbose)
    partyA.load_weights(party_a_modelperemeter)
    partyB.load_weights(party_b_modelperemeter)
    return partyA,partyB



def GenerateHeatmap(party,X_train,index=1):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        party.set_session(sess)
        sess.run(init)
        heatmap=party.localModel.getHeatMap(X_train[index:index+1])
        print(heatmap.shape)
    return heatmap[0,:,:,0],X_train[index]

def save_and_display_gradcam(img, heatmap, img_path="img.jpg",cam_path="imgHeatmap.jpg", alpha=0.1):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    img = np.concatenate((img, img, img),axis=-1)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    superimposed_array = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_array)
    # superimposed_img = keras.preprocessing.image.array_to_img(img)
    # Save the superimposed image
    superimposed_img.save(cam_path)
    keras.preprocessing.image.array_to_img(img).save(img_path)
    # Display Grad CAM
    # display(Image(cam_path))
    plt.imshow( img)
    plt.show()
    plt.imshow(superimposed_array)
    plt.show()


def saveHeatMapImages(images,heatmaps,heatmpapath=outpath+"heatmap/",alpha=.4):
    os.makedirs(heatmpapath,exist_ok=True)
    # Rescale heatmap to a range 0-255
    heatmaps = np.uint8(255 * heatmaps)
    images=np.repeat(images, 3, axis=-1)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmaps]
    # Superimpose the heatmap on original image
    superimposed_array = jet_heatmap * alpha + images
    superimposed_imgs = [keras.preprocessing.image.array_to_img(arr) for arr in superimposed_array]
    # superimposed_img = keras.preprocessing.image.array_to_img(img)
    # Save the superimposed images
    [superimposed_img.save(heatmpapath+str(i)+".jpg") for i,superimposed_img in enumerate(superimposed_imgs)]
    # keras.preprocessing.image.array_to_img(img).save(img_path)


def gererateHeatMapInBatch(party,X_train,batch_size):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        party.set_session(sess)
        sess.run(init)
        heatmaplist=[]
        for i in range(0,X_train.shape[0],batch_size):
            heatmaplist.append(party.localModel.getHeatMap(X_train[i:i+batch_size]))
    heatmap=np.concatenate(heatmaplist)
    return heatmap[:,:,:,0]

def BuildAndSaveHeatMap(party,X,outputptah,bacthsize=256):
    heatmap = gererateHeatMapInBatch(party, X, batch_size=bacthsize)
    saveHeatMapImages(X, heatmap, outputptah)


# simpleCNN_A,simpleCNN_B=BuildCnnModels(learning_rate,proximal_lbda)
partyA,partyB=BuildModels()

[X_train_b_left, X_train_b_right], Ytrain_b, [X_test_b_left, X_test_b_right], Ytest_b=getData(data_shuffle=False)


leftshapvalues=[[1,1],
                [1,1],
                [1,1],
                [1,1],]

rightshapvalues=[[1,1],
                [1,1],
                [1,1],
                [1,1],]


X_train_b_left,X_train_b_right=makeImageOFF(X_train_b_left,X_train_b_right,(leftshapvalues,rightshapvalues))
X_test_b_left,X_test_b_right=makeImageOFF(X_test_b_left,X_test_b_right,(leftshapvalues,rightshapvalues))

# heatmap,image=GenerateHeatmap(partyB,X_train_b_right,10)
# save_and_display_gradcam(image, heatmap, alpha=0.4)

BuildAndSaveHeatMap(partyA,X_train_b_left,outpath+"heatmap_train_left/",bacthsize=256)
BuildAndSaveHeatMap(partyB,X_train_b_right,outpath+"heatmap_train_right/",bacthsize=256)
BuildAndSaveHeatMap(partyA,X_test_b_left,outpath+"heatmap_test_left/",bacthsize=256)
BuildAndSaveHeatMap(partyB,X_test_b_right,outpath+"heatmap_test_right/",bacthsize=256)


