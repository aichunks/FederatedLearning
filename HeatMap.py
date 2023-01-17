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
# model_path="vnn_demo/models/CNN.pk"
model_path="vnn_demo/muli_party_models/CNN.pk"
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
    return hostparty,partylist,numberofparties



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
hostparty,party_list,numberofparties=BuildModels()

shapvalues=[[1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],]

X_train_b_list, Ytrain_b, X_test_b_list, Ytest_b=getData(data_shuffle=False,numberofParties=numberofparties,shapvalues=shapvalues)

# heatmap,image=GenerateHeatmap(partyB,X_train_b_right,10)
# save_and_display_gradcam(image, heatmap, alpha=0.4)

BuildAndSaveHeatMap(hostparty,X_train_b_list[0],outpath+"heatmap_train_1/",bacthsize=256)
BuildAndSaveHeatMap(hostparty,X_test_b_list[0],outpath+"heatmap_test_1/",bacthsize=256)
for i,party in enumerate(party_list):
    BuildAndSaveHeatMap(party,X_train_b_list[i+1],outpath+"heatmap_train_"+str(i+2)+"/",bacthsize=256)
    BuildAndSaveHeatMap(party,X_test_b_list[i+1],outpath+"heatmap_test_"+str(i+2)+"/",bacthsize=256)


