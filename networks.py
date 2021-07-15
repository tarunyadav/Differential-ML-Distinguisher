# Networks for SPECK-32, SIMON-32 and GIFT-64 

import os

# import comet_ml in the top of your file
#from comet_ml import Experiment
# Add the following code anywhere in your machine learning file
#experiment = Experiment(api_key="", project_name="differentialml", workspace="")

import speck as sp
import simon as si
import gift_64 as gift
import numpy as np


from pickle import dump
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense



def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_acc', save_best_only = True);
  return(res);

def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=1024, d2=1024, word_size=16, ks=3,depth=2, reg_param=0.0001, final_activation='sigmoid'):
  
  model_mlp = Sequential();
  dense1 = Dense(d1,activation="relu",input_shape=(num_blocks * word_size * 2,));
  model_mlp.add(dense1);
  for i in range(depth):
    dense2 = Dense(d2,activation="relu");
    model_mlp.add(dense2);
  out = Dense(num_outputs, activation=final_activation);
  model_mlp.add(out);
  return(model_mlp);



def train_distinguisher(num_epochs, num_rounds=7, depth=1, neurons=1024, data_train=2**18, data_test=2**16,cipher="speck",difference=(0,0x0020),start_round=1,pre_trained_model="fresh"):
    if (cipher=="speck"): 
      wdir = './speck_nets/';
      if (pre_trained_model=="fresh"):
          net = make_resnet(depth=depth, reg_param=10**-5, d1=32, d2=neurons,num_blocks=2,word_size=int(16/2));#word size decreased(8) 
          net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      else:
          net = load_model(pre_trained_model);
      net.summary();
      X, Y = sp.make_train_data(data_train,num_rounds,diff=difference,r_start=start_round);
      X_eval, Y_eval = sp.make_train_data(data_test, num_rounds,diff=difference,r_start=start_round);
    elif (cipher=="simon"):
      wdir = './simon_nets/';
      if (pre_trained_model=="fresh"):
          net = make_resnet(depth=depth, reg_param=10**-5, d1=32, d2=neurons,num_blocks=2,word_size=int(16/2));#word size decreased(8) because c0^c1
          net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      else:
          net = load_model(pre_trained_model);
      net.summary();
      X, Y = si.make_train_data(data_train,num_rounds,diff=difference,r_start=start_round);
      X_eval, Y_eval = si.make_train_data(data_test, num_rounds,diff=difference,r_start=start_round);
    elif (cipher=="GIFT_64"):
      wdir = './gift_64_nets/';
      if (pre_trained_model=="fresh"): 
          net = make_resnet(depth=depth,d1=64,d2=neurons,num_blocks=4,word_size=int(16/2)); #word size decreased(8) because c0^c1
          net.compile(optimizer='adam',loss='mse',metrics=['acc']);
      else:
          net = load_model(pre_trained_model);
      net.summary();
      
      X, Y = gift.make_train_data(data_train,num_rounds,diff=difference,r_start=start_round);
      X_eval, Y_eval = gift.make_train_data(data_test, num_rounds,diff=difference,r_start=start_round);

    print(difference);
    if not os.path.exists(wdir):
      os.makedirs(wdir)
    #set up model checkpoint
    if (pre_trained_model=="fresh"): 
      check = make_checkpoint(wdir+'best_'+str(num_rounds)+'_start_'+str(start_round)+'_depth_'+str(depth)+'_diff_'+str(difference)+'_data_train_'+str(data_train)+'_data_test_'+str(data_test)+"_epoch-{epoch:02d}_val_acc-{val_acc:.2f}" + '.h5');
      print("Model will be stroed in File: " + wdir+'best_'+str(num_rounds)+'_start_'+str(start_round)+'_depth_'+str(depth)+'_diff_'+str(difference)+'_data_train_'+str(data_train)+'_data_test_'+str(data_test)+'.h5');
    else:
      check = make_checkpoint(pre_trained_model[:pre_trained_model.index("_epoch")]+"_epoch_imp-{epoch:02d}_val_acc_imp-{val_acc:.2f}" + '.h5'); 
      print("Model will be stroed in File: " + pre_trained_model[:pre_trained_model.index("_epoch")]+"_epoch_imp---_val_acc_imp----.h5");
    
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    h = net.fit(X,Y,epochs=num_epochs,batch_size=5000,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);
 