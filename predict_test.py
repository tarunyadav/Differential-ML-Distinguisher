#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:50:33 2020

@author: tarunyadav
"""


from keras.models import load_model
import numpy as np
import gift_64_opt as gi_64_opt
import speck as sp
import simon as si
import sys
from os import urandom 
import os
from tqdm.notebook import tqdm
from random import randint
# load model
model = load_model(sys.argv[2]);
file_name = os.path.basename(sys.argv[2]);
#model.summary();

def convert_to_binary_64_block(arr,WORD_SIZE=16,NO_OF_WORDS=8):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

cutoff = int(sys.argv[10]);
debug = False;
if (sys.argv[-1] == "debug"):
  debug = True;
distinguisher_counter = [];
for test in tqdm(range(0,50)):
  prediction_counter = [];
  for loop in range(1,int(sys.argv[3])+1):
        if(debug):
          print("======== Running Iteration %d ========\n"%(loop));

        
        n = int(sys.argv[4])**int(sys.argv[5]); 
        nr = int(sys.argv[6]);
        r_start = int(sys.argv[7]);
        r_mid = int(sys.argv[8]);
        Y_Prob_good = float(sys.argv[9]);
        if (sys.argv[1]=="GIFT_64"):
          diff_default=(int(file_name.split('_')[4][1:-1].split(',')[0]),int(file_name.split('_')[4][1:-1].split(',')[1]),int(file_name.split('_')[4][1:-1].split(',')[2]),int(file_name.split('_')[4][1:-1].split(',')[3]))
          if (sys.argv[11] == "default_diff"):
              diff = diff_default;
          elif (sys.argv[11] == "random_diff" and loop==1):
              diff = (randint(0,(2**16)-1),randint(0,(2**16)-1),randint(0,(2**16)-1),randint(0,(2**16)-1));
          elif (sys.argv[11] == "fix_diff"):
              diff = (int(sys.argv[12],16),int(sys.argv[13],16),int(sys.argv[14],16),int(sys.argv[15],16));

          if (loop==1):
            keys = np.repeat(np.frombuffer(urandom(16),dtype=np.uint16).reshape(8,-1),n,axis=1);
        
          plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
          plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
          plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
          plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);
        
          plain1_0 = plain0_0 ^ diff[0];
          plain1_1 = plain0_1 ^ diff[1];
          plain1_2 = plain0_2 ^ diff[2];
          plain1_3 = plain0_3 ^ diff[3];
          
          encrypt_data = 1; 
        
          if(encrypt_data==1):
              ks = gi_64_opt.expand_key(keys, (r_start-1) + nr);
              cdata0= gi_64_opt.encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
              cdata1 = gi_64_opt.encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);
              cdata0_mid= gi_64_opt.encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks[0:r_mid],r_start);
              cdata1_mid = gi_64_opt.encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks[0:r_mid],r_start);
          else:
              ks = gi_64_opt.expand_key(keys,r_start);
              cdata0 = gi_64_opt.decrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks, r_start,r_end = r_start-nr);
              cdata1 = gi_64_opt.decrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks, r_start,r_end = r_start-nr);
          X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,4);

        elif (sys.argv[1]=="speck"):
          diff_default=(int(file_name.split('_')[4][1:-1].split(',')[0]),int(file_name.split('_')[4][1:-1].split(',')[1]))
          if (sys.argv[11] == "default_diff"):
              diff = diff_default;
          elif (sys.argv[11] == "random_diff"  and loop==1 ):
              diff = (randint(0,(2**16)-1),randint(0,(2**16)-1));
          elif (sys.argv[11] == "fix_diff"):
              diff = (int(sys.argv[12],16),int(sys.argv[13],16));
          if (loop==1):
            keys = np.repeat(np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1),n,axis=1);
        
          plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
          plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
        
          plain1_0 = plain0_0 ^ diff[0];
          plain1_1 = plain0_1 ^ diff[1];
      
          ks = sp.expand_key(keys, (r_start-1) + nr);
          cdata0= sp.encrypt((plain0_0, plain0_1), ks,r_start);
          cdata1 = sp.encrypt((plain1_0, plain1_1), ks,r_start);
          cdata0_mid= sp.encrypt((plain0_0, plain0_1), ks[0:r_mid],r_start);
          cdata1_mid = sp.encrypt((plain1_0, plain1_1), ks[0:r_mid],r_start);
          X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,2);

        elif (sys.argv[1]=="simon"):
          diff_default=(int(file_name.split('_')[4][1:-1].split(',')[0]),int(file_name.split('_')[4][1:-1].split(',')[1]))
          if (sys.argv[11] == "default_diff"):
              diff = diff_default;
          elif (sys.argv[11] == "random_diff"  and loop==1):
              diff = (randint(0,(2**16)-1),randint(0,(2**16)-1));
          elif (sys.argv[11] == "fix_diff"):
              diff = (int(sys.argv[12],16),int(sys.argv[13],16)); 
          if (loop==1):
            keys = np.repeat(np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1),n,axis=1);
        
          plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
          plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
        
          plain1_0 = plain0_0 ^ diff[0];
          plain1_1 = plain0_1 ^ diff[1];
      
          ks = sp.expand_key(keys, (r_start-1) + nr);
          cdata0= si.encrypt((plain0_0, plain0_1), ks,r_start);
          cdata1 = si.encrypt((plain1_0, plain1_1), ks,r_start);
          cdata0_mid= si.encrypt((plain0_0, plain0_1), ks[0:r_mid],r_start);
          cdata1_mid = si.encrypt((plain1_0, plain1_1), ks[0:r_mid],r_start);
          X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,2);
        
        X_mid = (np.array(cdata0_mid^cdata1_mid)).transpose();

        Y_Predict = model.predict_classes(X,batch_size=5000);
        unique_1, counts_1 = np.unique(Y_Predict, return_counts=True);
        if(debug):
          print("Predicted No. of desired output diff %s after %d rounds is : %s\n"%(str(diff_default),r_mid,str(dict(zip(unique_1, counts_1))[1])));
          print("Real/Random (percentage) %f \n"%(dict(zip(unique_1, counts_1))[1] * 100/dict(zip(unique_1, counts_1))[0]));
        Y_Prob = model.predict(X,batch_size=5000);
        unique_1, counts_1 = np.unique(Y_Prob, return_counts=True);

        if(debug):
          print("Predicted(Probability >= : %s) No. of desired output diff %s after %d rounds is : %s\n"%(str(Y_Prob_good),str(diff_default),r_mid,len(np.where(Y_Prob >= Y_Prob_good)[0])));
          print(np.where(Y_Prob >= Y_Prob_good)[0]);
        prediction_counter.append(len(np.where(Y_Prob >= Y_Prob_good)[0]));

        X_mid_diff_indices = np.where(np.all(X_mid==diff_default,axis=1))[0]

        if(debug):
          print("Total No. of desired output diff %s after %d rounds is : %d\n"%(str(diff_default),r_mid,len(X_mid_diff_indices)));

  np_prediction_counter = np.array(prediction_counter);
  if(debug):
    print("Total No. of samples where prediction is >= %d is: %d\n"%(cutoff,len(np.where(np_prediction_counter >= cutoff)[0])));
  distinguisher_counter.append(len(np.where(np_prediction_counter >= cutoff)[0]));
np_distinguisher_counter = np.array(distinguisher_counter);

print ("Total No. of Samples Distinguished: %d"%(len(np.where(np_distinguisher_counter >= (int(int(sys.argv[3])/2)+1) )[0])));
print(distinguisher_counter);