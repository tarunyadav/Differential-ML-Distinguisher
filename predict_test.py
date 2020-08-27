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
        ####Print#######
        if(debug):
          print("======== Running Iteration %d ========\n"%(loop));
      #if (sys.argv[1]=="GIFT_64_ENCRYPT"):
          # data_in = np.array(range(0,2**int(sys.argv[3])))
          # data = data_in << int(sys.argv[4]);
          # block_data = np.array([[(diff >> 48) & 0xffff, (diff >> 32) & 0xffff, (diff >> 16) & 0xffff, diff & 0xffff] for diff in data]).transpose();
          # X = convert_to_binary_64_block(block_data,16,4);
          # Y = model.model.predict_classes(X,batch_size=5000);
          # Y_Prob = model.model.predict_proba(X,batch_size=5000);
          # #unique, counts = np.unique(Y, return_counts=True);
          # #print(dict(zip(unique, counts)));
          # #print(np.where(Y==0));
          
          # #unique_1, counts_1 = np.unique(Y_Prob, return_counts=True);
          # #print(dict(zip(unique_1, counts_1)));
          # print(np.max(Y_Prob));
          # print(len(np.where(Y_Prob==np.max(Y_Prob))[0]));
          # #print(np.sort(Y_Prob));
          # #for index, val in np.ndenumerate(Y_Prob):
          # #   print(index[0], val)
        
        
        n = int(sys.argv[4])**int(sys.argv[5]); 
        nr = int(sys.argv[6]);
        r_start = int(sys.argv[7]);
        r_mid = int(sys.argv[8]);
        Y_Prob_good = float(sys.argv[9]);
        if (sys.argv[1]=="GIFT_64_ENCRYPT"):
          diff_default=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1]),int(file_name.split('_')[7][1:-1].split(',')[2]),int(file_name.split('_')[7][1:-1].split(',')[3]))
          if (sys.argv[11] == "default_diff"):
              diff = diff_default;
          elif (sys.argv[11] == "random_diff" and loop==1):
              diff = (randint(0,(2**16)-1),randint(0,(2**16)-1),randint(0,(2**16)-1),randint(0,(2**16)-1));
          elif (sys.argv[11] == "fix_diff"):
              diff = (int(sys.argv[12],16),int(sys.argv[13],16),int(sys.argv[14],16),int(sys.argv[15],16));
        
          # output_Y = int(sys.argv[12]);
          # Y = np.frombuffer(urandom(n), dtype=np.uint8); 
          # if (output_Y==0):
          #   Y = (Y & 0);
          # elif (output_Y==1):
          #   Y = (Y & 1) | 1;
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
          diff_default=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1]))
          if (sys.argv[11] == "default_diff"):
              diff = diff_default;
          elif (sys.argv[11] == "random_diff"  and loop==1 ):
              #diff = (int(sys.argv[11],16),int(sys.argv[12],16));
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
          diff_default=(int(file_name.split('_')[7][1:-1].split(',')[0]),int(file_name.split('_')[7][1:-1].split(',')[1]))
          if (sys.argv[11] == "default_diff"):
              diff = diff_default;
          elif (sys.argv[11] == "random_diff"  and loop==1):
              #diff = (int(sys.argv[11],16),int(sys.argv[12],16));
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
        
        #X_mid_binary = convert_to_binary_64_block(np.array(cdata0_mid^cdata1_mid),16,4);
        X_mid = (np.array(cdata0_mid^cdata1_mid)).transpose();
        #print(np.where(X_mid==diff));
        #print(len(X_mid));
        #print(len(X_mid[0]));
        #print("Total No. of desired output diff %s after %d rounds is : %d\n"%(str(diff),r_mid,len(np.where(X_mid==diff)[0])))
        #print(np.where(X_mid==diff));
        Y_Predict = model.predict_classes(X,batch_size=5000);
        unique_1, counts_1 = np.unique(Y_Predict, return_counts=True);
        ####Print#######
        if(debug):
          print("Predicted No. of desired output diff %s after %d rounds is : %s\n"%(str(diff_default),r_mid,str(dict(zip(unique_1, counts_1))[1])));
          print("Real/Random (percentage) %f \n"%(dict(zip(unique_1, counts_1))[1] * 100/dict(zip(unique_1, counts_1))[0]));
        Y_Prob = model.predict(X,batch_size=5000);
        #print(Y_Prob)
        unique_1, counts_1 = np.unique(Y_Prob, return_counts=True);
        #print(np.max(Y_Prob));
        #print(len(np.where(Y_Prob==np.max(Y_Prob))[0]));
        #Y_Prob_good =(np.max(Y_Prob)-float(sys.argv[8]));
        ####Print#######
        if(debug):
          print("Predicted(Probability >= : %s) No. of desired output diff %s after %d rounds is : %s\n"%(str(Y_Prob_good),str(diff_default),r_mid,len(np.where(Y_Prob >= Y_Prob_good)[0])));
          print(np.where(Y_Prob >= Y_Prob_good)[0]);
        prediction_counter.append(len(np.where(Y_Prob >= Y_Prob_good)[0]));
        # X_mid_diff_count = 0;
        # X_mid_diff = [];
        # X_mid_diff_indices = [];
        # for i in range(0, len(X_mid)):
        #     #print(X_mid[i]);
        #     if ((X_mid[i][0]==diff[0]) and (X_mid[i][1]==diff[1]) and (X_mid[i][2]==diff[2]) and (X_mid[i][3]==diff[3])):
        #         X_mid_diff_count += 1;
        #         X_mid_diff.append(X_mid[i]);
        #         X_mid_diff_indices.append(i);
        #print(X_mid_diff_count);
        #print("Total No. of desired output diff %s after %d rounds is : %d\n"%(str(diff),r_mid,X_mid_diff_count));
        X_mid_diff_indices = np.where(np.all(X_mid==diff_default,axis=1))[0]
        #print("");
        ####Print#######
        if(debug):
          print("Total No. of desired output diff %s after %d rounds is : %d\n"%(str(diff_default),r_mid,len(X_mid_diff_indices)));
        #print(X_mid_diff);
        #print(X_mid_diff_indices);
  np_prediction_counter = np.array(prediction_counter);
  if(debug):
    print("Total No. of samples where prediction is >= %d is: %d\n"%(cutoff,len(np.where(np_prediction_counter >= cutoff)[0])));
  distinguisher_counter.append(len(np.where(np_prediction_counter >= cutoff)[0]));
np_distinguisher_counter = np.array(distinguisher_counter);
print ("Total No. of Samples Distinguished: %d"%(len(np.where(np_distinguisher_counter >= (int(int(sys.argv[3])/2)+1) )[0])));
#if (sys.argv[1]=="speck"):
#  print ("Total No. of Samples Distinguished: %d"%(len(np.where(np_distinguisher_counter >= 9)[0])));
#elif (sys.argv[1]=="simon"):
#  print ("Total No. of Samples Distinguished: %d"%(len(np.where(np_distinguisher_counter >= 1)[0])));
#elif (sys.argv[1]=="GIFT_64_ENCRYPT"):
#  print ("Total No. of Samples Distinguished: %d"%(len(np.where(np_distinguisher_counter >= 3)[0])));
print(distinguisher_counter);
  #call predict_test.py GIFT_64_ENCRYPT ./simon/nets/..hd5 2 12(n) 8(nr) 1(r_start) 4(r_mid) 0.000001(error) 0x0000 0x0000 0x0000 0x0000/default_diff 5(iteration)