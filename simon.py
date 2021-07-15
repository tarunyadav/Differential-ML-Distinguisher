# SIMON-32 Implementation

import numpy as np
from os import urandom
#import sys

def WORD_SIZE():
    return(16);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    x , y  = p[0] , p[1];
    tmp_1 = rol(x,1) & rol(x,8); 
    tmp_2 = rol(x,2); 
    x = y ^ tmp_1 ^ tmp_2 ^ k;  
    return(x,p[0]);

def dec_one_round(c,k):
    x , y  = c[0] , c[1];
    tmp_1 = rol(y,1) & rol(y,8);
    tmp_2 = rol(y,2);
    y = x ^ tmp_1 ^ tmp_2 ^ k;
    return(c[1],y)

def expand_key(k, t):
    m = 4;
    z = "11111010001001010110000111001101111101000100101011000011100110";
    ks = [0 for i in range(t)];
    ks[0:m] = list(reversed(k[:len(k)]));
    for i in range(m, t):
      tmp = ror(ks[i-1],3);
      if (m==4):
          tmp = tmp ^ ks[i-3];
      tmp_1 = ror(tmp,1);
      tmp = tmp ^ tmp_1;
      tmp_2 = int(z[(i-m)%62]);
      ks[i] = ks[i-m] ^ tmp ^ tmp_2 ^ 0xfffc
    return(ks);

def encrypt(p, ks,r_start=1):
    x, y = p[0], p[1];
    for k in ks[r_start-1:]:
        x,y = enc_one_round((x,y), k);
    return(np.array((x, y)));


def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6565, 0x6877)
  ks = expand_key(key, 32)
  ct = encrypt(pt, ks)
  #if (ct == (0xa868, 0x42f2)):
  if (ct == (0xc69b, 0xe9bb)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    print(ct)
    return(False);
#check_testvector();
#sys.exit()

def convert_to_binary(arr):
  X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(4 * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);
def convert_to_binary_new(arr,WORD_SIZE=16,NO_OF_WORDS=2):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

def make_train_data(n, nr, diff=(0,0x0020),r_start=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.repeat(np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1),n,axis=1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, (r_start-1) + nr);
  ctdata0 = encrypt((plain0l, plain0r), ks,r_start);
  ctdata1= encrypt((plain1l, plain1r), ks,r_start);
  X = convert_to_binary_new(np.array(ctdata0^ctdata1),16,2);
  return(X,Y);
def make_train_data_no_random(n, nr, diff=(0,0x0020),r_start=1,output_Y=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); 
  if (output_Y==0):
    Y = (Y & 0);
  elif (output_Y==1):
    Y = (Y & 1) | 1;
  keys = np.repeat(np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1),n,axis=1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  ks = expand_key(keys, (r_start-1) + nr);
  ctdata0 = encrypt((plain0l, plain0r), ks,r_start);
  ctdata1= encrypt((plain1l, plain1r), ks,r_start);
  X = convert_to_binary_new(np.array(ctdata0^ctdata1),16,2);
  return(X,Y);
def real_differences_data(n, nr, diff=(0,0x0020)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);
