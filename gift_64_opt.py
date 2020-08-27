#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 03:24:45 2020

@author: tarunyadav
"""
import numpy as np
from os import urandom
import sys
GIFT_RC = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
    0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
    0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
    0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
    0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
    0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28,
    0x10, 0x20];

def SWAPMOVE_1(X,M,n):
    # A = X;
    # B = X;
    # T=0x0000;
    # T = (B ^ (A >> n)) & M;
    # B ^= T;
    # A ^= (T << n);
    return( (X ^ ((X ^ (X >> n)) & M)) ^ (((X ^ (X >> n)) & M)<< n));

def SWAPMOVE_2(A,B,M,n):
    # A = X;
    # B = Y;
    # T=0;
    # T = (B ^ (A >> n)) & M;
    # B ^= T;
    # A ^= (T << n);
    # return(A,B);
    return(A ^ (((B ^ (A >> n)) & M)<< n), B ^ ((B ^ (A >> n)) & M));
    
def rowperm(S, B0_pos, B1_pos, B2_pos, B3_pos):
    T=0x0000;
    for b in range(0,4):
        T |= ((S>>(4*b+0))&0x1)<<(b + 4*B0_pos);
        T |= ((S>>(4*b+1))&0x1)<<(b + 4*B1_pos);
        T |= ((S>>(4*b+2))&0x1)<<(b + 4*B2_pos);
        T |= ((S>>(4*b+3))&0x1)<<(b + 4*B3_pos);
    return(T); 




def expand_key(k, nr):
    #W = [k_i for k_i in k]
    W = np.copy(k);
    #W = list(k);
    ks = [0 for i in range(nr)];
    ks[0] = W.copy();
    for i in range(1, nr):
        T6 = (W[6]>>2) | (W[6]<<14);
        T7 = (W[7]>>12) | (W[7]<<4);
        W[7] = W[5];
        W[6] = W[4];
        W[5] = W[3];
        W[4] = W[2];
        W[3] = W[1];
        W[2] = W[0];
        W[1] = T7;
        W[0] = T6;
        ks[i] = np.copy(W);
    return(ks);
    
def enc_one_round(p, k,round_const):
    #S = np.copy(p[::-1]);
    S = np.copy(p);
    #S = np.array([p[1],p[3],p[0],p[2]],dtype=np.uint32);
    #S = np.array([p[3],p[2],p[1],p[0]],dtype=np.uint16);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a0a0a, 3);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc00cc, 6);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0000ff00, 8);
    # #for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f000f, 4*i);
    # #for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x000f000f, 4*(i-1));
    # #for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f000f00, 4*(i-2));
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x0f0f0f0f, 4);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x0f0f0f0f, 4);
    # S[0], S[2] = SWAPMOVE_2(S[0], S[2], 0x0000ffff, 16);
    # S[1], S[3] = SWAPMOVE_2(S[1], S[3], 0x0000ffff, 16);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x00000f0f, 4);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x00000f0f, 4);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x0000ffff, 16);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x0000ffff, 16);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0000ff00, 8);
#     SWAPMOVE(state[0], state[0], 0x0000ff00, 8);
#     SWAPMOVE(state[1], state[1], 0x0000ff00, 8);
#     SWAPMOVE(state[2], state[2], 0x0000ff00, 8);
#     SWAPMOVE(state[3], state[3], 0x0000ff00, 8);
#    print("SWAPMOVE");
#    print([hex(S_i[0])[2:].zfill(4) for S_i in S ]);
    #===SubCells===#
    S[1] ^= S[0] & S[2];
    S[0] ^= S[1] & S[3];
    S[2] ^= S[0] | S[1];
    S[3] ^= S[2];
    S[1] ^= S[3];
    S[3] ^= 0xffffffff;
    S[2] ^= S[0] & S[1];
    
    T = np.copy(S[0]);
    S[0] = np.copy(S[3]);
    S[3] = np.copy(T);
    # print("SBOX_1")
    # print([hex(S_i[0])[2:].zfill(4) for S_i in S ]);
    #===PermBits===#
    S[0] = rowperm(S[0],0,3,2,1);
    S[1] = rowperm(S[1],1,0,3,2);
    S[2] = rowperm(S[2],2,1,0,3);
    S[3] = rowperm(S[3],3,2,1,0);

    #===AddRoundKey===#
    S[1] ^= k[6];
    S[0] ^= k[7];

    #Add round constant#
    S[3] ^= 0x8000 ^ round_const;
    
    return(S);



def encrypt(p, ks,r_start=1):
    #c0, c1, c2, c3 = p[1], p[3], p[0], p[2];
    #S = np.array([p[1],p[3],p[0],p[2]],dtype=np.uint32);
    S = np.copy(p[::-1]);
    #print("SWAPMOVE_1");
    #print([hex(S_i[0])[2:].zfill(8) for S_i in S ]);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a0a0a, 3);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc00cc, 6);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0000ff00, 8);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a, 3);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc, 6);
    for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f, 4*i);
    for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x00f0, 4*(i-1));
    for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f00, 4*(i-2));
    #print("SWAPMOVE_2");
    #print([hex(S_i[0])[2:].zfill(8) for S_i in S ]);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x0f0f0f0f, 4);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x0f0f0f0f, 4);
    # S[0], S[2] = SWAPMOVE_2(S[0], S[2], 0x0000ffff, 16);
    # S[1], S[3] = SWAPMOVE_2(S[1], S[3], 0x0000ffff, 16);
    # print("SWAPMOVE");
    # print([hex(S_i[0])[2:].zfill(4) for S_i in S ]);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x00000f0f, 4);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x00000f0f, 4);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x0000ffff, 16);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x0000ffff, 16);
    #for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0000ff00, 8);
    c0, c1, c2, c3 = S[0], S[1], S[2], S[3];
    # print(c0, c1, c2, c3 );
    for i in range(r_start-1,len(ks)):
        (c0, c1, c2, c3) = enc_one_round((c0, c1, c2, c3), ks[i], GIFT_RC[i]);
    #S = np.array([c0,c1,c2,c3],dtype=np.uint32);
    S = np.array([c0,c1,c2,c3],dtype=np.uint16);
    # print("After Encryption");
    # print([hex(S_i[0])[2:].zfill(4) for S_i in S ]);
    for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f00, 4*(i-2));
    for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x00f0, 4*(i-1));
    for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f, 4*i);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc, 6);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a, 3);
    # S[0], S[2] = SWAPMOVE_2(S[0], S[2], 0x0000ffff, 16);
    # S[1], S[3] = SWAPMOVE_2(S[1], S[3], 0x0000ffff, 16);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x0f0f0f0f, 4);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x0f0f0f0f, 4);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0000ff00, 8);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc00cc, 6);
    # for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a0a0a, 3);
    #for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0000ff00, 8);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x0000ffff, 16);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x0000ffff, 16);
    # S[0], S[1] = SWAPMOVE_2(S[0], S[1], 0x00000f0f, 4);
    # S[2], S[3] = SWAPMOVE_2(S[2], S[3], 0x00000f0f, 4);
    #print("After Encryption SWAOMOVE");
    #print([hex(S_i[0])[2:].zfill(4) for S_i in S ]);
    #return(S[2], S[0], S[3], S[1]);
    #return(S[0], S[1], S[2], S[3]);
    return(S[::-1]);

def rowperm_dec(S, B_pos):
    T=0x0000;
    for b in range(0,4):
        T |= ((S>>(4*b+0))&0x1)<<(4*0 + B_pos[b]);
        T |= ((S>>(4*b+1))&0x1)<<(4*1 + B_pos[b]);
        T |= ((S>>(4*b+2))&0x1)<<(4*2 + B_pos[b]);
        T |= ((S>>(4*b+3))&0x1)<<(4*3 + B_pos[b]);
    return(T); 

def dec_one_round(p, k,round_const):
    S = np.copy(p);
    #Add round constant#
    S[3] ^= 0x8000 ^ round_const;
    #===AddRoundKey===#
    S[1] ^= k[6];
    S[0] ^= k[7];
    #===PermBits===#
    S[0] = rowperm_dec(S[0],[0,3,2,1]);
    S[1] = rowperm_dec(S[1],[1,0,3,2]);
    S[2] = rowperm_dec(S[2],[2,1,0,3]);
    S[3] = rowperm_dec(S[3],[3,2,1,0]);
    #===SubCells===#
    T = np.copy(S[0]);
    S[0] = np.copy(S[3]);
    S[3] = np.copy(T);
    S[2] ^= S[0] & S[1];  
    S[3] ^= 0xffffffff;
    S[1] ^= S[3];
    S[3] ^= S[2];
    S[2] ^= S[0] | S[1]; 
    S[0] ^= S[1] & S[3];    
    S[1] ^= S[0] & S[2];
    
    return(S);

def decrypt(p, ks,r_start=28,r_end=0):
    
    S = np.copy(p[::-1]);

    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a, 3);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc, 6);
    for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f, 4*i);
    for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x00f0, 4*(i-1));
    for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f00, 4*(i-2));

    c0, c1, c2, c3 = S[0], S[1], S[2], S[3];

    for i in range(r_start-1,r_end-1,-1):
        (c0, c1, c2, c3) = dec_one_round((c0, c1, c2, c3), ks[i], GIFT_RC[i]);
    S = np.array([c0,c1,c2,c3],dtype=np.uint16);

    for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f00, 4*(i-2));
    for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x00f0, 4*(i-1));
    for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f, 4*i);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc, 6);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a, 3);
    return(S[::-1]);
# key_test = np.array([[48529, 48529], [29470, 29470], [46780, 46780], [10003, 10003], [41465, 41465], [63231, 63231], [51024, 51024], [17639, 17639]],dtype=np.uint16)
# #plain_text = np.array([[63693, 17070], [ 1784, 58987], [ 4727, 51159], [28008, 28338]],dtype=np.uint16)
# # #plain_text = np.array([[3293628274, 3293628274], [ 2057013885, 2057013885]],dtype=np.uint32)
# plain_text = np.array([[50256, 50256], [ 51058, 51058],  [ 31387, 31387],  [ 35453, 35453]],dtype=np.uint16)

# # #print([hex(plain_text_i[0])[2:].zfill(8) for plain_text_i in plain_text ]);
# print([hex(plain_text_i[0])[2:].zfill(4) for plain_text_i in plain_text ]);

# expanded_key = expand_key(key_test,28);
# print([hex(expanded_key_i[j][0])[2:].zfill(4) for expanded_key_i in expanded_key for j in range(0,len(expanded_key_i))]);
# # #ctdata0_0, ctdata0_1, ctdata1_0, ctdata1_1 = encrypt((plain_text[0], plain_text[1], plain_text[0], plain_text[1]), expanded_key);
# ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3 = encrypt((plain_text[0], plain_text[1], plain_text[2], plain_text[3]), expanded_key);
# print("cdata0");
# # #print([hex(ctdata0_0[0])[2:].zfill(4), hex(ctdata0_1[0])[2:], hex(ctdata1_0[0])[2:], hex(ctdata1_1[0])[2:]]);
# print([hex(ctdata0_0[0])[2:].zfill(4), hex(ctdata0_1[0])[2:], hex(ctdata0_2[0])[2:], hex(ctdata0_3[0])[2:]]);

 
def convert_to_binary_64_block(arr,WORD_SIZE=16,NO_OF_WORDS=8):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);


  
def make_train_data(n, nr, diff=(0x0000,0x0000,0x0000,0x0000),r_start=1,encrypt_data=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  # original multiple keys
  #keys = np.frombuffer(urandom(16*n),dtype=np.uint16).reshape(8,-1); # 16*8*n no of key bits
  #same keys
  keys = np.repeat(np.frombuffer(urandom(16),dtype=np.uint16).reshape(8,-1),n,axis=1);

  #plain0_0 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  #plain0_1 = np.frombuffer(urandom(4*n),dtype=np.uint32);
  

  plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);

   
  #plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #plain1l = plain0l ^ diff[0]; #plain1r = plain0r ^ diff[1];
  plain1_0 = plain0_0 ^ diff[0];
  plain1_1 = plain0_1 ^ diff[1];
  plain1_2 = plain0_2 ^ diff[2];
  plain1_3 = plain0_3 ^ diff[3];
  
  #plain1_0[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
  #plain1_1[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32);
  
  #original multiple difference for Y=0
  num_rand_samples = np.sum(Y==0);
  plain1_0[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1_1[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1_2[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1_3[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #same difference for Y=0
  # diff_1=[0,0,0,0];
  # diff_1[0] = int.from_bytes(urandom(2),byteorder='big', signed=False);
  # diff_1[1] = int.from_bytes(urandom(2),byteorder='big', signed=False);
  # diff_1[2] = int.from_bytes(urandom(2),byteorder='big', signed=False);
  # diff_1[3] = int.from_bytes(urandom(2),byteorder='big', signed=False);
  # plain1_0[Y==0] = plain1_0[Y==0] ^ diff_1[0];
  # plain1_1[Y==0] = plain1_1[Y==0] ^ diff_1[1];
  # plain1_2[Y==0] = plain1_2[Y==0] ^ diff_1[2];
  # plain1_3[Y==0] = plain1_3[Y==0] ^ diff_1[3];
  
  #plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #print(keys);
  #ks = list(map(gi.expand_key,list(zip(keys[0],keys[1])),[nr]*len(keys[0])));
  
  #ctdata0 = np.array(list(map(gi.encrypt,plain0,ks)),dtype=np.uint64);
  #ctdata0.astype(np.uint64);
  #ctdata1 = np.array(list(map(gi.encrypt,plain1,ks)),dtype=np.uint64);
  #ctdata1.astype(np.uint64);
  #ctdata0 = gi.encrypt(plain0, ks);
  #ctdata1= gi.encrypt(plain1, ks); np.binary_repr
  
  #print plain and key
  # print("plain0");
  #   #print(plain0_0, plain0_1,plain0_2, plain0_3);
  # print("".join([hex(plain_text_i[0])[2:].zfill(4) for plain_text_i in [plain0_0, plain0_1,plain0_2, plain0_3] ]));
  # print("plain1");
  #   #print(plain1_0, plain1_1,plain1_2, plain1_3);
  # print("".join([hex(plain_text_i[0])[2:].zfill(4) for plain_text_i in [plain1_0, plain1_1,plain1_2, plain1_3] ]));

  
  
  #print(ks);
  # print("ctdata0");
  # print(ctdata0);
  # print("ctdata1");
  # print(ctdata1);
  # cdata without XOR
  # ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3 = encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks);
  # ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks);
  # X = convert_to_binary_64_block([ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3, ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3],16,8);
  #print(X,Y);
  #return(X,Y);
  #cipher text with XOR
  if(encrypt_data==1):
      ks = expand_key(keys, (r_start-1) + nr);
      cdata0= encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
      cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);
  else:
      ks = expand_key(keys,r_start);
      cdata0 = decrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks, r_start,r_end = r_start-nr);
      cdata1 = decrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks, r_start,r_end = r_start-nr);

  X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,4);
  
  # plain00 = decrypt(cdata0, ks, (r_start-1) + nr,r_end = r_start-1);
  # plain11 = decrypt(cdata1, ks, (r_start-1) + nr,r_end = r_start-1);
  # print("key_opt");
  # print("".join([hex(expanded_key_i[j][0])[2:].zfill(4) for expanded_key_i in ks for j in range(0,len(expanded_key_i))]));
  # #print ciphertext
  # print("cdata0");
  # print("".join([hex(cipher_text_i[0])[2:].zfill(4) for cipher_text_i in cdata0 ]));
  # print("cdata1");
  # print("".join([hex(cipher_text_i[0])[2:].zfill(4) for cipher_text_i in cdata1]));
  
  # print("plain00");
  # print("".join([hex(cipher_text_i[0])[2:].zfill(4) for cipher_text_i in plain00 ]));
  # print("plain11");
  # print("".join([hex(cipher_text_i[0])[2:].zfill(4) for cipher_text_i in plain11]));
  # # print([ctdata0_0, ctdata0_1, ctdata0_2,ctdata0_3]);
  # # print([ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3]);
  
  return (X,Y);
#make_train_data(1,28,r_start=1,encrypt_data=1)
#make_train_data(1,5,r_start=28,encrypt_data=0)
#sys.exit() 
def make_train_data_no_random(n, nr, diff=(0x0000,0x0000,0x0000,0x0000),output_Y=1,r_start=1,encrypt_data=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8);
  if (output_Y==0):
    Y = (Y & 0);
  elif (output_Y==1):
    Y = (Y & 1) | 1;
  #keys = np.frombuffer(urandom(16*n),dtype=np.uint16).reshape(8,-1); # 16*8*n no of key bits
  # origins multiple keys
  #keys = np.frombuffer(urandom(16*n),dtype=np.uint16).reshape(8,-1); # 16*8*n no of key bits
  #same keys
  keys = np.repeat(np.frombuffer(urandom(16),dtype=np.uint16).reshape(8,-1),n,axis=1);

  plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);

  plain1_0 = plain0_0 ^ diff[0];
  plain1_1 = plain0_1 ^ diff[1];
  plain1_2 = plain0_2 ^ diff[2];
  plain1_3 = plain0_3 ^ diff[3];

  #ks = expand_key(keys, (r_start-1)+nr);
  #cipher text without XOR
  # ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3 = encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks);
  # ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks);
  # X = convert_to_binary_64_block([ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3, ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3],16,8);
  #cipher text with XOR
  #cdata0= encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
  #cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);
  if(encrypt_data==1):
    ks = expand_key(keys, (r_start-1) + nr);
    cdata0= encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
    cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);
  else:
    ks = expand_key(keys,r_start);
    cdata0 = decrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks, r_start,r_end = r_start-nr);
    cdata1 = decrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks, r_start,r_end = r_start-nr);
  X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,4);
  return (X,Y);
# Without Comments
# def convert_to_binary_64_block(arr,WORD_SIZE=16):
#   X = np.zeros((8 * WORD_SIZE,len(arr[0])),dtype=np.uint8);
#   for i in range(8 * WORD_SIZE):
#     index = i // WORD_SIZE;
#     offset = WORD_SIZE - (i % WORD_SIZE) - 1;
#     X[i] = (arr[index] >> offset) & 1;
#   X = X.transpose();
#   return(X);

# def make_train_data_gift_opt(n, nr, diff=(0x0000,0x0000,0x0000,0x0000)):
#   Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
#   keys = np.frombuffer(urandom(16*n),dtype=np.uint16).reshape(8,-1); # 16*8*n no of key bits
#   plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);
#   plain1_0 = plain0_0 ^ diff[0];
#   plain1_1 = plain0_1 ^ diff[1];
#   plain1_2 = plain0_2 ^ diff[2];
#   plain1_3 = plain0_3 ^ diff[3];
#   num_rand_samples = np.sum(Y==0);
#   plain1_0[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   plain1_1[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   plain1_2[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   plain1_3[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#   ks = expand_key(keys, nr);
#   ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3 = encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks);
#   ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks);
#   X = convert_to_binary_64_block([ctdata0_0, ctdata0_1, ctdata0_2, ctdata0_3, ctdata1_0, ctdata1_1, ctdata1_2, ctdata1_3],16);
#   return (X,Y);











