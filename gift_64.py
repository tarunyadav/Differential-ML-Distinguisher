#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GIFT-64 Implementation
"""
Created on Sun Jun 14 03:24:45 2020

@author: tarunyadav
"""
import numpy as np
from os import urandom
# import sys
GIFT_RC = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
    0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
    0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
    0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
    0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
    0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28,
    0x10, 0x20];

def SWAPMOVE_1(X,M,n):
    return( (X ^ ((X ^ (X >> n)) & M)) ^ (((X ^ (X >> n)) & M)<< n));

def SWAPMOVE_2(A,B,M,n):
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
    W = np.copy(k);
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
    S = np.copy(p);
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
    S = np.copy(p[::-1]);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a, 3);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc, 6);
    for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f, 4*i);
    for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x00f0, 4*(i-1));
    for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f00, 4*(i-2));
    c0, c1, c2, c3 = S[0], S[1], S[2], S[3];
    for i in range(r_start-1,len(ks)):
        (c0, c1, c2, c3) = enc_one_round((c0, c1, c2, c3), ks[i], GIFT_RC[i]);
    S = np.array([c0,c1,c2,c3],dtype=np.uint16);
    for i in range(3,4): S[2], S[i] = SWAPMOVE_2(S[2], S[i], 0x0f00, 4*(i-2));
    for i in range(2,4): S[1], S[i] = SWAPMOVE_2(S[1], S[i], 0x00f0, 4*(i-1));
    for i in range(1,4): S[0], S[i] = SWAPMOVE_2(S[0], S[i], 0x000f, 4*i);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x00cc, 6);
    for i in range(0,4): S[i] = SWAPMOVE_1(S[i], 0x0a0a, 3);
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

def convert_to_binary_64_block(arr,WORD_SIZE=16,NO_OF_WORDS=8):
  X = np.zeros((NO_OF_WORDS * WORD_SIZE,len(arr[0])),dtype=np.uint8);
  for i in range(NO_OF_WORDS * WORD_SIZE):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - (i % WORD_SIZE) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);


def make_train_data(n, nr, diff=(0x0000,0x0000,0x0000,0x0000),r_start=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.repeat(np.frombuffer(urandom(16),dtype=np.uint16).reshape(8,-1),n,axis=1);
  plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1_0 = plain0_0 ^ diff[0];
  plain1_1 = plain0_1 ^ diff[1];
  plain1_2 = plain0_2 ^ diff[2];
  plain1_3 = plain0_3 ^ diff[3];
  num_rand_samples = np.sum(Y==0);
  plain1_0[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1_1[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1_2[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1_3[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);

  ks = expand_key(keys, (r_start-1) + nr);
  cdata0= encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
  cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);

  X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,4);
  return (X,Y);

def make_train_data_no_random(n, nr, diff=(0x0000,0x0000,0x0000,0x0000),output_Y=1,r_start=1):
  Y = np.frombuffer(urandom(n), dtype=np.uint8);
  if (output_Y==0):
    Y = (Y & 0);
  elif (output_Y==1):
    Y = (Y & 1) | 1;
  keys = np.repeat(np.frombuffer(urandom(16),dtype=np.uint16).reshape(8,-1),n,axis=1);
  plain0_0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_2 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0_3 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1_0 = plain0_0 ^ diff[0];
  plain1_1 = plain0_1 ^ diff[1];
  plain1_2 = plain0_2 ^ diff[2];
  plain1_3 = plain0_3 ^ diff[3];

  ks = expand_key(keys, (r_start-1) + nr);
  cdata0= encrypt((plain0_0, plain0_1, plain0_2, plain0_3), ks,r_start);
  cdata1 = encrypt((plain1_0, plain1_1, plain1_2, plain1_3), ks,r_start);

  X = convert_to_binary_64_block(np.array(cdata0^cdata1),16,4);
  return (X,Y);












