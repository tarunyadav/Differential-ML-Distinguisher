# Differential-ML-Distinguisher
## Source Code for Differential-ML Distinguisher ###

There are 11 files in this repository.

**Cipher Implementartion:**
* speck.py
* simon.py
* gift_64.py

**Data Generation, Training and Predictions:**
* networks.py
* train_distinguisher.py
* predictions.py

**Trained Model Files:**
* SPECK_3_Round_Model.h5
* SIMON_5_Round_Model.h5
* GIFT64_4_Round_Model.h5

**Differential_ML_Distinguisher.ipynb** (Notebook for execution)\
**README.MD**

## Instructions for Execution (https://colab.research.google.com/)
### To model ML Distinguisher ###
**For SPECK32:**\
```%run train_distinguisher.py speck 10 3 2 1024 2 25 2 22 0x850a 0x9520 fresh``` &#8594; Expected Accuracy .79\
**For SIMON32:**\
```%run train_distinguisher.py simon 10 5 2 1024 2 25 2 22 0x1d01 0x4200 fresh``` &#8594; Expected Accuracy .57\
**For GIFT64:**\
```%run train_distinguisher.py GIFT_64 10 4 2 1024 2 25 2 22 0x0044 0x0000 0x0011 0x0000 fresh``` &#8594; Expected Accuracy .65

***Arguments:***\
train_distinguisher.py cipher num_epochs num_rounds depth neurons data_train(2<sup>x</sup>) data_test(2<sup>y</sup>) difference(Hex) pre_trained_model/fresh


### To get predictions using Differential-ML Distinguisher ###
**For SPECK32:**\
```%run predictions.py speck SPECK_3_Round_Model.h5 2 20 9 1 6 .79 73100 fix_diff 0x0211 0x0a04``` &#8594; Expected Prediction 50\
```%run predictions.py speck SPECK_3_Round_Model.h5 2 20 9 1 6 .79 73100 random_diff``` &#8594; Expected Prediction 0


**For SIMON32:**\
```%run predictions.py simon SIMON_5_Round_Model.h5 2 22 12 1 7 .57 656300 fix_diff 0x0400 0x1900``` &#8594; Expected Prediction 50\
```%run predictions.py simon SIMON_5_Round_Model.h5 2 22 12 1 7 .57 656300 random_diff``` &#8594; Expected Prediction 0


**For GIFT64:**\
```%run predictions.py GIFT_64 GIFT64_4_Round_Model.h5 2 20 8 1 4 .65 103750 fix_diff 0x0000 0x0000 0x0000 0x000a``` &#8594; Expected Prediction 50\
```%run predictions.py GIFT_64 GIFT64_4_Round_Model.h5 2 20 8 1 4 .65 103750 random_diff``` &#8594; Expected Prediction 0

***Arguments:***\
predictions.py cipher trained_model data_complexity(β) num_round start_round mid_round probability_cutoff(T) cutoff_threshold(C<sub>T</sub>) fix_diff/random_diff difference(Hex)

*Sample size for all tests is 50. For first test in all cases expected prediction is 50(all ciphertext pairs belongs to correct difference distinguished correctly). For second test expected prediction is 0(all ciphertext pairs not belonging to correct difference distinguished correctly.) Therefore, in total if prediction in first test is 50 and in second test is 0 then accuracy is 100%(50+(50-0)).*

## Acknowledgement ##
1. Gohr A. (2019) Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning. In: Boldyreva A., Micciancio D. (eds) Advances in Cryptology – CRYPTO 2019. CRYPTO 2019. Lecture Notes in Computer Science, vol 11693. Springer, Cham. https://doi.org/10.1007/978-3-030-26951-7_6 (https://github.com/agohr/deep_speck)
2. Baksi A., Breier J., Dong X., Yi C.:  Machine Learning Assisted DifferentialDistinguishers For Lightweight Ciphers. https://eprint.iacr.org/2020/571, (2020)
3. GIFT_64 differential characteristic can be verified using https://github.com/zhuby12/MILP-basedModel.