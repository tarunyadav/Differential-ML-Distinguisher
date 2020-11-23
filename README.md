# Differential-ML-Distinguisher
Source Code for Differential-ML Distinguisher

**For SPECK32:**\
```predict_test.py SPECK_9_Round_Model_(34058,38176)_diff.h5 1 2 20 9 1 6 .79 73100 fix_diff 0x0211 0x0a04``` &#8594; Expected Prediction 50\
```predict_test.py SPECK_9_Round_Model_(34058,38176)_diff.h5 1 2 20 9 1 6 .79 73100 random_diff``` &#8594; Expected Prediction 0


**For SIMON32:**\
```predict_test.py simon SIMON_12_Round_Model_(7425,16896)_diff.h5 1 2 22 12 1 7 .57 656300 fix_diff 0x0400 0x1900``` &#8594; Expected Prediction 50\
```predict_test.py simon SIMON_12_Round_Model_(7425,16896)_diff.h5 1 2 22 12 1 7 .57 656300 random_diff``` &#8594; Expected Prediction 0


**For GIFT64:**\
```predict_test.py GIFT_64 GIFT64_8_Round_Model_(68,0,17,0)_diff.h5 1 2 20 8 1 4 .65 103750 fix_diff 0x0000 0x0000 0x0000 0x000a``` &#8594; Expected Prediction 50\
```predict_test.py GIFT_64 GIFT64_8_Round_Model_(68,0,17,0)_diff.h5 1 2 20 8 1 4 .65 103750 random_diff``` &#8594; Expected Prediction 0

*Sample size for all tests is 50. For first test in all cases expected prediction is 50(all ciphertext pairs belongs to correct difference distinguished correctly). For second test expected prediction is 0(all ciphertext pairs not belonging to correct difference distinguished correctly.) Therefore, in total if prediction in first test is 50 and in second test is 0 then accuracy is 100%(50+(50-0)).*
