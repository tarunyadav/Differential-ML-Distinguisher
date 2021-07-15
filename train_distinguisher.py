import networks as nets
import sys

if ("GIFT_64" in sys.argv[1]):
    nets.train_distinguisher(cipher=sys.argv[1],num_epochs=int(sys.argv[2]),num_rounds=int(sys.argv[3]),depth=int(sys.argv[4]),neurons=int(sys.argv[5]),data_train=int(sys.argv[6])**int(sys.argv[7]),data_test=int(sys.argv[8])**int(sys.argv[9]),difference=(int(sys.argv[10],16),int(sys.argv[11],16),int(sys.argv[12],16),int(sys.argv[13],16)),pre_trained_model=sys.argv[14]);
else:
    nets.train_distinguisher(cipher=sys.argv[1],num_epochs=int(sys.argv[2]),num_rounds=int(sys.argv[3]),depth=int(sys.argv[4]),neurons=int(sys.argv[5]),data_train=int(sys.argv[6])**int(sys.argv[7]),data_test=int(sys.argv[8])**int(sys.argv[9]),difference=(int(sys.argv[10],16),int(sys.argv[11],16)),pre_trained_model=sys.argv[12]);