from keras.optimizers import Adamax

NB_EPOCH = 30  # Number of training epoch
BATCH_SIZE = 128  # Batch size
VERBOSE = 2  # Output display mode
LENGTH = 10000  # Packet sequence length
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # Optimizer
NB_CLASSES = 95  # number of outputs = number of classes
INPUT_SHAPE = (LENGTH, 1)
