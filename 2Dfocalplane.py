import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
data = scipy.io.loadmat(r'D:/FocusShaping/optimization.mat')#
x_data = np.array(data['X'])
can1 = np.array(data['can1'])
can2 = np.array(data['can2'])
y_data = np.array(data['Y'])
nor = np.array(data['nor'])
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))


print(np.shape(x_data))#
print(np.shape(y_data))
print(np.shape(can1))
print(np.shape(can2))


BATCH_SIZE = 1#
image_count = x_data.shape[0]
dataset = dataset.batch(BATCH_SIZE)

pointsize = np.size(can1,0)
initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)

def gen_model():
    seed = layers.Input(shape=((1)))
    x = layers.Dense(300, activation='relu', kernel_initializer = initial)(seed)
    x = layers.Dense(300, activation='relu', kernel_initializer = initial)(x)
    x = layers.Dense(300, activation='relu', kernel_initializer = initial)(x)
    x = layers.Dense(25, activation='tanh', kernel_initializer = initial)(x)
    model = tf.keras.Model(inputs=seed, outputs=x)
    return model#
generator = gen_model()

def physics(y_pred):
    a = tf.matmul(can1, tf.transpose(y_pred))
    b = tf.matmul(can2, tf.transpose(y_pred))
    I = tf.square(a) + tf.square(b)
    maxvalue = tf.squeeze(I[0,:])
    I = I / maxvalue
    return  I, maxvalue

def my_loss(I, y_true, maxvalue):
    I_loss1 = tf.reduce_mean(tf.square(I[:,0:np.size(y_true,-1)] - y_true))
    I_loss2 = tf.reduce_mean(tf.square(tf.reduce_max(I[:,np.size(y_true,-1):]) - 0.2))
    return  I_loss1 + 0.0* I_loss2 + 1e-4/(maxvalue/nor)

generator_optimizer = tf.keras.optimizers.Adam()

#checkpoint = tf.train.Checkpoint(mygen=generator,
                                 #mygenerator_optimizer=generator_optimizer)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as r_tape:
        tt = generator((x_batch), training=True)#
        I, maxvalue = physics(tt/tf.reduce_max(tf.abs(tt)))#
        gen_loss = my_loss(tf.transpose(I), y_batch, maxvalue)#
    gradients_of_generator = r_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss, I, maxvalue, tt

gen_loss_results = []
I_results = []
tt_results = []
maxvalue_results = []
#sidelobe_results = []

def train(dataset, epochs):
    time_start = time.time()  #
    for epoch in range(epochs):
        for x_data, y_data in dataset:
            #print('epoch ', epoch+1)
            gen_loss, I,  maxvalue,tt = train_step(x_data, y_data)
            #print("gloss：%s" % (gen_loss.numpy()))
            #print(I[:,0:np.size(y_data,-1)].numpy())
            #print('SR',maxvalue/nor)
            #print('2',tf.reduce_max(I[:,np.size(y_data,-1):]))
            gen_loss_results.append(gen_loss.numpy())
            #I_results.append(I[:,0:np.size(y_data,-1)].numpy())
            tt_results.append(tt.numpy())
            maxvalue_results.append(maxvalue.numpy())
            #sidelobe_results.append(tf.reduce_max(I[:,np.size(y_data,-1):]).numpy())
    #checkpoint.save('D:/FocusShaping/model.ckpt')
    #plt.plot(np.squeeze(tt))
    #plt.show()
    time_end = time.time()
    print('totally cost', time_end - time_start)
    scipy.io.savemat('D:/mytf/optimization/mse/time.mat', {'Time': np.squeeze(time_end - time_start)})
EPOCHS = 1000#
#check1 = tf.train.latest_checkpoint('D:/FocusShaping')
#checkpoint.restore(check1)
#print('retored from checkpoint...')
train(dataset, EPOCHS)

print('loss', gen_loss_results[-1])
print('SR', maxvalue_results[-1]/nor)

scipy.io.savemat('D:/FocusShaping/gloss.mat', {'gloss': np.squeeze(gen_loss_results)})
#scipy.io.savemat('D:/FocusShaping/I.mat', {'I': np.squeeze(I_results)})
scipy.io.savemat('D:/FocusShaping/maxvalue.mat', {'maxvalue': np.squeeze(maxvalue_results/nor)})
scipy.io.savemat('D:/FocusShaping/T.mat', {'T': np.squeeze(tt_results)})
#scipy.io.savemat('D:/FocusShaping/sidelobe.mat', {'sidelobe': np.squeeze(sidelobe_results)})
