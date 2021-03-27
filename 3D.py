import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy.io
import os
np.set_printoptions(threshold=np.inf)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
data = scipy.io.loadmat(r'D:/FocusShaping/optimization.mat')
x_data = np.array(data['X'])
can1 = np.array(data['can1'])
can2 = np.array(data['can2'])
can3 = np.array(data['can3'])
nor = np.array(data['nor'])
y_data = np.array(data['Y'])
y_data = np.swapaxes(y_data,0,-1)
y_data = np.expand_dims(y_data,0)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))


print(np.shape(x_data))
print(np.shape(y_data))
print(np.shape(can1))
print(np.shape(can2))


BATCH_SIZE = 1
image_count = x_data.shape[0]

dataset = dataset.batch(BATCH_SIZE)
initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
pointsize = np.size(can1,0)
def gen_model():
    seed = layers.Input(shape=((1)))
    x = layers.Dense(300,activation='relu',kernel_initializer=initial)(seed)
    x = layers.Dense(300, activation='relu',kernel_initializer=initial)(x)
    x = layers.Dense(300, activation='relu',kernel_initializer=initial)(x)
    x = layers.Dense(25, activation='tanh', kernel_initializer=initial)(x)
    model = tf.keras.Model(inputs=seed, outputs=x)
    return model
generator = gen_model()


def physics(y_pred):
    y_com = tf.transpose(tf.complex(y_pred,0.0))
    a = tf.matmul(can1, y_com)
    b = tf.matmul(can2, y_com)
    c = tf.matmul(can3, y_com)

    I1 = tf.square(tf.abs(a))
    I2 = tf.square(tf.abs(b)) + tf.square(tf.abs(c))
    I = tf.concat([I1, I2], axis=0)
    maxvalue = tf.reduce_max(I)
    I = I / maxvalue
    return  I, maxvalue

def my_loss(I, y_true, maxvalue):
    I_loss = tf.reduce_mean(tf.square(I - tf.squeeze(y_true)))
    return  I_loss + 1e-5/(maxvalue/nor)

generator_optimizer = tf.keras.optimizers.Adam()

#checkpoint = tf.train.Checkpoint(mygen=generator,
                                 #mygenerator_optimizer=generator_optimizer)

@tf.function
def train_step1(x_batch, y_batch):
    with tf.GradientTape() as r_tape:
        tt = generator((x_batch), training=True)  #
        I, maxvalue = physics(tt/tf.reduce_max(tf.abs(tt)))  #
        gen_loss = my_loss(tf.transpose(I), y_batch, maxvalue)

    gradients_of_generator = r_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss,  tt, I, maxvalue




gen_loss_results = []
#I_results = []
T_results = []
maxvalue_results = []
def train(dataset, epochs):
    time_start = time.time()
    for epoch in range(epochs):
        for x_data, y_data in dataset:
            #print('epoch ', epoch+1)
            gen_loss, T, I, maxvalue = train_step1(x_data, y_data)
            #print("gloss：%s" % (gen_loss.numpy()))
            #print(I.numpy())
            #print(maxvalue.numpy()/nor)
            #print(y_data)
            gen_loss_results.append(gen_loss.numpy())
            #I_results.append(I.numpy())
            T_results.append(T.numpy())
            maxvalue_results.append(maxvalue.numpy())
    #checkpoint.save('D:/mytf/optimization/mse/model.ckpt')
    time_end = time.time()
    print(gen_loss_results[-1])
    print(maxvalue_results[-1]/nor)
    print('totally cost', time_end - time_start)
    scipy.io.savemat('D:/FocusShaping/time.mat', {'Time': np.squeeze(time_end - time_start)})

EPOCHS = 5000
#check = tf.train.latest_checkpoint('D:/FocusShaping')
#checkpoint.restore(check)
#print('retored from checkpoint...')
train(dataset, EPOCHS)


scipy.io.savemat('D:/FocusShaping/gloss.mat', {'gloss': np.squeeze(gen_loss_results)})
#scipy.io.savemat('D:/FocusShaping/I.mat', {'I': np.squeeze(I_results)})
scipy.io.savemat('D:/FocusShaping/T.mat', {'T': np.squeeze(T_results)})
#scipy.io.savemat('D:/FocusShaping/time.mat', {'Time': np.squeeze(time_end-time_start)})
scipy.io.savemat('D:/FocusShaping/maxvalue.mat', {'maxvalue': np.squeeze(maxvalue_results)})