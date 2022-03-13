import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time, os, glob
import PIL, cv2

img_files = glob.glob('./data_enlighten/*.JPG')
img_files = [(file, int(file[32:34]), int(file[36:38])) for file in img_files if file[32:34]!='GT']

## deviding files manually
img_files = sorted(img_files, key=lambda x: (x[1], x[2]))
img_files.reverse()

LLs     = img_files[:-1] # low-light images
HLs     = img_files[1:]  # high-light images
dataset = pd.DataFrame({'LL':LLs, 'HL':HLs})

dataset['discard'] = 0
for i in range(len(dataset)):
    if dataset.iloc[i].LL[1] != dataset.iloc[i].HL[1]:
        dataset.discard.iloc[i] = 1

dataset = dataset.loc[dataset.discard==0] # pairs(low-light, high-light)

img_w, img_h           = 160, 160
patch_size             = 8
START_TOKEN, END_TOKEN = [256], [257] ## 255까지 RGB값이므로, RGB값과 관련없는 숫자 2개를 시작토큰과 종료토큰에 부여.
padding_size           = 0

def ImgDivision(file):
    imgs  = []
    img   = cv2.imread(file)
    img   = cv2.resize(img, (img_w, img_h))
    for i in range(0,int(img_w/patch_size)):
        for j in range(0,int(img_h/patch_size)):
            imgs.append(np.array(START_TOKEN+\
                                 list(img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].flatten())+\
                                 [0]*padding_size+END_TOKEN))
    return imgs


LLs, HLs = [], []
for i in range(len(dataset)):
    LLs_ = ImgDivision(dataset.iloc[i].LL[0])
    HLs_ = ImgDivision(dataset.iloc[i].HL[0])
    LLs  += LLs_
    HLs  += HLs_
    
img_dataset = pd.DataFrame({'LL':LLs, 'HL':HLs})

## Transformer code (Inspired by https://github.com/ukairia777/tensorflow-transformer)
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
        position = tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])

        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
def scaled_dot_product_attention(query, key, value, mask):
  # query size   : (batch_size, num_heads, sentence length of query, d_model/num_heads)
  # key size     : (batch_size, num_heads, sentence length of key, d_model/num_heads)
  # value size   : (batch_size, num_heads, sentence length of value, d_model/num_heads)
  # padding_mask : (batch_size, 1, 1, sentence length of key)

  # Q*K, Attention Score matrix
    matmul_qk = tf.matmul(query, key, transpose_b=True)

  # Scaling
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

  # Masking. Input a very small negative value to a masked position in the Attention Score Matrix.
  # It will be 0 after computing by a softmax function
    if mask is not None:
        logits += (mask * -1e9)

  # attention weight : (batch_size, num_heads, sentence length of query, sentence length of key)
    attention_weights = tf.nn.softmax(logits, axis=-1)

  # output : (batch_size, num_heads, sentence length of query, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # Defining dense layers for each WQ, WK, WV
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # Defining a dense layer of WO
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sentence length of key)
    return mask[:, tf.newaxis, tf.newaxis, :]

def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': padding_mask # using a padding mask
      })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

# Modified version
def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': look_ahead_mask # look ahead mask
      })

    attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
          'mask': padding_mask # padding mask
      })

    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

    inputs = tf.keras.Input(shape=(None,), name="inputs")

    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
sample_learning_rate = CustomSchedule(d_model=128)

NUM_LAYERS  = 2
D_MODEL     = 512
NUM_HEADS   = 8
DFF         = 512
DROPOUT     = 0.1

BATCH_SIZE  = 32
EPOCHS      = 10
BUFFER_SIZE = 20000
MAX_LENGTH  = (patch_size**2)*3 + padding_size + 2

dataset_dec_inputs = np.array([vec[:-1] for vec in img_dataset.HL], dtype=np.int32)
dataset_outputs    = np.array([vec[1:]  for vec in img_dataset.HL], dtype=np.int32)
dataset_inputs     = np.array([vec  for vec in img_dataset.LL], dtype=np.int32)

training_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': dataset_inputs,
        'dec_inputs': dataset_dec_inputs
    },
    {
        'outputs': dataset_outputs
    },
))

training_dataset = training_dataset.cache()
training_dataset = training_dataset.shuffle(BUFFER_SIZE)
training_dataset = training_dataset.batch(BATCH_SIZE)
training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()

model = transformer(
                    vocab_size=(255+3), # the color range 255 + 2 tokens + 1 buffer
                    num_layers=NUM_LAYERS,
                    dff=DFF,
                    d_model=D_MODEL,
                    num_heads=NUM_HEADS,
                    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.fit(training_dataset, epochs=EPOCHS)

img_files = glob.glob('./data_enlighten/*.JPG')
img_files = [(file, str(file[32:34]), int(file[36:38])) for file in img_files if file[32:34]=='GT']
img_files = sorted(img_files, key=lambda x: (x[1], x[2]))
img_files.reverse()

LLs          = img_files[:-1]
HLs          = img_files[1:]
eval_dataset = pd.DataFrame({'LL':LLs, 'HL':HLs})

eval_file = eval_dataset.LL[0][0]
answ_file = eval_dataset.HL[0][0]

def Enlighten(file):
    canvas = np.zeros(shape=(img_w,img_h,3))
    img    = cv2.imread(file)
    img    = cv2.resize(img, (img_w, img_h))
    for i in range(0,int(img_w/patch_size)):
        for j in range(0,int(img_w/patch_size)):
            crop = np.array(START_TOKEN+\
                            list(img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].flatten())+\
                            [0]*padding_size+END_TOKEN)
            crop = tf.expand_dims(crop, 0)
            
            pred = tf.expand_dims(START_TOKEN,0)
            for _ in range(MAX_LENGTH-padding_size-2):
                predictions  = new_model(inputs=[crop, pred], training=False) # (1, 1, 258)
                predictions  = predictions[:, -1:, :]
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                if tf.equal(predicted_id, END_TOKEN[0]):
                    break
                pred = tf.concat([pred, predicted_id], axis=-1)
            
            pred = tf.squeeze(pred, axis=0)
            pred = np.reshape(pred[1:], (5,5,3))
            
            canvas[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred
        print("-"*5, i, "/", int(img_w/patch_size), "-"*10)
    return img, canvas

eval_img, pred_img = Enlighten(eval_file)

pred_img_uint8     = pred_img.astype(np.uint8)

plt.subplot(3,1,1)
plt.imshow(eval_img)
plt.subplot(3,1,2)
plt.imshow(pred_img_uint8)

answ_img = cv2.imread(answ_file)
answ_img = cv2.resize(answ_img, (160,160))
plt.subplot(3,1,3)
plt.imshow(answ_img)