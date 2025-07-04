import tensorflow as tf
from loss.loss import *
from model.simclr_model import *
from byol_tf import *
from simsiam_tf import *
from moco_tf import *
import numpy as np

def pre_train_simclr(data_1, data_2, batch_size=128, epochs=10, learning_rate=0.0001, pre_model = "ResNet", seed=42):
    # 데이터 타입 변환 및 확인
    data_1 = tf.cast(data_1, tf.float32)  # (1000, 8, 8, 20)
    data_2 = tf.cast(data_2, tf.float32)  # (1000, 8, 8, 20)
    num_samples = data_1.shape[0]
    steps_per_epoch = num_samples // batch_size
    
    # 모델 초기화
    model = build_simclr_model(pre_model = pre_model)
    # model = build_cnn_model(input_shape=data_1.shape[1:])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 학습 스텝 정의
    def train_step(pair_1, pair_2):
        with tf.GradientTape() as tape:
            z1 = model(pair_1, training=True)
            z2 = model(pair_2, training=True)
            loss = nt_xent(z1, z2)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    data = np.concatenate((data_1, data_2), axis=3)

    np.random.seed(seed)
    # 학습 루프
    for epoch in range(epochs):
        np.random.shuffle(data)
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # 배치 단위로 학습
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]
            
            loss = train_step(batch_data[:,:,:,:20],batch_data[:,:,:,20:])
            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return model

def pre_train_byol(data_1, data_2, batch_size=128, epochs=10, learning_rate=0.0001, pre_model = "ResNet", seed=42):
    # 데이터 타입 변환 및 확인
    data_1 = tf.cast(data_1, tf.float32)  # (1000, 8, 8, 20)
    data_2 = tf.cast(data_2, tf.float32)  # (1000, 8, 8, 20)
    num_samples = data_1.shape[0]
    steps_per_epoch = num_samples // batch_size
    
    input_shape = data_1.shape[1:]  # (18, 18, 20)
    # 모델 초기화
    if pre_model=='ResNet':
        base_encoder = build_encoder(input_shape, layer_name="encoder")
    elif pre_model=='CNN':
        base_encoder = lsm_CNN(input_shape)
    elif pre_model=='ViT':
        base_encoder = vit_model(input_shape)

    image_size = data_1.shape[1:3]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    byol_kwargs = {
        'projection_size': 128,
        'projection_hidden_size': 1024,
        'moving_average_decay': 0.99
    }
    
    byol = BYOL(
            base_encoder,
            image_size=image_size,
            **byol_kwargs
        )

    # 학습 스텝 정의
    def train_step(x1, x2):
        with tf.GradientTape() as tape:
            loss = byol(x1, x2, training=True)
        gradients = tape.gradient(loss, byol.trainable_variables)
        optimizer.apply_gradients(zip(gradients, byol.trainable_variables))
        byol.update_moving_average()
        return loss

    data = np.concatenate((data_1, data_2), axis=3)
    np.random.seed(seed)
    # 학습 루프
    for epoch in range(epochs):
        np.random.shuffle(data)
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # 배치 단위로 학습
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]
            
            loss = train_step(x1 = batch_data[:,:,:,:20],x2 = batch_data[:,:,:,20:])

            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return byol

def pre_train_simsiam(data_1, data_2, batch_size=128, epochs=10, learning_rate=0.0001, pre_model = "ResNet", seed=42):
    # 데이터 타입 변환 및 확인
    data_1 = tf.cast(data_1, tf.float32)  # (1000, 8, 8, 20)
    data_2 = tf.cast(data_2, tf.float32)  # (1000, 8, 8, 20)
    num_samples = data_1.shape[0]
    steps_per_epoch = num_samples // batch_size
    
    input_shape = data_1.shape[1:]  # (18, 18, 20)

    # 모델 초기화
    if pre_model=='ResNet':
        base_encoder = build_encoder(input_shape, layer_name="encoder")
    elif pre_model=='CNN':
        base_encoder = lsm_CNN(input_shape)
    elif pre_model=='ViT':
        base_encoder = vit_model(input_shape)

    image_size = data_1.shape[1:3]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    simsiam_kwargs = {
        'projection_size': 128,
        'projection_hidden_size': 1024,
        'predictor_hidden_size': 512
    }
    
    simsiam = SimSiam(
            base_encoder,
            image_size=image_size,
            **simsiam_kwargs
        )

    # 학습 스텝 정의
    def train_step(x1, x2):
        with tf.GradientTape() as tape:
            loss = simsiam(x1, x2, training=True)
            reg_loss = tf.add_n(simsiam.losses) if simsiam.losses else 0.0
            total_loss = loss + reg_loss
        gradients = tape.gradient(loss, simsiam.trainable_variables)
        optimizer.apply_gradients(zip(gradients, simsiam.trainable_variables))
        return loss

    data = np.concatenate((data_1, data_2), axis=3)
    
    np.random.seed(seed)
    # 학습 루프
    for epoch in range(epochs):
        np.random.shuffle(data)
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # 배치 단위로 학습
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]
            
            loss = train_step(x1=batch_data[:,:,:,:20], x2=batch_data[:,:,:,20:])

            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return simsiam

def pre_train_moco(data_1, data_2, batch_size=128, epochs=10, learning_rate=0.0001, pre_model = "ResNet", seed=42):
    # 데이터 타입 변환 및 확인
    data_1 = tf.cast(data_1, tf.float32)  # (1000, 8, 8, 20)
    data_2 = tf.cast(data_2, tf.float32)  # (1000, 8, 8, 20)
    num_samples = data_1.shape[0]
    steps_per_epoch = num_samples // batch_size
    
    input_shape = data_1.shape[1:]  # (18, 18, 20)

    # 모델 초기화
    if pre_model=='ResNet':
        base_encoder = build_encoder(input_shape, layer_name="encoder")
    elif pre_model=='CNN':
        base_encoder = lsm_CNN(input_shape)
    elif pre_model=='ViT':
        base_encoder = vit_model(input_shape)
        
    image_size = data_1.shape[1:3]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    moco_kwargs = {
        'feature_dim': 128,
        'K': 4096,
        'm': 0.999,
        'temperature': 0.07
    }
    
    moco = MoCo(
            base_encoder,
            **moco_kwargs
        )

    # 학습 스텝 정의
    def train_step(x_q, x_k):
        with tf.GradientTape() as tape:
            logits, labels = moco(x_q, x_k, training=True)
            # cross-entropy loss 계산
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, moco.trainable_variables)
        optimizer.apply_gradients(zip(gradients, moco.trainable_variables))
        return loss

    data = np.concatenate((data_1, data_2), axis=3)

    np.random.seed(seed)
    # 학습 루프
    for epoch in range(epochs):
        np.random.shuffle(data)
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # 배치 단위로 학습
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]
            
            loss = train_step(x_q=batch_data[:,:,:,:20], x_k=batch_data[:,:,:,20:])

            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return moco