# DNN을(Dense layer) 가지고 GAN모델 만들거임 그리고 여기선 오토엔코더말고 제너레이터 사용함.
# 오토엔코더의 디코더 부분만을 만들거임.
# 모델을 두개 만들어서 서로 경쟁하게끔 학습시킬거임.
# 오토엔코더는 학습한거를 다시 가져오는거라고 보면 제너레이터는 새로운걸 만드는 거라 보면됨. 제너레이터는 주로 생성적 적대 신경망(GAN)과 관련이 있습니다.
# 오토인코더는 학습한 데이터를 복원하는 데 중점을 둔 반면, 제너레이터는 새로운 데이터를 생성하는 데 중점을 둡니다.
# 오토인코더는 입력 데이터를 가능한 정확하게 재구성하는 것을 목표로 합니다.
# 이 과정에서 오토인코더는 데이터의 중요한 특징을 추출하고 새로운 표현 형태를 제공합니다.
# 제너레이터는 학습 데이터의 분포를 기반으로 새로운 데이터를 생성합니다.
# 이 과정에서 제너레이터는 새로운 특징을 조합하고 창의적인 결과물을 만들어냅니다.
# 오토인코더와 제너레이터는 종종 함께 사용됩니다. 예를 들어, 오토인코더를 사용하여 데이터를 전처리한 후 제너레이터를 사용하여 새로운 데이터를 생성할 수 있습니다.
# 오토인코더와 제너레이터는 다양한 분야에서 활용됩니다. 예를 들어, 이미지 생성, 음악 생성, 텍스트 생성 등에 사용됩니다.
# 요약하면, 오토인코더는 입력 데이터의 표현을 학습하고 입력 데이터를 재구성하는데 사용되는 반면, 제너레이터는 잠재 공간에서 새로운 데이터를 생성하는데 사용됩니다.

# 이진 분류기에서는 sigmoid를 사용함 다중 분류기에서는 softmax를 사용함
# 이진 분류에서는 sigmoid 함수를 사용하여 출력값을 0과 1 사이의 확률로 변환합니다.
# 다중 분류에서는 softmax 함수를 사용하여 각 클래스에 대한 확률을 계산합니다.
# ReLU는 은닉층에서 주로 사용되며, 신경망의 비선형성을 증가시키고, 효율적인 학습을 돕는 데 사용됩니다.
# 일반적으로 은닉층에서는 ReLU가 사용되고, 출력층에서는 다양한 상황에 따라 sigmoid 또는 softmax 함수가 사용됩니다.
# 그래서 출력이 2가지를 분류하는거면 sigmoid 3개 이상으로 분류하는거면 softmax임
# MNIST 데이터는 0에서 9까지의 숫자 이미지로 구성되어 있습니다.
# MNIST 데이터는 0에서 9까지의 숫자 이미지로 구성되어 있습니다. 각 픽셀 값은 0과 255 사이의 값을 가지며, 이는 이미지의 밝기를 나타냅니다.
# 오토인코더는 입력 이미지를 가능한 정확하게 재구성하는 것을 목표로 합니다.
# 그래서 sigmoid를 사용해서 0과 1사이의 값을 활용하는데 픽셀을 255로 나눠서 0과1사이에 값 만듬

# DNN (Deep Neural Network)(딥 러닝 신경망):
# DNN은 입력층, 은닉층, 출력층으로 이루어진 심층 신경망 구조를 가지고 있습니다.
# 각 층은 완전 연결된 뉴런으로 구성되어 있습니다.
# 주로 이미지 분류, 텍스트 분류, 회귀 즉 이미지 인식, 자연어 처리, 음성 인식 등 다양한 분야에서 사용됩니다.
# DNN은 깊은 신경망 구조를 가지고 있어서 복잡한 패턴을 학습할 수 있지만, 많은 양의 데이터와 연산 리소스가 필요합니다.
# DNN은 입력층과 출력층 사이에 여러 개의 은닉층을 가진 인공 신경망
# 이러한 은닉층은 네트워크가 입력과 출력 데이터 사이의 복잡한 관계를 학습하도록 합니다.
# ANN기법의 여러문제가 해결되면서 모델 내 은닉층을 많이 늘려서 학습의 결과를 향상시키는 방법이 등장하였고
# 이를 DNN(Deep Neural Network)라고 합니다. DNN은 은닉층을 2개이상 지닌 학습 방법을 뜻합니다.
# 컴퓨터가 스스로 분류레이블을 만들어 내고 공간을 왜곡하고 데이터를 구분짓는 과정을 반복하여 최적의 구번선을 도출해냅니다.
# 많은 데이터와 반복학습, 사전학습과 오류역전파 기법을 통해 현재 널리 사용되고 있습니다.
# 그리고, DNN을 응용한 알고리즘이 바로 CNN, RNN인 것이고 이 외에도 LSTM, GRU 등이 있습니다.

# CNN (Convolutional Neural Network)(컨볼루션 신경망)(이미지에서 주로사용):
# CNN은 이미지 처리를 위한 딥러닝 아키텍처로, 이미지의 공간적인 구조를 활용하여 학습합니다.
# 합성곱층(Convolutional layer)과 풀링층(Pooling layer)으로 구성되어 있으며, 특징 추출을 위해 필터를 사용합니다.
# 주로 이미지 인식, 객체 감지, 이미지 분할 즉 이미지 분류, 객체 감지, 이미지 세분화 등의 작업에 사용됩니다.
# CNN은 이미지의 공간적인 관계를 고려하여 학습하기 때문에 이미지 처리에 매우 효과적입니다.
# CNN은 이미지 데이터를 처리하도록 특별히 설계됨. CNN은 필터를 사용하여
# 이미지에서 가장자리, 선, 모양과 같은 특징을 추출합니다. 이러한 특징은 이미지를 분류하거나 이미지의 다른 속성을 예측하는 데 사용됩니다.

# RNN (Recurrent Neural Network)(순환 신경망):
# RNN은 순차적인 데이터를 처리하는데 사용되는 딥러닝 아키텍처입니다.
# 순환층(Recurrent layer)을 가지고 있으며, 시간적인 의존성을 가진 데이터를 처리할 수 있습니다.
# 주로 자연어 처리, 시계열 데이터 분석, 음성 인식 즉 기계 번역, 감정 분석, 텍스트 생성 등에 사용됩니다.
# RNN은 이전의 정보를 현재의 상태에 반영하여 순차적인 데이터를 처리할 수 있어서, 시간에 따른 패턴을 학습하는데 효과적입니다.
# RNN은 텍스트 또는 시계열 데이터와 같은 순차 데이터를 처리하도록 설계됨
# RNN은 메모리를 사용하여 과거에 대한 정보를 저장하므로 데이터의 맥락에 따라 예측을 할 수 있습니다.

#LSTM (Long Short-Term Memory)과 GRU (Gated Recurrent Unit)는 둘 다 순환 신경망 (RNN)의 한 종류이며 순차 데이터 처리에 뛰어납니다.
#LSTM vs GRU 쉽게 설명: LSTM과 GRU는 둘 다 기억력이 좋은 신경망 모델이라고 생각하면 됩니다.

# LSTM:
# 장기 기억력이 뛰어나 과거의 정보를 오랫동안 기억할 수 있습니다.
# 복잡한 구조로 인해 학습 속도가 느리고 계산 비용이 많이 드는 단점이 있습니다.
# 시계열 데이터 분석, 음성 인식, 자연어 처리와 같은 복잡한 작업에 적합합니다.
#
# GRU:
# LSTM보다 기억력이 약간 떨어지지만 학습 속도가 빠르고 계산 비용이 적습니다.
# 비교적 짧은 시퀀스 데이터나 LSTM만큼 강력한 기억 능력이 필요하지 않은 작업에 적합합니다.

# GAN 만들 때 생성기와 판별기를 번갈아가면서 학습해야해야 학습횟수가 조절되어야함.

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

OUT_DIR = './DNN_out'
img_shape = (28,28,1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

# 오토엩코더의 디코더부분만 만들거에요
# 이진분류기를 이용해서 글씨를 판별하게끔. ex) 조금이라도 비슷하면 손글씨라고 판단하게끔 만들어야해. 안그러면 조금이라도 틀리면 이거는 잡음으로 판단을 해. 그래서 너무 정확하게 판단하게끔 만들면 안돼
# 생성모델이 점점 성능이 좋아지면서 이진분류기의 성능도 점점 좋아져야해.
# 값모델이 2개가 있어야해. 이 2개의 모델이 서로 경쟁하면서 성능이 좋아짐.

(x_train, _),(_, _) = mnist.load_data()
print(x_train.shape)

x_train = x_train / 127.5 - 1   ## 이런경우 음수값이 있을 때 leakyRelu를 사용해
x_train = np.expand_dims(x_train, axis=3) # axis=3 이말은 3차원짜리 데이터 6만개 !
print(x_train.shape)

generator = Sequential()
generator.add(Dense(128, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01))
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape(img_shape))
generator.summary()

# LeakyReLU(Leaky Rectified Linear Unit)는 일반적인 ReLU(Rectified Linear Unit) 함수의 변형 중 하나입니다. LeakyReLU는 ReLU의 변종 중 하나로, 주로 생성 모델에서 사용됩니다. LeakyReLU는 다음과 같은 특징을 가지고 있습니다.
#
# 음수 값에 대한 처리:
#
# LeakyReLU는 일반적인 ReLU와는 달리 음수 값에 대해 완전히 0이 아닌 작은 음수 값을 반환합니다. 즉, 입력이 음수일 때도 작은 기울기를 가지므로, 항상 양수의 출력을 생성합니다.
# 그래디언트 소실 문제 완화:
#
# 일반적인 ReLU는 입력이 음수인 경우 그래디언트가 0이 되어 그 이후의 역전파에서 가중치 업데이트가 이루어지지 않는 "그래디언트 소실" 문제를 가질 수 있습니다. LeakyReLU는 이를 완화하여 일부 음수 값에 대해서도 그래디언트를 전달합니다.
# 생성 모델에서의 활용:
#
# 생성 모델에서는 LeakyReLU가 학습의 안정성을 향상시킬 수 있습니다. 특히, 생성자(generator) 부분에서 사용되는 경우, LeakyReLU를 통해 모델이 다양한 특징을 학습하고 모드 붕괴(mode collapse)를 피할 수 있습니다.
# 따라서, 코드에서 생성자 모델의 첫 번째 레이어에 LeakyReLU를 사용하는 이유는 모델이 학습하는 동안 다양한 특징을 포착하고 그래디언트 소실 문제를 완화하기 위해서입니다. LeakyReLU의 alpha 매개변수는 음수 영역의 기울기를 조절하는 값으로, 작은 음수 값으로 설정하여 음수 영역에서도 정보를 유지하도록 합니다.


# 이진분류기
lrelu = LeakyReLU(alpha=0.01)
discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(128, activation=lrelu))
discriminator.add(Dense(1,activation='sigmoid')) # 이진분류기 sigmoid, 다중분류기 softmax
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()

gan_model.compile(loss='binary_crossentropy',optimizer='adam')
# ganmodel 컴파일을 할 때 학습을 안하는 걸 할려면
discriminator.trainable=False

real = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

# print(real)
# print(fake)


# 만약에 이미지가 쉬우면 discriminator이 앞지르게 됨. 그럴 경우 generator를 두번 학습시키고  discriminator을 한 번만 학습시키게끔 해야해.
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0],batch_size)
    real_img = x_train[idx]

    z= np.random.normal(0,1,(batch_size,noise))
    fake_img = generator.predict(z)

    # 가짜 이미지(잡음이미지) -> fake_img
    d_hist_real = discriminator.train_on_batch(real_img,real)
    d_hist_fake = discriminator.train_on_batch(fake_img, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    # generator를 한 번 해야해.
    z = np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z,real)

    # 학습이 오래걸릴테니까 print를 중간중간에 해보는 용
    if epoch % sample_interval == 0:
        print('%d [ D loss: %f, acc: %.2f%%] [G loss:%f]'%(
            epoch, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z =np.random.normal(0,1,(row*col,noise))
        fake_imgs= generator.predict(z)
        fake_imgs = 0.5 * fake_imgs
        _, axs = plt.subplots(row,col,figsize= (row,col),sharey=True,sharex=True)
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[count, :, :,0],cmap='gray')
                axs[i,j].axis('off')
                count += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1))
        plt.savefig(path)
        plt.close()



















