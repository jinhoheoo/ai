# 여기서는 DNN 즉 Dense layer을 이용해 오토 인코더 만듬
# DNN(Deep Neural Network)을 사용하여 오토인코더를 만들 수 있습니다. 오토인코더는 인코더(encoder)와 디코더(decoder)로 구성되어 있으며,
# 인코더는 입력 데이터를 잠재 공간으로 압축하고, 디코더는 잠재 공간의 표현을 사용하여 원래의 입력 데이터를 재구성합니다

# 인코더(encoder):
# 입력 데이터를 잠재 공간(latent space)으로 압축하는 역할을 합니다.
# 입력 데이터의 중요한 특징을 추출하여 잠재 공간에 표현됩니다.
# 일반적으로는 입력 데이터를 저차원 벡터로 매핑하는 신경망으로 구성됩니다.

# 디코더(decoder):
# 잠재 공간의 표현을 사용하여 원래의 입력 데이터를 재구성하는 역할을 합니다.
# 잠재 공간의 특징을 이용하여 입력 데이터를 재구성합니다.
# 일반적으로는 인코더와 대칭되는 구조를 가지며, 입력 데이터와 동일한 차원의 출력을 생성합니다.

# 오토인코더는 입력 데이터를 재구성하는 과정에서 입력 데이터의 중요한 특징을 잘 보존하려고 합니다.
# 이를 위해 학습 과정에서는 입력 데이터와 재구성된 데이터 간의 차이를 최소화하도록 학습됩니다.
# 이렇게 함으로써 오토인코더는 입력 데이터를 효과적으로 표현하는 잠재 공간을 학습하게 됩니다.

# 오토인코더는 주로 데이터 압축, 잡음 제거, 특징 추출 등의 작업에 사용됩니다. 또한, 생성 모델로 활용하여 가짜 데이터를 생성하는 데도 사용될 수 있습니다.
# 즉 오토인코더로 학습된 걸 넣어두면 그걸 다시 꺼내 쓸 수 있는거임. 그래서 생성형 모델이나 뭐 이미지 다시 불러온다거나 뭐등등 다양하게 쓸 수 있음.
# 20번에 설명 더 잘되있음

#오토인코더(Autoencoder)는 입력 데이터를 학습하여 압축된 표현으로 인코딩한 후,
# 디코더를 사용하여 원래 데이터로 재구성하는 신경망 아키텍처입니다. 오토인코더는 데이터를 효과적으로 표현하는 방법을 학습하는 데 사용됩니다.
# 이 과정에서는 입력 데이터의 핵심적인 특성이 잘 보존되는 압축된 표현이 만들어집니다.

# 이 코드는 MNIST 데이터 세트에서 이미지 재구성을 위해 간단한 오토인코더를 구현합니다.
# 오토인코더는 32차원 인코딩 레이어에서 이미지의 압축 표현을 학습합니다.
# 디코더는 이 압축 표현에서 원본 이미지를 재구성하려고 합니다.
# 이 코드는 오토인코더의 기본 구조와 학습 과정을 보여줍니다.

# 추가 정보:
# MNIST 데이터 세트는 손으로 쓴 숫자 이미지로 구성된 대표적인 데이터 세트입니다.
# 오토인코더는 이미지 압축, 이상 탐지, 데이터 증강 등 다양한 분야에서 활용됩니다.
# 이 코드는 Keras를 사용하여 오토인코더를 구현하는 기본적인 예시입니다.

# 오토인코더에서 시그모이드 함수의 역할:
# 오토인코더 출력층에서 시그모이드 함수는 데이터 값을 0에서 1 사이의 범위로 변환합니다.
# 이는 데이터 값을 정규화하여 모델 학습 과정을 안정화하는 데 도움이 됩니다.
# 특히, 시그모이드 함수는 데이터 값의 분포를 정규화하여 모델이 데이터의 특징을 더욱 효과적으로 학습하도록 합니다.
# MNIST 재구축 AutoEncoder 예제에서 최종 output layer에 sigmoid function을 사용한 이유는 인풋 이미지의 pixel intensity를
# 255로 나눠서 [0~1] range값으로 사용하기에 출력결과도 sigmoid 함수를 적용해서 [0~1] range로 맞춰준 것인데요.
# 다만 학습과정에서 어차피 input과 똑같은 형태가 되도록 계속 optimize를 진행하기에 ReLU를 사용해도 충분히 학습을 시킨다면
# 출력 결과값이 [0~1] range와 근사한 값이 될 것이므로 큰 문제는 없을 것입니다.

import matplotlib.pyplot as plt     #그래프를 생성하고 시각화하는 데 사용됩니다.
import numpy as np                  #숫자 계산 기능을 제공합니다.
from tensorflow.keras.models import * #신경망 모델을 구축하는 데 사용되는 클래스를 포함합니다.
from tensorflow.keras.layers import * #신경망을 구축하는 데 사용되는 레이어를 포함합니다.
from tensorflow.keras.datasets import mnist # MNIST를 포함한 일반적인 데이터세트에 대한 액세스를 제공합니다.


input_img = Input(shape=(784,)) #입력 레이어: 784차원으로 평탄화된 이미지를 받습니다.
encoded = Dense(32, activation='relu') #인코딩 레이어: ReLU 활성화 함수를 사용하는 32개의 뉴런을 가진 밀집 레이어입니다.
encoded = encoded(input_img)

decoded = Dense(784, activation='sigmoid') #디코딩 레이어: Sigmoid 활성화 함수를 사용하는 784개의 뉴런을 가진 밀집 레이어입니다.
decoded = decoded(encoded)

autoencoder = Model(input_img, decoded) #오토인코더 모델: 입력을 디코딩된 출력에 연결하여 입력을 재구성하는 것을 목표로 합니다.
autoencoder.summary()

encoder = Model(input_img, encoded) #인코더 모델: 입력에서 인코딩된 표현을 추출합니다.
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))  #디코더 모델: 인코딩된 표현에서 입력을 재구성합니다.
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#옵티마이저: 효율적인 학습을 위해 Adam 옵티마이저가 선택됩니다.
#손실 함수: 이진 교차 엔트로피는 재구성 오류를 측정하는 데 사용됩니다.

(x_train, _), (x_test, _) = mnist.load_data() #MNIST 데이터세트 로드: keras.datasets에서 학습 및 테스트 세트를 로드합니다.

x_train = x_train / 255 #픽셀 값 정규화: 학습 안정성을 위해 픽셀 값을 [0, 1] 범위로 조정합니다.
x_test = x_test / 255

flatted_x_train = x_train.reshape(-1, 28 * 28) #이미지 평탄화: 이미지를 784차원 벡터로 재구성하여 오토인코더 입력으로 사용합니다.
flatted_x_test = x_test.reshape(-1, 28 * 28)
print(flatted_x_train.shape)
print(flatted_x_test.shape)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train,
               epochs=50, batch_size=256,
               validation_data=(flatted_x_test, flatted_x_test))
# 모델 학습: 학습 이미지를 재구성하도록 오토인코더를 학습합니다.
# 에포크: 전체 데이터 세트를 50번 반복합니다.
# 배치 크기: 각 학습 단계에서 256개의 이미지를 처리합니다.
# 검증 데이터: 학습 중에 테스트 세트에서 성능을 모니터링합니다.

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

# 테스트 이미지 인코딩 및 디코딩: 인코더와 디코더를 적용하여 재구성을 시각화합니다.
# 원본 이미지와 재구성된 이미지 표시: 원본 테스트 이미지와 재구성된 이미지를 함께 표시합니다.
# 손실 곡선 표시: 학습 과정을 평가하기 위해 에포크에 따른 학습 및 검증 손실을 시각화합니다.


#DNN으로 오토인코더 구현
# 1. 모델 설계
# 입력 레이어: 데이터의 차원만큼 뉴런을 가진 레이어입니다.
# 인코딩 레이어: 여러 개의 은닉 레이어로 구성됩니다. 각 레이어는 ReLU 또는 Sigmoid와 같은 활성화 함수를 사용합니다.
# 디코딩 레이어: 인코딩 레이어와 대칭적인 구조로 구성됩니다. 각 레이어는 ReLU 또는 Sigmoid와 같은 활성화 함수를 사용합니다.
# 출력 레이어: 입력 레이어와 동일한 차원을 가진 레이어입니다. Sigmoid 활성화 함수를 사용하여 데이터를 원래 범위로 변환합니다.

# 2. 데이터 준비
# 오토인코더 모델 학습에 사용할 데이터 세트를 준비합니다.
# 데이터를 정규화하여 0과 1 사이의 값으로 변환합니다.

# 3. 모델 학습
# Adam 또는 RMSProp와 같은 최적화 알고리즘을 사용하여 모델을 학습합니다.
# Binary Cross Entropy 또는 Mean Squared Error와 같은 손실 함수를 사용합니다.
# 모델 학습 과정에서 과적합을 방지하기 위해
# Early Stopping
# Dropout
# L1/L2 정규화와 같은 기법을 사용합니다.

# 4. 모델 평가
# 학습된 모델이 입력 데이터를 정확하게 재구성하는지 평가합니다.
# Mean Squared Error 또는 Peak Signal-to-Noise Ratio와 같은 평가 지표를 사용합니다.