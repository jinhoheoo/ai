# 여기서는 CNN을 가지고 오토인코더를 만듬

# 생성형모델 만들려고 오토인코더 사용함.
# 오토인코더는 입력 데이터를 주요 특징으로 효율적으로 압축(인코딩)한 후
# 이 압축된 표현에서 원본 입력을 재구성(디코딩)하도록 설계된 일종의 신경망 아키텍처입니다.

# 특성만 기억해서 그 특성 가지고 그림을 그려주는 사람이 몽타주 작가임
# 오토엔코더의 앞부분이 목격자 뒷부분이 몽타주 작가임 즉 앞부분에서 특성을 기억하고 뒷부분에서 그걸 가지고 얼굴을 그려내는게 몽타주 작가임
# 오토엔코더는 학습한 것만 할 수 있음.
# 줄거리만 가지고 소설을 쓸 수 있음 근데 학습한 줄거리만 가능함.
# 오토인코더는 데이터를 효율적으로 압축하고 다시 복원하는 컴퓨터 프로그램이야. 예를 들어, 사진을 생각해봐.
# 오토인코더는 그 사진을 작은 크기로 압축해서 저장하고, 그 압축된 버전을 사용해서 다시 원래의 사진을 복원해줄 수 있어.
# 이렇게 하면 메모리를 적게 쓰면서도 원래의 데이터를 보관할 수 있어. 그리고 노이즈가 있는 데이터에서도
# 깨끗한 데이터를 만들어내는 데도 사용돼. 그래서 데이터를 다루고 분석하는 데에 도움이 돼는 유용한 도구야.
#
# 오토인코더를 사용하는 방법은 다음과 같아:
#
# 1.데이터 압축 및 재구성: 오토인코더를 사용하여 데이터를 압축하고, 그 압축된 데이터를 다시 복원할 수 있어.
# 이를 통해 데이터를 효율적으로 저장하고, 필요할 때 다시 복원하여 사용할 수 있어.
#
# 2.차원 축소: 데이터의 특징을 추출하고, 중요한 정보만을 보존하여 데이터의 차원을 줄일 수 있어.
# 이는 데이터 시각화나 머신러닝 모델 학습 시에 유용하게 사용될 수 있어.
#
# 3.노이즈 제거: 노이즈가 있는 데이터에서 정확한 정보를 추출하기 위해 사용될 수 있어. 예를 들어,
# 이미지에서 잡음을 제거하거나 음성 데이터에서 배경 소음을 제거하는 등의 작업에 활용될 수 있어.
#
# 4.생성 모델: 오토인코더를 사용하여 새로운 데이터를 생성할 수도 있어. 학습된 오토인코더를 사용하여 잠재 공간에서 샘플링을 하고,
# 이를 디코딩하여 새로운 데이터를 생성할 수 있어. 이는 이미지 생성, 음악 생성 등의 작업에 사용될 수 있어.

#GAN모델은 생성형모델등등 다양한 곳에 사용됨.
#GAN(Generative Adversarial Network)은 두 개의 신경망이 서로 경쟁하면서 학습하는 시스템이야.
# -생성자(Generator): 랜덤한 잡음을 받아들여 진짜 같은 가짜 데이터(예를 들면, 이미지)를 만들어내는 역할을 해.
# -판별자(Discriminator): 진짜와 가짜를 구별하는 역할을 해. 생성자가 만든 가짜 데이터와 진짜 데이터를 구별하려고 노력해.
# 생성자는 판별자를 속이기 위해 더 진짜 같은 데이터를 만들려고 노력하고, 판별자는 생성자가 만든 가짜 데이터와 진짜 데이터를 잘 구별할 수 있도록 학습해.
# 결국 생성자는 더 진짜 같은 가짜 데이터를 만들고, 판별자는 더 정확하게 구별하게 되는거야. 이런 식으로 서로 경쟁하면서 모델이 학습되는 거야.

# GAN의 학습 과정은 다음과 같이 진행됩니다:
# 생성자는 랜덤한 입력을 받아들여 가짜 데이터를 생성합니다.
# 판별자는 생성자가 생성한 가짜 데이터와 실제 데이터를 받아들여 어느 것이 진짜인지 판별합니다.
# 이 과정에서 생성자는 판별자를 속이기 위해 더욱 진짜와 비슷한 데이터를 생성하고, 판별자는 더 정확하게 가짜 데이터를 식별하도록 학습합니다.
# 두 신경망은 서로 경쟁하면서 점점 더 발전하고, 생성자는 실제와 구별하기 힘든 가짜 데이터를 생성하고, 판별자는 이를 더욱 정확하게 판별하도록 학습합니다.
# 이러한 과정을 반복하면서 생성자와 판별자는 서로를 발전시키며 점차적으로 훈련됩니다. GAN은 이미지 생성 및 변환, 스타일 전이, 이미지 향상, 이미지 생성 및 분할 등 다양한 이미지 처리 작업에 사용됩니다.

# 사이클링(Cycling)은 주로 컴퓨터 비전 및 이미지 처리 분야에서 사용되는 개념입니다.
# 주로 이미지를 다루는 작업에서 발생하는 문제를 해결하기 위해 등장한 개념 중 하나입니다.
# CycleGAN이나 Pix2Pix와 같은 모델은 주로 사이클링을 기반으로 합니다. 이러한 모델은 이미지의 스타일을 변환하거나,
# 이미지에서 특정한 특징을 강조하거나 삭제하는 등의 작업에 사용됩니다.
# 예를 들어, CycleGAN은 한 도메인의 이미지를 다른 도메인의 이미지로 변환하는 데 사용됩니다.
# 예를 들어, 말 이미지를 얼룩말 이미지로, 혹은 여름 풍경을 겨울 풍경으로 변환하는 등의 작업이 가능합니다.
# 이렇게 함으로써 어떤 도메인의 이미지를 다른 도메인의 이미지로 변환하면서도 원본 이미지의 중요한 특징을 보존하는 것이 가능해집니다.
# 사이클링은 이전에는 특히 동일한 종류의 데이터가 있는 두 가지 도메인 간의 변환에 사용되었지만,
# 최근에는 텍스트 데이터나 음성 데이터와 같은 다양한 유형의 데이터에서도 사용되고 있습니다.
# 이러한 변환을 통해 데이터를 보다 효과적으로 활용할 수 있고, 다양한 응용 분야에서 활용될 수 있습니다.

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

input_img = Input(shape=(28, 28, 1,))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)    # 28 x 28
x = MaxPool2D((2, 2), padding='same')(x)                                # 14 x 14
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # 14 x 14
x = MaxPool2D((2, 2), padding='same')(x)                                # 7 x 7
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # 7 x 7
encoded = MaxPool2D((2, 2), padding='same')(x)                          # 4 x 4
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)       # 4 x 4
x = UpSampling2D((2, 2))(x)                                             # 8 x 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # 8 x 8
x = UpSampling2D((2, 2))(x)                                             # 16 x 16
x = Conv2D(16, (3, 3), activation='relu')(x)                            # 14 x 14
x = UpSampling2D((2, 2))(x)                                             # 28 x 28
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)       # 28 x 28


autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

fit_hist = autoencoder.fit(conv_x_train, conv_x_train,
               epochs=100, batch_size=256,
               validation_data=(conv_x_test, conv_x_test))


decoded_img = autoencoder.predict(conv_x_test[:10])

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
autoencoder.save('./models/autoencoder.h5')

#
# DNN vs CNN 오토인코더 비교

# 1.입력 데이터 처리 방식
# DNN 오토인코더는 데이터를 벡터로 변환하여 처리합니다. 이미지 데이터의 경우 픽셀 값을 1차원 벡터로 변환하여 사용합니다.
# CNN 오토인코더는 2차원 이미지 데이터를 그대로 입력으로 사용합니다. 컨볼루션 필터를 사용하여 이미지의 공간적 특징을 추출합니다.

# 2.장점
# DNN 오토인코더: 다양한 유형의 데이터에 적용 가능, 데이터 압축 및 표현 학습에 효과적, 간단한 구조
# CNN 오토인코더: 이미지 데이터 처리에 특화, 공간적 특징 추출에 효과적, 이미지 재구성, 잡음 제거, 스타일 변환 등에 효과적

# 3.단점
# DNN 오토인코더: 이미지 데이터 처리에 비효율적, 공간적 특징 추출에 어려움, 과적합 가능성
# CNN 오토인코더: 데이터 벡터화 과정 불필요, 이미지 데이터 처리에 효과적, 모델 구조가 복잡, 학습 데이터 크기가 충분하지 않을 경우 과적합 가능성

# 4.적용 분야
# DNN 오토인코더: 추천 시스템, 협업 필터링, 이상 탐지, 데이터 증강, 텍스트 요약, 음성 합성
# CNN 오토인코더: 이미지 재구성, 잡음 제거, 이미지 색상화, 스타일 변환, 객체 인식, 이미지 분류

# 5.결론
# DNN 오토인코더는 다양한 유형의 데이터에 적용 가능하고 데이터 압축 및 표현 학습에 효과적이지만, 이미지 데이터 처리에는 비효율적입니다.
# CNN 오토인코더는 이미지 데이터 처리에 특화되어 있고 공간적 특징 추출에 효과적이지만, 모델 구조가 복잡하고 학습 데이터 크기가 충분하지 않을 경우 과적합 가능성이 있습니다.
# 따라서, 사용 목적과 데이터 유형에 따라 적절한 오토인코더 모델을 선택해야 합니다.