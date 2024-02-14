# -*- coding: utf-8 -*-
"""exam03_keras_xor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gOTBwotcvejznYwsm22pjin7amEzKZ_U

인공지능 두가지

값을 예측하는 회귀와 분류

두개의 입력을 받아서 32개로 늘린후 1개로 줄임 즉 2-32-1 이렇게 됨.
그래서 두개의 입력 각가의 32개와 각각 연결되고 32개가 1개와 모두 연결된다. 즉 전부 연결됬다고 보면 됨. 이렇게 된게 퍼셉트론임 퍼셉트론은 인간의 신경세포를 프로그램으로 만들어 본게 퍼셉트론임.
인간의 신경세포는 디지털과 같기 때문에 0아니면 1인데 미분은 0점이 있거나 불연속할 때 할 수 없는데 그것을 해결하기 위해 시그모이드를 써서 xor 문제를 해결함
근데 시그몰은 0아니면 1이아니라 반올림을 써서 해결함.

시그모이드 함수는 실수 전체의 입력을 0과 1 사이의 값으로 변환하는 비선형 함수입니다.
그래서 시그모이드 함수는 모든 입력에대해서 미분가능하기 때문에 사용한거임.

시그모이드 함수는 오랫동안 사용되었지만, 최근에는 ReLU(Rectified Linear Unit)와 같은 활성화 함수가 더 선호되는 경우가 많습니다. 이는 ReLU가 학습이 빠르고 계산이 효율적이며, 양수 부분에서는 선형으로 동작하여 그래디언트 소실 문제를 완화하기 때문입니다.

1.퍼셉트론 (1957): 프랑크 로젠블라트(Frank Rosenblatt)가 제안한 퍼셉트론은 인공 뉴런(퍼셉트론)을 기반으로 한 단일 레이어의 신경망입니다. 이는 입력과 가중치의 선형 조합을 활성화 함수를 통해 출력으로 변환하는 구조를 가지고 있습니다. 하지만 퍼셉트론은 XOR 같은 비선형 문제를 해결할 수 없는 한계가 있었습니다.

2.다층 퍼셉트론 (1960s - 1970s): 이후 Marvin Minsky와 Seymour Papert가 퍼셉트론의 한계를 지적하면서, 단일 레이어의 퍼셉트론으로는 XOR과 같은 비선형 문제를 해결할 수 없다는 문제가 드러났습니다. 이에 따라 다층 퍼셉트론이 제안되었고, 여러 층의 퍼셉트론을 결합하여 비선형 문제를 해결할 수 있게 되었습니다.

3.시그모이드 함수 (1970s - 1980s): 시그모이드 함수와 같은 비선형 활성화 함수의 사용이 도입되면서 뉴럴 네트워크의 표현력이 향상되었습니다. 시그모이드 함수는 연속적이면서 미분 가능하여 역전파 알고리즘을 사용하여 가중치를 효과적으로 학습할 수 있게 했습니다.

4.역전파 알고리즘 (1986): 역전파 알고리즘은 신경망의 가중치를 효과적으로 조정하여 학습하는 방법으로, 이를 통해 다층 퍼셉트론이 비선형 문제를 학습하고 효과적으로 표현할 수 있게 되었습니다. 이 알고리즘이 제안되면서 뉴럴 네트워크의 훈련이 가능해지면서 발전이 가속화되었습니다.

5.컴퓨터 성능 향상과 데이터 양 증가 (1990s - 현재): 컴퓨터의 성능이 향상되고 대규모 데이터셋의 사용이 가능해지면서 뉴럴 네트워크는 더욱 복잡하고 깊어지게 되었습니다. 이로 인해 딥 러닝이 부상하게 되었고, 다양한 신경망 아키텍처와 활성화 함수가 등장하면서 이미지 분류, 음성 인식, 자연어 처리 등 다양한 분야에서 뛰어난 성능을 보이게 되었습니다.

6.최신 발전 (현재): 현재에 이르러서는 ResNet, Transformer, GPT 등과 같은 많은 고급 신경망 아키텍처들이 등장하고 있습니다. 또한 강화학습, 생성 모델 등 다양한 분야에서의 응용이 이루어지고 있습니다.


비선형(Nonlinear)은 선형이 아닌 형태를 의미합니다. 수학적으로 말하면, 어떤 함수가 비선형이라는 것은 그 함수가 직선 형태가 아니라 곡선 형태를 띄고 있다는 것을 의미합니다. 비선형 함수는 입력과 출력 간의 관계가 간단한 비례 또는 역비례 관계가 아닌 경우에 해당합니다.

특징:
곡선 형태: 비선형 함수는 직선이 아닌 곡선 형태를 가지고 있습니다.

비선형성의 필요성: 신경망과 같은 복잡한 모델에서는 비선형 함수가 필요합니다. 만약 활성화 함수나 모델 자체가 선형이라면 여러 층을 쌓는 것이 의미가 없어져 표현력이 제한될 것입니다.

복잡한 패턴 학습: 비선형 함수를 사용하면 모델이 데이터에서 복잡한 비선형 패턴을 학습할 수 있습니다. 이는 실제 세계의 많은 현상과 관련이 있습니다.

##XOR
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

training_data = np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')
target_data = np.array([[0], [1], [1], [0]],'float32')

#0,1이기 때문에 스케일링 필요없음, Dense가 레이어임 2개 썻으니 2개의 층 쌓은거임.
model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
model.summary()

"""퍼셉트론은 뉴런을 모델로 한 간단한 인공 신경망 구조입니다.

두개의 입력을 받아서 32개로 늘린후 1개로 줄임 즉 2-32-1 이렇게 됨. 그래서 두개의 입력 각가의 32개와 각각 연결되고 32개가 1개와 모두 연결된다. 즉 전부 연결됬다고 보면 됨.  

주어진 코드에서 첫 번째 Dense 레이어는 32개의 뉴런을 가지고 있습니다. 이는 첫 번째 레이어에서 사용된 Dense(32, input_dim=2, activation='relu') 코드에서 나타납니다.

해석해보면:

32: 레이어의 뉴런 개수입니다. 즉, 첫 번째 은닉층은 32개의 뉴런을 가지게 됩니다.
input_dim=2: 입력 차원은 2입니다. 즉, 입력 데이터의 특성(feature)가 2개입니다.
activation='relu': 활성화 함수로 ReLU(Rectified Linear Unit)를 사용합니다.
따라서, 첫 번째 Dense 레이어는 2차원의 입력을 받아 32개의 뉴런을 가진 은닉층을 형성하고, ReLU 활성화 함수를 통과시킵니다. 두 번째 Dense 레이어는 1개의 뉴런을 가진 출력 레이어로, 시그모이드 활성화 함수를 사용하여 이진 분류 문제를 해결합니다.

위코드를 그림으로 표현하면 아래의 사진과 같음.

입력: 2개의 특성(feature)을 가진 데이터가 사용됩니다.
첫 번째 레이어(Dense(32, input_dim=2, activation='relu')): 2개의 입력을 받아 32개의 퍼셉트론(뉴런)을 가진 은닉층을 형성하고, ReLU 활성화 함수를 통과시킵니다. 이는 32개의 특성을 가진 새로운 표현을 만듭니다.
두 번째 레이어(Dense(1, activation='sigmoid')): 32개의 퍼셉트론의 출력을 받아 1개의 출력을 가진 출력 레이어를 형성하고, 시그모이드 활성화 함수를 통과시켜 이진 분류를 위한 확률 값을 출력합니다.
따라서, 입력 레이어에 있는 특성들이 첫 번째 은닉층에 있는 32개의 퍼셉트론에 의해 변환되고, 그 다음 출력 레이어에 의해 최종 출력이 생성됩니다. 이렇게 설계된 모델은 2개의 입력, 32개의 은닉층 뉴런, 1개의 출력 뉴런을 가진 간단한 다층 퍼셉트론입니다.

그래서 아래 값들을 전부 더한게 출력값이 되는거임. 아래와같이 엄청 복잡해 지는데 이를 해결하기 위해 미분으로 추적하는 역전파 알고리즘을 사용함.

층 퍼셉트론은 여러 개의 은닉층과 비선형 활성화 함수를 사용하여 복잡한 함수를 모델링할 수 있습니다. 그러나 이로 인해 전체 네트워크의 파라미터를 효과적으로 학습하기 위해서는 역전파(backpropagation) 알고리즘이 필요합니다.

역전파 알고리즘은 경사 하강법(Gradient Descent)을 사용하여 네트워크의 가중치를 조정하는 과정에서 미분을 효과적으로 활용합니다. 역전파는 연쇄 법칙(Chain Rule)을 이용하여 네트워크의 출력에서부터 입력 방향으로 역으로 거슬러 올라가며 각 가중치에 대한 손실 함수의 편미분을 계산합니다.

여러 은닉층과 비선형성을 사용하면 네트워크의 출력을 입력에 대해 비선형적으로 매핑할 수 있습니다. 이러한 비선형성은 다양한 패턴과 특징을 학습하는 데 도움이 됩니다. 역전파 알고리즘은 이런 복잡한 구조에서 파라미터를 조정하여 손실 함수를 최소화하는 방향으로 학습을 진행합니다.

역전파 알고리즘은 딥러닝에서 가중치를 효과적으로 업데이트하는 핵심 알고리즘 중 하나입니다. 역전파는 경사 하강법과 함께 사용되어 신경망이 주어진 작업에서 학습하도록 도와줍니다. 역전파는 각 파라미터에 대한 손실 함수의 기울기를 계산하고, 이를 사용하여 가중치를 업데이트합니다.

이 과정을 통해 신경망은 입력과 출력 간의 복잡한 비선형 관계를 학습하며, 다층 퍼셉트론과 같은 구조에서 역전파 알고리즘은 효과적으로 미분을 활용하여 학습을 진행합니다

model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
model.summary()

여기서 Sequential()은 선형뿐만 아니라 비선형을 만들때도 사용함.
dense저걸로 1개의 계층을 만든다는거고 32저걸로 32개의 뉴런만든다는거고, dim저거는 입력2개 만든다는거고

activation은 활성화 함수로 relu와 sigmoid와 같은 비선형함수를 활성화해서 복잡해서 해결되지 않던 문제를 해결한거임.

relu(Rectified Linear Unit) 활성화 함수를 사용하고 있습니다. ReLU는 입력이 양수인 경우에는 그 값을 그대로 출력하고, 음수인 경우에는 0으로 출력하는 함수입니다. 이러한 특성으로 인해 전체적인 모델이 비선형적인 특성을 갖게 됩니다.
그래서 비선형이 됨. sigmoid도 비선형을 만드는 함수임. 시그모이드 함수를 사용함으로써 모델은 입력에 대한 비선형 변환을 수행하고, 최종적으로는 이진 분류를 위한 확률 값을 출력하게 됩니다. 이것이 일반적으로 사용되는 다층 퍼셉트론의 구조입니다.

ReLU(Rectified Linear Unit)와 시그모이드 함수는 주로 비선형성을 도입하기 위해 사용됩니다. 인공 신경망에서 활성화 함수를 사용하는 주된 이유 중 하나는 네트워크에 비선형성을 주입하여 복잡한 문제를 해결할 수 있게 하는 것입니다.

ReLU (Rectified Linear Unit): ReLU는
f(x)=max(0,x)로 정의되며, 입력이 양수인 경우에는 그 값을 그대로 반환하고, 음수인 경우에는 0으로 변환합니다. 이 비선형 함수는 학습을 더 빠르게 만들 수 있고, 특히 이미지 처리와 같은 분야에서 성능이 좋습니다.

시그모이드 함수: 시그모이드 함수는 σ(x)=  1/(1+(e-x제곱)) 로 정의되며, 모든 실수 입력을 0과 1 사이의 값으로 압축합니다. 주로 이진 분류 문제에서 출력을 확률로 해석하는 데 사용됩니다. 즉 입력을 0과 1 사이의 값으로 압축하는 비선형 함수로, 출력값을 확률로 해석할 수 있습니다.

둘 다 비선형 함수이기 때문에, 다층으로 쌓았을 때 여러 층의 결합으로 인해 전체적인 모델이 비선형성을 가질 수 있습니다. 이러한 비선형성은 모델이 더 복잡한 데이터 패턴을 학습할 수 있도록 도와줍니다.

loss='mse':
손실 함수(loss function)는 모델이 예측한 값과 실제 값 간의 차이를 측정하는 함수입니다.
'mse'는 Mean Squared Error(평균 제곱 오차)를 의미합니다. 이는 회귀 문제에서 주로 사용되며, 예측 값과 실제 값의 차이를 제곱한 후 평균을 계산합니다.
목표는 이 손실을 최소화하여 모델이 정확한 예측을 하도록 하는 것입니다.

optimizer='adam':
옵티마이저(optimizer)는 모델의 가중치를 업데이트하는 알고리즘을 지정합니다. 가중치 업데이트는 손실 함수를 최소화하기 위한 방향으로 진행됩니다.
'adam'은 Adam 옵티마이저를 사용한다는 것으로, 경사 하강법의 변종 중 하나입니다. Adam은 학습률을 조절하면서 가중치를 업데이트하며, 다양한 문제에 효과적으로 사용됩니다.

metrics=['binary_accuracy']:
모델의 평가 지표(metrics)는 학습 중에 모니터링할 지표를 지정합니다. 이는 모델의 성능을 측정하는 데 사용됩니다.
여기서는 이진 분류 문제에서의 정확도('binary_accuracy')를 사용하고 있습니다. 정확도는 모델이 올바르게 예측한 샘플의 비율을 나타냅니다.

metrics는 모델을 평가할 때 사용되는 지표(metric)들을 지정하는 파라미터입니다. 모델이 훈련되고 검증되는 동안, 지정된 metrics에 따라 모델의 성능이 평가됩니다.
간단히 말하면, metrics는 모델의 품질을 측정하는 방법입니다. 주로 훈련된 모델의 성능을 이해하고 개선하기 위해 사용됩니다.
즉 모델의 함수값이 제대로 나왔는지 판단해주는거로 봐도되는거임 즉 기울기와 상수값 즉 웨이트와 바이어스가 비슷비슷하게 나오는지 판단해준다고 보면됨.

자세히 말하면 모델의 평가 지표(metrics)는 모델이 주어진 작업을 얼마나 잘 수행하는지를 측정하고 판단하는데 도움을 줍니다. 이 지표들은 모델의 성능을 다양한 측면에서 평가하며, 예측 결과의 품질을 확인하는 데 사용됩니다.
일반적으로, 모델의 출력(예측)과 실제 레이블(타겟) 간의 비교를 통해 이러한 평가 지표들이 계산됩니다. 예를 들어, 정확도(accuracy)의 경우는 모델이 전체 데이터셋에서 얼마나 정확하게 예측을 수행했는지를 나타냅니다. 다른 지표들도 각각의 관점에서 모델의 성능을 평가합니다.
훈련이나 검증 과정에서 이러한 지표들을 모니터링하여 모델의 학습이 제대로 진행되고 있는지 확인할 수 있습니다. 이러한 평가 지표들을 통해 모델을 조정하고 향상시킬 수 있습니다.


이렇게 설정된 모델은 주어진 데이터셋을 사용하여 훈련되며, 훈련 과정에서는 Mean Squared Error를 최소화하는 방향으로 Adam 옵티마이저를 사용합니다. 훈련 중에는 정확도를 모니터링하여 모델의 성능을 추적합니다.

Adam과 RMSprop은 둘 다 경사 하강법을 기반으로 하는 최적화 알고리즘으로, 주로 신경망 모델을 학습할 때 사용됩니다. 각각의 특징에 따라 어떤 알고리즘을 사용할지 결정할 수 있습니다.

Adam을 사용하는 경우:
다양한 데이터 및 문제에 대응: Adam은 다양한 종류의 데이터와 문제에 대해 효과적으로 작동하는 경향이 있습니다. 따라서 다양한 응용 분야에서 사용할 수 있습니다.
초기 학습 단계에서 효과적: Adam은 초기 학습 단계에서 특히 효과적일 수 있습니다. 적응적 학습률과 모멘텀 개념을 결합하여 초기 수렴 속도를 높일 수 있습니다.

RMSprop을 사용하는 경우:
제한된 컴퓨팅 자원: RMSprop은 메모리 사용량이 적은 특징이 있어, 자원이 제한된 환경에서 효과적일 수 있습니다.
강한 편향 제어: RMSprop은 각각의 파라미터에 대한 학습률을 조절하는 데 있어서 이전 그래디언트 제곱값을 이용하므로, 강한 편향을 효과적으로 제어할 수 있습니다.
일반적인 사용 권장 사항:

일반적으로는 Adam이 RMSprop보다 더 일반적으로 사용됩니다.
특별한 경우(메모리 사용 제한 등)를 제외하고는 두 알고리즘 중 하나를 선택하는 것이 일반적입니다.
모델을 학습할 때 성능을 비교하여 어떤 알고리즘이 더 효과적인지 확인하는 것이 좋습니다.
"""

fit_hist = model.fit(training_data, target_data, epochs=500, verbose=1)

plt.plot(fit_hist.history['loss'])
plt.show()

inp = list(map(int, input().spilt()))
qwe = np.array(inp)
print('입력값')
qwe = qwe.reshape(1, 2)
print('결과 값', model.predict(qwe)[0][0].round())