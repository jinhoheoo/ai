# -*- coding: utf-8 -*-
"""exam02_principles_of_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-qiUn06iVtXn3QRafUJ3xEtO0nM2TvT4

##그래프 연산 (미분,mse를 다룸)

미분은 체인룰때문에 해봄
원래의 웨이트값에서 mse웨이트값을 빼주면 됨 이때 mse값에 러닝레이트를 곱해줘서 발산하는걸 막음
그리고 결과값에 영향 주는게 다 다르기 때문에 각각의 웨이트와 바이어스의 미분값이 나

즉 기계학습의 원리는 덧셈과 곱셈만 쓰이는 1차식으로 된 앞에서 말한 웨이트와 바이어스가 갱신되는 거임
weight = weight - learning_rate * weight_mse_gradient * np.average(dweight)
bias = bias - learning_rate * bias_mse_gradient * dbias



이 페이지에 적은 코드를 파이썬에서 사용해보려함.
pip install jupyter이거 cmd 들어가서 설치하고

model.compile(loss='mse', optimizer='rmsprop')  #mse 에러를 제곱해서 평균하는거임 그것을 loss로 하는데 loss가 작아지게끔 학습을 함.
모델한테 섭씨온도를 줘서 화씨로 만든 예측값을 받는데 그것을 정답값과 비교해서 오차 구한 후 제곱해서 절대값으로 만든후 평균낸게 loss임
처음에 웨이트를 랜덤하게 줘서 해당 웨이트에서 미분해서
경사하강법 미분을해서 미분값이 +이면 더해나가고 -이면 빼나가는 거임.
이런식으로 값을 찾아나가는거임. 근데 이렇게 더하고 빼다 보면 엄청 크게 무한대 까지 값이 커지는 발산 할 수 있는데 이럴 방지하기 위해 런닝레이트를 곱해주는데 0.01 이렇게 곱해줘서 조금씩 하게 함 어떻게하냐면 미분값에 런닝레이트를 곱해서 빼주면 됨.

간단한 선형 회귀 모델을 예로 들어 설명하면, 모델의 출력은 다음과 같습니다:

출력= 입력×웨이트 + 바이어스

이것은 간단한 모델이지만, 웨이트와 바이어스의 개념은 더 복잡한 신경망에서도 동일하게 사용됩니다. 웨이트와 바이어스는 학습 데이터에 맞게 조정되어 모델이 원하는 출력을 생성할 수 있도록 합니다.

웨이트(Weight):
역할: 모델이 입력에서 출력까지의 변환을 결정하는 매개변수입니다.
설명: 웨이트는 각 입력 피처에 대한 가중치를 나타냅니다. 모델은 입력과 각 웨이트를 곱한 후 이를 조합하여 출력을 생성합니다. 웨이트가 크면 해당 피처가 출력에 미치는 영향이 커지고, 작으면 미치는 영향이 작아집니다.

바이어스(Bias):
역할: 모델이 입력에서 출력까지의 변환을 조절하는 추가적인 매개변수입니다.
설명: 바이어스는 각 출력에 더해지는 상수입니다. 이를 통해 모델이 특정 입력 값에 대해 출력을 이동시키거나 변형시킬 수 있습니다. 바이어스는 특정 패턴이나 경향성을 학습하는 데 도움을 줄 수 있습니다.
"""

class add_graph:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = 1 * dout
        dy = 1 * dout
        return dx, dy

class mul_graph:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy

"""여러개의 인스턴스로 값 넣으려 할 때 즉 여러개의 x와 y를 만들기 위해 self를 씀
self를 쓰면서 객체가 가지고 있는 값을 표시해주는거임.
"""

#이제 위에 적은 모델을 기반으로 우리가 쓰는 mse 만들어보려함 이거 keras 거기에 다 있는 거니까 이렇게 적을 일 앞으로 없음 그냥 내부 어떤식으로 되어있는지 이해하려고 작성함
class mse_graph:
  def __init__(self):
    self.loss = None
    self.y = None     #예측값
    self.t = None     #모델이 맞춰야하는 정답
    self.x = None
  def forward(self, y, t):
    self.t = t
    self.y = y
    self.loss = np.square(self.t - self.y).sum() / self.t.shape[0]  #square(self.t - self.y) 이게 제곱 정답-예측값의 제곱  이것들의 합을 더해서 개수를 나눠 즉 평균냄
    return self.loss
  def backward(self, x, dout=1):
    data_size = self.t.shape[0]
    dweight_mse = (((self.y - self.t) * x).sum() * 2 /data_size)
    dbias_mse = (self.y - self.t).sum() * 2 / data_size
    return dweight_mse, dbias_mse

apple = 100
apple_num = 3
orange = 150
orange_num = 3
tax = 1.1

mul_apple_graph = mul_graph()
mul_orange_graph = mul_graph()
add_apple_orange_graph = add_graph()
mul_tax_graph  = mul_graph()

apple_price = mul_apple_graph.forward(apple, apple_num)
orange_price = mul_orange_graph.forward(orange, orange_num)
all_price = add_apple_orange_graph.forward(apple_price, orange_price)
total_price = mul_tax_graph.forward(all_price, tax)
print(total_price)

"""매서드 호출할 때 self 부분을 따로 호출하지 않아도 됨.
즉 self를 직접 명시하지 않습니다. 파이썬이 자동으로 obj를 self로 전달합니다.

float 숫자를 2진수로 정확하게 저장할 수 없어서 00001 이렇게 나옴 이게 양자화오류? 그거임
"""

dprice = 1
dall_price, dtax = mul_tax_graph.backward(dprice)
dapple_price, dorange_price = add_apple_orange_graph.backward(dall_price)
dorange, dorange_num = mul_orange_graph.backward(dorange_price)
dapple, dapple_num = mul_apple_graph.backward(dapple_price)
print('dApple', dapple)
print('dApple_num', dapple_num)
print('dOrange', dorange)
print('dOrange_num', dapple_num)

"""이러한 방식이 체인룰(연쇄방식)임 dz/dx= (dz/dy)*(dy/dx) 이거임.

역전파가 체인룰가지고 미분한거임
"""

#모델이 어떻게 쓰이는지 이해하면 하면됨 다 만들어져 있어서 다음에는 이용하기만 하면 됨.
import numpy as np

def celcius_to_fahrenheit(x):
  return x * 1.8 + 32          #웨이트가 1.8 바이어스가 0.32

#모델 만드는거임(keras 안쓴거임)
weight = np.random.uniform(0, 5, 1) #uniform을 쓰면 0에서 5사이에 어떤값이 나올 확률이 다 동일함 즉 4.5가 나올 확률과 4.1이 나올 확률이 같음
print(weight)
bias = 0

#위의 모델을 기반으로 학습을 시킴
data_C = np.arange(0,100)
data_F = celcius_to_fahrenheit(data_C)
scaled_data_C = data_C / 100
scaled_data_F = data_F / 100
print(scaled_data_C)
print(scaled_data_F)

weight_graph = mul_graph()
bias_graph = add_graph()

weighted_data = weight_graph.forward(weight, scaled_data_C)
predict_data = bias_graph.forward(weighted_data, bias)
print(predict_data)

dout = 1
dbias, dbiased_data = bias_graph.backward(dout)
dweight, dscaled_data_C = weight_graph.backward(dbiased_data)
print(dbias)
print(dweight)

mseGraph = mse_graph()
mse = mseGraph.forward(predict_data, scaled_data_F)
print(mse)

weight_mse_gradient, bias_mse_gradient = mseGraph.backward(scaled_data_C)
print(weight_mse_gradient)
print(bias_mse_gradient)

learning_rate = 0.1
learned_weight = weight - learning_rate * weight_mse_gradient *np.average(dweight)  # np이거 weight의 미분값 곱해준거임
print('before learning weight :', weight)
print('after learning weight :', learned_weight)

#바이어스 학습시키려함
learned_bias = bias - learning_rate * bias_mse_gradient * dbias
print('before learning weight :', bias)
print('after learning weight :', learned_bias)

"""이렇게 한번 학습시킨거임."""

#1000번 학습시켜서 정답찾아내려함.

error_list = []
weight_list = []
bias_list = []        #빈 리스트 만들어서 그때그때의 웨이트 바이어스값 저장하려함

for i in range(1000):
  #forward 이걸로 예측값 구함
  weighted_data = weight_graph.forward(weight, scaled_data_C)
  predict_data = bias_graph.forward(weighted_data, bias)

  #backward 이걸로 미분값 구함
  dout = 1
  dbias, dweighted_data = bias_graph.backward(dout)
  dweight, dscaled_data_C = weight_graph.backward(dweighted_data)

  #mse
  mse = mseGraph.forward(predict_data, scaled_data_F)
  error_list.append(mse)
  weight_mse_gradient, bias_mse_gradient = mseGraph.backward(scaled_data_C)

  #learning
  weight_list.append(weight)
  weight = weight - learning_rate * weight_mse_gradient * np.average(dweight)
  bias_list.append(bias)
  bias = bias - learning_rate * bias_mse_gradient * dbias

weight_list.append(weight)
bias_list.append(bias)
print(weight)
print(bias)

print(error_list[-1])

import matplotlib.pyplot as plt
plt.plot(bias_list)
plt.show()

plt.plot(weight_list)
plt.show()

plt.plot(error_list)
plt.show()