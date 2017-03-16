# 3層ニューラルネットワーク実装
# 各層のニューロン数は下記の通り
#  第0層（入力層）：2
#  第1層（隠れ層）：3
#  第2層（隠れ層）：2
#  第3層（出力層）：2

require 'numo/narray'

# 活性化関数としてシグモイド関数を使う
def sigmoid(x)
  1 / (1 + Numo::DFloat::Math.exp(-x))
end

# 出力層の活性化関数として恒等関数を使う
def identity_function(x)
  x
end

# 重みとバイアスの初期化
def init_network
  network = {}
  network['w1'] = Numo::DFloat[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
  network['b1'] = Numo::DFloat[0.1, 0.2, 0.3]
  network['w2'] = Numo::DFloat[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
  network['b2'] = Numo::DFloat[0.1, 0.2]
  network['w3'] = Numo::DFloat[[0.1, 0.3], [0.2, 0.4]]
  network['b3'] = Numo::DFloat[0.1, 0.2]
  network
end

# 入力から出力までの処理
def forward(network, x)
  w1 = network['w1']
  w2 = network['w2']
  w3 = network['w3']
  b1 = network['b1']
  b2 = network['b2']
  b3 = network['b3']

  a1 = x.dot(w1) + b1
  z1 = sigmoid(a1)
  a2 = z1.dot(w2) + b2
  z2 = sigmoid(a2)
  a3 = z2.dot(w3) + b3
  identity_function(a3)
end

network = init_network     # 重みとバイアスのデータを用意
x = Numo::DFloat[1.0, 0.5] # 入力層として2つのニューロンを用意
y = forward(network, x)    # 入力層、重み、バイアスのデータを渡して出力層のデータを取得
puts y.inspect             # 出力層のデータを表示
