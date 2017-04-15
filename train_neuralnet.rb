require 'numo/narray'
require 'numo/gnuplot'
require './mnist.rb'
require './two_layer_net.rb'

# データの読み込み
x_train, t_train, x_test, t_test = load_mnist(normalize: true, one_hot_label: true)

network = TwoLayerNet.new(input_size: 784, hidden_size: 50, output_size: 10)

iters_num = 10_000 # 繰り返し回数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = [train_size / batch_size, 1].max

iters_num.times do |i|
  Numo::NArray.srand
  batch_mask = Numo::Int32.new(batch_size).rand(0, train_size)
  x_batch = x_train[batch_mask, true]
  t_batch = t_train[batch_mask, true]

  # 勾配の計算
  #grad = network.numerical_gradients(x_batch, t_batch)
  grad = network.gradient(x: x_batch, t: t_batch)

  # パラメータの更新
  %i(w1 b1 w2 b2).each do |key|
    network.params[key][0] -= learning_rate * grad[key]
  end

  loss = network.loss(x: x_batch, t: t_batch)
  train_loss_list << loss

  next if i % iter_per_epoch != 0

  train_acc = network.accuracy(x: x_train, t: t_train)
  test_acc = network.accuracy(x: x_test, t: t_test)
  train_acc_list << train_acc
  test_acc_list << test_acc
  puts "train acc, test acc | #{train_acc}, #{test_acc}"
end

# グラフの描画
x = (0..(train_acc_list.size - 1)).to_a
Numo.gnuplot do
  plot x, train_acc_list, { w: :lines, t: 'train acc', lc_rgb: 'blue', lw: 5 },
       x, test_acc_list, { w: :lines, t: 'test acc', lc_rgb: 'green', lw: 5 }
  set xlabel: 'epochs'
  set ylabel: 'accuracy'
  set yrange: 0..1
end
