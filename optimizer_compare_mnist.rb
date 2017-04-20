require 'numo/gnuplot'
require './mnist.rb'
require './optimizers.rb'
require './multi_layer_net.rb'

# 0: MNISTデータの読み込み
x_train, t_train, x_test, t_test = load_mnist(normalize: true)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 1500

# 1: 実験の設定
optimizers = {
  sgd:      SGD.new,
  momentum: Momentum.new,
  adagrad:  AdaGrad.new,
  adam:     Adam.new
}

networks = {}
train_loss = {}
optimizers.each do |key, optimizer|
  networks[key]   = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100], output_size: 10)
  train_loss[key] = []
end

# 2: 訓練の開始
max_iterations.times do |i|
  Numo::NArray.srand
  batch_mask = Numo::Int32.new(batch_size).rand(0, train_size)
  x_batch = x_train[batch_mask, true]
  t_batch = t_train[batch_mask]

  optimizers.each do |key, optimizer|
    grads = networks[key].gradient(x: x_batch, t: t_batch)
    optimizers[key].update(params: networks[key].params, grads: grads)

    loss = networks[key].loss(x: x_batch, t: t_batch)
    train_loss[key] << loss
  end

  next unless i % 100 == 0

  puts "========== iteration: #{i} =========="
  optimizers.keys.each do |key|
    loss = networks[key].loss(x: x_batch, t: t_batch)
    puts "#{key}: #{loss}"
  end
end

# 3: グラフの描画
x = (0..(max_iterations - 1)).to_a
Numo.gnuplot do
  set xlabel: 'iterations'
  set ylabel: 'loss'
  set yrange: 0...1
  plot x, train_loss[:sgd],      { w: :lines, t: 'SGD',      lc_rgb: 'green',  lw: 1 },
       x, train_loss[:momentum], { w: :lines, t: 'Momentum', lc_rgb: 'orange', lw: 1 },
       x, train_loss[:adagrad],  { w: :lines, t: 'AdaGrad',  lc_rgb: 'red',    lw: 1 },
       x, train_loss[:adam],     { w: :lines, t: 'Adam',     lc_rgb: 'blue',   lw: 1 }
end
