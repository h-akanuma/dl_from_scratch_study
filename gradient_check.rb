require './mnist.rb'
require './two_layer_net.rb'

x_train, t_train, x_test, t_test = load_mnist(normalize: true, one_hot_label: true)

network = TwoLayerNet.new(input_size: 784, hidden_size: 50, output_size: 10)

x_batch = x_train[0..2, true]
t_batch = t_train[0..2, true]

grad_numerical = network.numerical_gradients(x: x_batch, t: t_batch)
grad_backprop = network.gradient(x: x_batch, t: t_batch)

grad_numerical.keys.each do |key|
  diff = (grad_backprop[key] - grad_numerical[key]).abs.mean
  puts "#{key}: #{diff}"
end
