require 'numo/narray'
require 'json'
require './sigmoid.rb'
require './softmax.rb'
require './mnist.rb'

def get_data
  x_train, t_train, x_test, t_test = load_mnist(true, true, false)
  [x_test, t_test]
end

def init_network
  nw = JSON.load(File.read('sample_weight.json'))
  network = {}
  nw.each do |k, v|
    network[k.to_sym] = Numo::DFloat[*v]
  end
  network
end

def predict(network, x)
  w1 = network[:w1]
  w2 = network[:w2]
  w3 = network[:w3]
  b1 = network[:b1]
  b2 = network[:b2]
  b3 = network[:b3]

  a1 = x.dot(w1) + b1
  z1 = sigmoid(a1)
  a2 = z1.dot(w2) + b2
  z2 = sigmoid(a2)
  a3 = z2.dot(w3) + b3
  softmax(a3)
end

x, t = get_data
network = init_network

batch_size = 100
accuracy_cnt = 0
x.to_a.each_slice(batch_size).with_index do |x_batch, idx|
  y_batch = predict(network, Numo::DFloat[*x_batch])
  p = y_batch.max_index(1) % 10
  accuracy_cnt += p.eq(t[(idx * batch_size)..(idx * batch_size + (batch_size - 1))]).cast_to(Numo::UInt8).sum
end

puts "Accuracy: #{accuracy_cnt.to_f / x.shape[0]}"
