require 'numo/narray'
require './softmax.rb'
require './sigmoid.rb'
require './cross_entropy_error.rb'
require './numerical_gradient.rb'

class TwoLayerNet
  def initialize(input_size, hidden_size, output_size, weight_init_std = 0.01)
    @params = {}
    @params[:w1] = weight_init_std * Numo::DFloat.new(input_size, hidden_size).rand_norm
    @params[:b1] = Numo::DFloat.zeros(hidden_size)
    @params[:w2] = weight_init_std * Numo::DFloat.new(hidden_size, output_size).rand_norm
    @params[:b2] = Numo::DFloat.zeros(output_size)
  end

  def params
    @params
  end

  def predict(x)
    w1 = @params[:w1]
    w2 = @params[:w2]
    b1 = @params[:b1]
    b2 = @params[:b2]

    a1 = x.dot(w1) + b1
    z1 = sigmoid(a1)
    a2 = z1.dot(w2) + b2
    softmax(a2)
  end

  def loss(x, t)
    y = predict(x)
    cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    y = predict(x)
    y = y.max_index(1) % 10
    t = t.max_index(1) % 10

    y.eq(t).cast_to(Numo::UInt8).sum / x.shape[0].to_f
  end

  def numerical_gradients(x, t)
    loss_w = lambda {|w| loss(x, t) }

    grads = {}
    grads[:w1] = numerical_gradient(loss_w, @params[:w1])
    grads[:b1] = numerical_gradient(loss_w, @params[:b1])
    grads[:w2] = numerical_gradient(loss_w, @params[:w2])
    grads[:b2] = numerical_gradient(loss_w, @params[:b2])

    grads
  end
end
