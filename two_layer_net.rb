require 'numo/narray'
require './numerical_gradient.rb'
require './layers.rb'

class TwoLayerNet
  def initialize(input_size:, hidden_size:, output_size:, weight_init_std: 0.01)
    # 重みの初期化
    Numo::NArray.srand
    @params = {
      w1: [weight_init_std * Numo::DFloat.new(input_size, hidden_size).rand_norm],
      b1: [Numo::DFloat.zeros(hidden_size)],
      w2: [weight_init_std * Numo::DFloat.new(hidden_size, output_size).rand_norm],
      b2: [Numo::DFloat.zeros(output_size)]
    }

    # レイヤの生成
    @layers = {
      affine1: Affine.new(w: @params[:w1], b: @params[:b1]),
      relu1:   Relu.new,
      affine2: Affine.new(w: @params[:w2], b: @params[:b2])
    }
    @last_layer = SoftmaxWithLoss.new
  end

  def params
    @params
  end

  def predict(x:)
    @layers.values.inject(x) do |x, layer|
      x = layer.forward(x: x)
    end
  end

  # x: 入力データ, t: 教師データ
  def loss(x:, t:)
    y = predict(x: x)
    @last_layer.forward(x: y, t: t)
  end

  def accuracy(x:, t:)
    y = predict(x: x)
    y = y.max_index(1) % 10
    if t.ndim != 1
      t = t.max_index(1) % 10
    end

    y.eq(t).cast_to(Numo::UInt16).sum / x.shape[0].to_f
  end

  def numerical_gradients(x:, t:)
    loss_w = lambda { loss(x: x, t: t) }

    {
      w1: numerical_gradient(loss_w, @params[:w1].first),
      b1: numerical_gradient(loss_w, @params[:b1].first),
      w2: numerical_gradient(loss_w, @params[:w2].first),
      b2: numerical_gradient(loss_w, @params[:b2].first)
    }
  end

  def gradient(x:, t:)
    # forward
    loss(x: x, t: t)

    # backward
    dout = 1
    dout = @last_layer.backward(dout: dout)

    layers = @layers.values.reverse
    layers.inject(dout) do |dout, layer|
      dout = layer.backward(dout: dout)
    end

    {
      w1: @layers[:affine1].dw,
      b1: @layers[:affine1].db,
      w2: @layers[:affine2].dw,
      b2: @layers[:affine2].db
    }
  end
end
