require 'numo/narray'
require './layers.rb'

class MultiLayerNet
  def initialize(input_size:, hidden_size_list:, output_size:, activation: :relu, weight_init_std: :relu, weight_decay_lambda: 0)
    @input_size          = input_size
    @output_size         = output_size
    @hidden_size_list    = hidden_size_list
    @hidden_layer_num    = hidden_size_list.size
    @weight_decay_lambda = weight_decay_lambda
    @params              = {}

    # 重みの初期化
    init_weight(weight_init_std)

    # レイヤの生成
    activation_layer = {
      sigmoid: Sigmoid,
      relu:    Relu
    }
    @layers = {}
    (1..@hidden_layer_num).each do |idx|
      @layers["Affine#{idx}"] = Affine.new(w: @params["w#{idx}"], b: @params["b#{idx}"])
      @layers["Activation_function#{idx}"] = activation_layer[activation].new
    end

    idx = @hidden_layer_num + 1
    @layers["Affine#{idx}"] = Affine.new(w: @params["w#{idx}"], b: @params["b#{idx}"])

    @last_layer = SoftmaxWithLoss.new
  end

  def params
    @params
  end

  def init_weight(weight_init_std)
    all_size_list = [@input_size] + @hidden_size_list + [@output_size]
    (1..(all_size_list.size - 1)).each do |idx|
      scale = weight_init_std
      if %i(relu he).include?(weight_init_std)
        scale = Numo::DFloat::Math.sqrt(2.0 / all_size_list[idx - 1])
      elsif %i(sigmoid xavier).include?(weight_init_std)
        scale = Numo::DFloat::Math.sqrt(1.0 / all_size_list[idx - 1])
      end

      Numo::NArray.srand
      @params["w#{idx}"] = [scale * Numo::DFloat.new(all_size_list[idx - 1], all_size_list[idx]).rand_norm]
      @params["b#{idx}"] = [Numo::DFloat.zeros(all_size_list[idx])]
    end
  end

  def predict(x:)
    @layers.values.inject(x) do |x, layer|
      x = layer.forward(x: x)
    end
  end

  # x: 入力データ, t: 教師データ
  def loss(x:, t:)
    y = predict(x: x)

    weight_decay = 0
    (1..(@hidden_layer_num + 1)).each do |idx|
      w = @params["w#{idx}"].first
      weight_decay += 0.5 * @weight_decay_lambda * (w ** 2).sum
    end
    @last_layer.forward(x: y, t: t) + weight_decay
  end

  def accuracy(x:, t:)
    y = predict(x: x)
    y = y.max_index(1) % 10
    if t.ndim != 1
      t = t.max_index(1) % 10
    end

    y.eq(t).cast_to(Numo::UInt16).sum / x.shape[0].to_f
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

    grads = {}
    (1..(@hidden_layer_num + 1)).each do |idx|
      grads["w#{idx}"] = @layers["Affine#{idx}"].dw + @weight_decay_lambda * @layers["Affine#{idx}"].w.first
      grads["b#{idx}"] = @layers["Affine#{idx}"].db
    end

    grads
  end
end
