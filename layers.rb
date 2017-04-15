require './sigmoid.rb'
require './softmax.rb'
require './cross_entropy_error.rb'

class Relu
  def initialize
    @mask = nil
  end

  def forward(x:)
    @mask = (x <= 0)
    out = x.copy
    out[@mask] = 0
    out
  end

  def backward(dout:)
    dout[@mask] = 0
    dout
  end
end

class Sigmoid
  def initialize
    @out = nil
  end

  def forward(x:)
    @out = sigmoid(x)
    @out
  end

  def backword(dout:)
    dout * (1.0 - @out) * @out
  end
end

class Affine
  def initialize(w:, b:)
    @w = w
    @b = b

    @x = nil
    @original_x_shape = nil

    # 重み・バイアスパラメータの微分
    @dw = nil
    @db = nil
  end

  def dw
    @dw
  end

  def db
    @db
  end

  def forward(x:)
    # テンソル対応
    @original_x_shape = x.shape
    x = x.reshape(x.shape[0], nil)
    @x = x

    @x.dot(@w.first) + @b.first
  end

  def backward(dout:)
    dx = dout.dot(@w.first.transpose)
    @dw = @x.transpose.dot(dout)
    @db = dout.sum(0)

    dx.reshape(*@original_x_shape)
  end
end

class SoftmaxWithLoss
  def initialize
    @loss = nil
    @y = nil # softmaxの出力
    @t = nil # 教師データ
  end

  def forward(x:, t:)
    @t = t
    @y = softmax(x)
    @loss = cross_entropy_error(@y, @t)

    @loss
  end

  def backward(dout: 1)
    batch_size = @t.shape[0]
    if @t.size == @y.size # 教師データがon-hot-vectorの場合
      return (@y - @t) / batch_size
    end

    dx = @y.copy

    (0..(batch_size - 1)).to_a.zip(@t).each do |index_array|
      dx[*index_array] -= 1
    end

    dx / batch_size
  end
end
