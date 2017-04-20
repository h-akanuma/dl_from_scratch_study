require 'numo/narray'

class SGD
  def initialize(lr: 0.01)
    @lr = lr
  end

  def update(params:, grads:)
    params.keys.each do |key|
      params[key][0] -= @lr * grads[key]
    end
  end
end

class Momentum
  def initialize(lr: 0.01, momentum: 0.9)
    @lr = lr
    @momentum = momentum
    @v = nil
  end

  def update(params:, grads:)
    if @v.nil?
      @v = {}
      params.each do |key, value|
        @v[key] = Numo::DFloat.zeros(value.first.shape)
      end
    end

    params.keys.each do |key|
      @v[key] = @momentum * @v[key] - @lr * grads[key]
      params[key][0] += @v[key]
    end
  end
end

class AdaGrad
  def initialize(lr: 0.01)
    @lr = lr
    @h = nil
  end

  def update(params:, grads:)
    if @h.nil?
      @h = {}
      params.each do |key, value|
        @h[key] = Numo::DFloat.zeros(value.first.shape)
      end
    end

    params.keys.each do |key|
      @h[key] += grads[key] * grads[key]
      params[key][0] -= @lr * grads[key] / (Numo::DFloat::Math.sqrt(@h[key]) + 1e-7)
    end
  end
end

class Adam
  def initialize(lr: 0.001, beta1: 0.9, beta2: 0.999)
    @lr = lr
    @beta1 = beta1
    @beta2 = beta2
    @iter = 0
    @m = nil
    @v = nil
  end

  def update(params:, grads:)
    if @m.nil?
      @m = {}
      @v = {}
      params.each do |key, value|
        @m[key] = Numo::DFloat.zeros(value.first.shape)
        @v[key] = Numo::DFloat.zeros(value.first.shape)
      end
    end

    @iter += 1
    lr_t = @lr * Numo::DFloat::Math.sqrt(1.0 - @beta2 ** @iter) / (1.0 - @beta1 ** @iter)

    params.keys.each do |key|
      @m[key] += (1 - @beta1) * (grads[key] - @m[key])
      @v[key] += (1 - @beta2) * (grads[key] ** 2 - @v[key])

      params[key][0] -= lr_t * @m[key] / (Numo::DFloat::Math.sqrt(@v[key]) + 1e-7)
    end
  end
end
