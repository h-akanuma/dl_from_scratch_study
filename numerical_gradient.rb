require 'numo/narray'

def numerical_gradient(f, x)
  h = 1e-4
  grad = Numo::DFloat.zeros(x.shape)

  x.size.times do |i|
    tmp_val = x[i]

    x[i] = tmp_val + h
    fxh1 = f.call(x)

    x[i] = tmp_val - h
    fxh2 = f.call(x)

    grad[i] = (fxh1 - fxh2) / (2 * h)
    x[i] = tmp_val
  end

  grad
end
