require 'numo/narray'

def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  batch_size = y.shape[0]
  -(t * (Numo::DFloat::Math.log(y))).sum / batch_size # one-hot表現用
end
