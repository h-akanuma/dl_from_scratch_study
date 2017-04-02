require 'numo/narray'

def softmax(a)
  if a.ndim == 2
    a = a.transpose
    a = a - a.max(0)
    y = Numo::DFloat::Math.exp(a) / Numo::DFloat::Math.exp(a).sum(0)
    return y.transpose
  end

  c = a.max
  exp_a = Numo::DFloat::Math.exp(a - c)
  sum_exp_a = exp_a.sum
  exp_a / sum_exp_a
end

#a = Numo::DFloat[0.3, 2.9, 4.0]
#y = softmax(a)
#puts y.inspect
#puts y.sum
