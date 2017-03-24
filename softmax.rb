require 'numo/narray'

def softmax(a)
  c = a.max
  exp_a = Numo::DFloat::Math.exp(a - c)
  sum_exp_a = exp_a.sum
  exp_a / sum_exp_a
end

#a = Numo::DFloat[0.3, 2.9, 4.0]
#y = softmax(a)
#puts y.inspect
#puts y.sum
