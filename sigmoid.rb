require 'numo/narray'

def sigmoid(x)
  1 / (1 + Numo::DFloat::Math.exp(-x))
end
