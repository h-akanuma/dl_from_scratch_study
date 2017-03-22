require 'rmagick'
require './mnist.rb'

x_train, t_train, x_test, t_test = load_mnist

img = x_train[0, true]
label = t_train[0]
puts img.shape
puts img.max
puts img.min
puts label

image = Magick::Image.new(28, 28)
image.import_pixels(0, 0, 28, 28, 'I', img, Magick::FloatPixel)
image.display
