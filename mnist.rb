require 'open-uri'
require 'zlib'
require 'fileutils'
require 'numo/narray'

URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
KEY_FILES = {
  train_img:   'train-images-idx3-ubyte.gz',
  train_label: 'train-labels-idx1-ubyte.gz',
  test_img:    't10k-images-idx3-ubyte.gz',
  test_label:   't10k-labels-idx1-ubyte.gz'
}

DATASET_DIR = "#{File.dirname(__FILE__)}/dataset"
SAVE_FILE = "#{DATASET_DIR}/mnist.dump"

IMG_SIZE = 784

def download(file_name)
  puts "Downloading #{file_name} ..."
  open(URL_BASE + file_name) do |file|
    open("#{DATASET_DIR}/#{file_name}", 'w+b') do |out|
      out.write(file.read)
    end
  end
  puts "Done."
end

def download_mnist
  FileUtils.mkdir_p(DATASET_DIR)
  KEY_FILES.each do |k, file|
    download(file)
  end
end

def load_img(file_name)
  puts "Converting #{file_name} to NArray ..."
  data = nil
  Zlib::GzipReader.open("#{DATASET_DIR}/#{file_name}") do |gz|
    data = gz.each_byte.to_a[16..-1].each_slice(IMG_SIZE).to_a
    data = Numo::UInt8[*data]
  end
  puts "Done"
  data
end

def load_label(file_name)
  puts "Converting #{file_name} to NArray ..."
  data = nil
  Zlib::GzipReader.open("#{DATASET_DIR}/#{file_name}") do |gz|
    data = Numo::UInt8[*gz.each_byte.to_a[8..-1]]
  end
  puts "Done"
  data
end

def convert_narray
  dataset = {}
  dataset[:train_img] = load_img(KEY_FILES[:train_img])
  dataset[:train_label] = load_label(KEY_FILES[:train_label])
  dataset[:test_img] = load_img(KEY_FILES[:test_img])
  dataset[:test_label] = load_label(KEY_FILES[:test_label])
  dataset
end

def init_mnist
  download_mnist
  dataset = convert_narray
  puts "Creating dump file ..."
  File.write(SAVE_FILE, Marshal.dump(dataset))
  puts "Done!"
end

def change_one_hot_label(x)
  one_hot_arrays = x.to_a.map do |v|
    one_hot_array = Array.new(10, 0)
    one_hot_array[v] = 1
    one_hot_array
  end
  Numo::UInt8[*one_hot_arrays]
end

def load_mnist(normalize = true, flatten = true, one_hot_label = false)
  unless File.exist?(SAVE_FILE)
    init_mnist
  end

  dataset = Marshal.load(File.read(SAVE_FILE))

  if normalize
    %i(train_img test_img).each do |key|
      dataset[key] = dataset[key].cast_to(Numo::DFloat)
      dataset[key] /= 255.0
    end
  end

  if one_hot_label
    dataset[:train_label] = change_one_hot_label(dataset[:train_label])
    dataset[:test_label] = change_one_hot_label(dataset[:test_label])
  end

  unless flatten
    %i(train_img test_img).each do |key|
      dataset[key] = dataset[key].reshape(dataset[key].shape[0], 28, 28)
    end
  end

  [dataset[:train_img], dataset[:train_label], dataset[:test_img], dataset[:test_label]]
end
