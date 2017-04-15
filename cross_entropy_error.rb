require 'numo/narray'

def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
  if t.size == y.size
    t = t.max_index(1) % 10
  end

  batch_size = y.shape[0]
  target_data = (0..(batch_size - 1)).to_a.zip(t).map do |index_array|
    y[*index_array]
  end
  -Numo::DFloat::Math.log(target_data).sum / batch_size
end
