#!/bin/bash

gem install daru
gem install rbczmq
gem install numo-gnuplot

git clone git://github.com/ruby-numo/narray
cd narray
gem build numo-narray.gemspec
gem install numo-narray-0.9.0.3.gem
