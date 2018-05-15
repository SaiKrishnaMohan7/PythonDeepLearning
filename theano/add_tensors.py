#! usr/bin/env python3

import theano
from theano import tensor
from theano import function

a = tensor.dscalar()
b = tensor.dscalar()

c = a + b

# Make above expression callable
add = function([a, b], c)

result = add(2, 3)

print(result)