# Tensorflow Notes

* Symbolic programming based as opposed to imperative. PyTorch is imperative.
  * A computation graph is defined and then compiled to get the desired result.
  * Example: Imperative v/s Symbolic
    ```python
      import numpy as np

      a = np.ones(10)
      b = np.ones(10) * 4
      c = a * b
      d = c + 1

      print(d)
    ```
    ```python
      import numpy as np

      A = Variable('A')
      B = Variable('B')
      C = A * B # no computation occurs here
      D = C + Constant(1)

      f = compile(D)
      d = f(A=np.ones(10), B=np.ones(10)*2)
    ```
    * Symbolic programs are efficient, inplace computation, memory efficient, tensorflow follows this paradigm

    * Imperative is flexible, Python suitable, Python's native feature can be used - PyTorch follows this paradigm

    * Tensorflow is define and run, Static Computation Graphing?
      * Like writing the whole prog before it could be run.
      * Conditions and iterations are defined in the graph.

    * PyTorch uses Dynamic Computation Graphing, defined by run.
      * Graph structure generated at runtime

    * Tensorflow's computaion graph can be defined once and are reusable
      * Good for fixed size networks, feed forward n/w

    * Variable amount of work, Dyanmic Computaion Graphs, ex: RNN

  * Tensorflow built for production, distributed computing in mind.