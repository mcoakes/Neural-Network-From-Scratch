# Neural-Network-From-Scratch
A Neural Network built from scratch in Python.  

This is my attempt at building a neural network using minimal libraries.  
Currently, you are able to create a network and run a feedforward using sigmoid 
&nbsp;

&nbsp;

## Libraries used:
- numpy
- collections.deque
&nbsp;

&nbsp;

## Example Use

```
user@hostname:~/Neural-Network-From-Scratch$ python3 -i network.py
>>> in_arry = np.array([1,2,3,4,5],dtype=int)
>>> n_hid = 2
>>> h_shapes = [(4,),(5,)]
>>> o_shape = (3,)
>>> Net = NNetwork(inputs=in_arry, num_hidden=n_hid, hidden_shapes=h_shapes, out_shape=o_shape)
>>> Net.feed_forward()
```
