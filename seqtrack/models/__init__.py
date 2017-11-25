'''
Model has the following interface.
Note that these functions are called to instantiate the TF graph.
At test time, we simply do sess.run() using the resulting tensors.

state = model.start(frame, is_training, enable_loss)
    Args:
        frame is a frame dict
        is_training is a boolean tensor
        enable_loss is a boolean

rect, state, loss = model.next(frame, state)
    Args:
        frame is a frame dict

    The method next() should not modify member variables of the model.
    Instead, it should pass all information to the next frame using state.

    The method next() should only use frame['rect'] during training.

    The loss operations are only added to the graph if enable_loss is true.

frame dict has:
    frame['image'] is an image dict
    frame['rect']
    frame['rect_valid']

image dict has:
    image['base_image']
    image['viewport']

'''
