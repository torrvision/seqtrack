from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages
from tensorflow.python.layers.maxout import maxout

# from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.layers import base
# from tensorflow.python.layers import utils
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import nn
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
# from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


################################################################################
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/layers/python/layers/layers.py


@add_arg_scope
def batch_norm(inputs,
               batch_mean=None,
               batch_variance=None,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=False,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
  Can be used as a normalizer function for conv2d and fully_connected.
  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:
  ```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```
  One can set updates_collections=None to force the updates in place, but that
  can have a speed penalty, especially in distributed settings.
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused: if `True`, use a faster, fused implementation if possible.
      If `None`, use the system recommended implementation.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if fused is None:
    fused = True

  # Only use _fused_batch_norm if all of the following three
  # conditions are true:
  # (1) fused is set True;
  # (2) it is possible to use (currently it doesn't support batch weights,
  #   renorm, and the case when rank is neither 2 nor 4);
  # (3) it is used with zero_debias_moving_mean, or an input shape of rank 2,
  #   or non-default updates_collections (not implemented in
  #   normalization_layers.BatchNormalization yet); otherwise use the fused
  #   implementation in normalization_layers.BatchNormalization.
  inputs = ops.convert_to_tensor(inputs)
  rank = inputs.get_shape().ndims
  possible_to_fuse = batch_weights is None and not renorm and rank in [2, 4]
  if fused and possible_to_fuse and (
      zero_debias_moving_mean or rank == 2 or
      updates_collections is not ops.GraphKeys.UPDATE_OPS):
    # return _fused_batch_norm(
    #     inputs,
    #     decay=decay,
    #     center=center,
    #     scale=scale,
    #     epsilon=epsilon,
    #     activation_fn=activation_fn,
    #     param_initializers=param_initializers,
    #     updates_collections=updates_collections,
    #     is_training=is_training,
    #     reuse=reuse,
    #     variables_collections=variables_collections,
    #     outputs_collections=outputs_collections,
    #     trainable=trainable,
    #     data_format=data_format,
    #     zero_debias_moving_mean=zero_debias_moving_mean,
    #     scope=scope)
    raise RuntimeError('fused_batch_norm not supported')

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Determine whether we can use the core layer class.
    if (batch_weights is None and
        updates_collections is ops.GraphKeys.UPDATE_OPS and
        not zero_debias_moving_mean):
      # Use the core layer class.
      axis = 1 if data_format == DATA_FORMAT_NCHW else -1
      if not param_initializers:
        param_initializers = {}
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      if not param_regularizers:
        param_regularizers = {}
      beta_regularizer = param_regularizers.get('beta')
      gamma_regularizer = param_regularizers.get('gamma')
      # layer = normalization_layers.BatchNormalization(
      layer = BatchNormalization(
          axis=axis,
          momentum=decay,
          epsilon=epsilon,
          center=center,
          scale=scale,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer,
          moving_mean_initializer=moving_mean_initializer,
          moving_variance_initializer=moving_variance_initializer,
          beta_regularizer=beta_regularizer,
          gamma_regularizer=gamma_regularizer,
          trainable=trainable,
          renorm=renorm,
          renorm_clipping=renorm_clipping,
          renorm_momentum=renorm_decay,
          name=sc.name,
          _scope=sc,
          _reuse=reuse,
          fused=fused)
      outputs, batch_mean, batch_variance = layer.apply(inputs, batch_mean, batch_variance, training=is_training)

      # Add variables to collections.
      _add_variable_to_collections(
          layer.moving_mean, variables_collections, 'moving_mean')
      _add_variable_to_collections(
          layer.moving_variance, variables_collections, 'moving_variance')
      if layer.beta is not None:
        _add_variable_to_collections(layer.beta, variables_collections, 'beta')
      if layer.gamma is not None:
        _add_variable_to_collections(
            layer.gamma, variables_collections, 'gamma')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return utils.collect_named_outputs(outputs_collections, sc.name, outputs), batch_mean, batch_variance

  raise RuntimeError('must use BatchNormalization')

#     # Not supported by layer class: batch_weights argument,
#     # and custom updates_collections. In that case, use the legacy BN
#     # implementation.
#     # Custom updates collections are not supported because the update logic
#     # is different in this case, in particular w.r.t. "forced updates" and
#     # update op reuse.
#     if renorm:
#       raise ValueError('renorm is not supported with batch_weights, '
#                        'updates_collections or zero_debias_moving_mean')
#     inputs_shape = inputs.get_shape()
#     inputs_rank = inputs_shape.ndims
#     if inputs_rank is None:
#       raise ValueError('Inputs %s has undefined rank.' % inputs.name)
#     dtype = inputs.dtype.base_dtype
#     if batch_weights is not None:
#       batch_weights = ops.convert_to_tensor(batch_weights)
#       inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
#       # Reshape batch weight values so they broadcast across inputs.
#       nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
#       batch_weights = array_ops.reshape(batch_weights, nshape)
# 
#     if data_format == DATA_FORMAT_NCHW:
#       moments_axes = [0] + list(range(2, inputs_rank))
#       params_shape = inputs_shape[1:2]
#       # For NCHW format, rather than relying on implicit broadcasting, we
#       # explicitly reshape the params to params_shape_broadcast when computing
#       # the moments and the batch normalization.
#       params_shape_broadcast = list(
#           [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
#     else:
#       moments_axes = list(range(inputs_rank - 1))
#       params_shape = inputs_shape[-1:]
#       params_shape_broadcast = None
#     if not params_shape.is_fully_defined():
#       raise ValueError('Inputs %s has undefined channels dimension %s.' % (
#           inputs.name, params_shape))
# 
#     # Allocate parameters for the beta and gamma of the normalization.
#     beta, gamma = None, None
#     if not param_initializers:
#       param_initializers = {}
#     if center:
#       beta_collections = utils.get_variable_collections(variables_collections,
#                                                         'beta')
#       beta_initializer = param_initializers.get('beta',
#                                                 init_ops.zeros_initializer())
#       beta = variables.model_variable('beta',
#                                       shape=params_shape,
#                                       dtype=dtype,
#                                       initializer=beta_initializer,
#                                       collections=beta_collections,
#                                       trainable=trainable)
#     if scale:
#       gamma_collections = utils.get_variable_collections(variables_collections,
#                                                          'gamma')
#       gamma_initializer = param_initializers.get('gamma',
#                                                  init_ops.ones_initializer())
#       gamma = variables.model_variable('gamma',
#                                        shape=params_shape,
#                                        dtype=dtype,
#                                        initializer=gamma_initializer,
#                                        collections=gamma_collections,
#                                        trainable=trainable)
# 
#     # Create moving_mean and moving_variance variables and add them to the
#     # appropriate collections. We disable variable partitioning while creating
#     # them, because assign_moving_average is not yet supported for partitioned
#     # variables (this needs to be handled carefully, as it may break
#     # the checkpoint backward compatibility).
#     with variable_scope.variable_scope(
#         variable_scope.get_variable_scope()) as local_scope:
#       local_scope.set_partitioner(None)
#       moving_mean_collections = utils.get_variable_collections(
#           variables_collections, 'moving_mean')
#       moving_mean_initializer = param_initializers.get(
#           'moving_mean', init_ops.zeros_initializer())
#       moving_mean = variables.model_variable(
#           'moving_mean',
#           shape=params_shape,
#           dtype=dtype,
#           initializer=moving_mean_initializer,
#           trainable=False,
#           collections=moving_mean_collections)
#       moving_variance_collections = utils.get_variable_collections(
#           variables_collections, 'moving_variance')
#       moving_variance_initializer = param_initializers.get(
#           'moving_variance', init_ops.ones_initializer())
#       moving_variance = variables.model_variable(
#           'moving_variance',
#           shape=params_shape,
#           dtype=dtype,
#           initializer=moving_variance_initializer,
#           trainable=False,
#           collections=moving_variance_collections)
# 
#     # If `is_training` doesn't have a constant value, because it is a `Tensor`,
#     # a `Variable` or `Placeholder` then is_training_value will be None and
#     # `needs_moments` will be true.
#     is_training_value = utils.constant_value(is_training)
#     need_moments = is_training_value is None or is_training_value
#     if need_moments:
#       # Calculate the moments based on the individual batch.
#       if batch_weights is None:
#         if data_format == DATA_FORMAT_NCHW:
#           mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
#           mean = array_ops.reshape(mean, [-1])
#           variance = array_ops.reshape(variance, [-1])
#         else:
#           mean, variance = nn.moments(inputs, moments_axes)
#       else:
#         if data_format == DATA_FORMAT_NCHW:
#           mean, variance = nn.weighted_moments(inputs, moments_axes,
#                                                batch_weights, keep_dims=True)
#           mean = array_ops.reshape(mean, [-1])
#           variance = array_ops.reshape(variance, [-1])
#         else:
#           mean, variance = nn.weighted_moments(inputs, moments_axes,
#                                                batch_weights)
# 
#       moving_vars_fn = lambda: (moving_mean, moving_variance)
#       if updates_collections is None:
#         def _force_updates():
#           """Internal function forces updates moving_vars if is_training."""
#           update_moving_mean = moving_averages.assign_moving_average(
#               moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
#           update_moving_variance = moving_averages.assign_moving_average(
#               moving_variance, variance, decay, zero_debias=False)
#           with ops.control_dependencies([update_moving_mean,
#                                          update_moving_variance]):
#             return array_ops.identity(mean), array_ops.identity(variance)
#         mean, variance = utils.smart_cond(is_training,
#                                           _force_updates,
#                                           moving_vars_fn)
#       else:
#         def _delay_updates():
#           """Internal function that delay updates moving_vars if is_training."""
#           update_moving_mean = moving_averages.assign_moving_average(
#               moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
#           update_moving_variance = moving_averages.assign_moving_average(
#               moving_variance, variance, decay, zero_debias=False)
#           return update_moving_mean, update_moving_variance
# 
#         update_mean, update_variance = utils.smart_cond(is_training,
#                                                         _delay_updates,
#                                                         moving_vars_fn)
#         ops.add_to_collections(updates_collections, update_mean)
#         ops.add_to_collections(updates_collections, update_variance)
#         # Use computed moments during training and moving_vars otherwise.
#         vars_fn = lambda: (mean, variance)
#         mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
#     else:
#       mean, variance = moving_mean, moving_variance
#     if data_format == DATA_FORMAT_NCHW:
#       mean = array_ops.reshape(mean, params_shape_broadcast)
#       variance = array_ops.reshape(variance, params_shape_broadcast)
#       if beta is not None:
#         beta = array_ops.reshape(beta, params_shape_broadcast)
#       if gamma is not None:
#         gamma = array_ops.reshape(gamma, params_shape_broadcast)
# 
#     # Compute batch_normalization.
#     outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
#                                      epsilon)
#     outputs.set_shape(inputs_shape)
#     if activation_fn is not None:
#       outputs = activation_fn(outputs)
#     return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, collections=collections, trainable=trainable,
      caching_device=caching_device, partitioner=partitioner,
      custom_getter=getter, use_resource=use_resource)


def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""
  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)
  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = utils.get_variable_collections(
      collections_set, collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf_variables.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in ops.get_collection(collection):
        ops.add_to_collection(collection, var)


################################################################################
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/layers/normalization.py


class BatchNormalization(base.Layer):
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.
  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"
  Sergey Ioffe, Christian Szegedy
  Arguments:
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Conv2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: An optional projection function to be applied to the `beta`
        weight after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    gamma_constraint: An optional projection function to be applied to the
        `gamma` weight after being updated by an `Optimizer`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation if possible.
      If `None`, use the system recommended implementation.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: A string, the name of the layer.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=init_ops.zeros_initializer(),
               gamma_initializer=init_ops.ones_initializer(),
               moving_mean_initializer=init_ops.zeros_initializer(),
               moving_variance_initializer=init_ops.ones_initializer(),
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               name=None,
               **kwargs):
    super(BatchNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer
    self.moving_mean_initializer = moving_mean_initializer
    self.moving_variance_initializer = moving_variance_initializer
    self.beta_regularizer = beta_regularizer
    self.gamma_regularizer = gamma_regularizer
    self.beta_constraint = beta_constraint
    self.gamma_constraint = gamma_constraint
    self.renorm = renorm
    if fused is None:
      fused = True

    self.fused = fused
    self._bessels_correction_test_only = True
    if renorm:
      renorm_clipping = renorm_clipping or {}
      keys = ['rmax', 'rmin', 'dmax']
      if set(renorm_clipping) - set(keys):
        raise ValueError('renorm_clipping %s contains keys not in %s' %
                         (renorm_clipping, keys))
      self.renorm_clipping = renorm_clipping
      self.renorm_momentum = renorm_momentum

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndim = len(input_shape)
    if self.axis < 0:
      axis = ndim + self.axis
    else:
      axis = self.axis
    if axis < 0 or axis >= ndim:
      raise ValueError('Value of `axis` argument ' + str(self.axis) +
                       ' is out of range for input with rank ' + str(ndim))

    if self.fused:
      # Currently fused batch norm doesn't support renorm and beta/gamma
      # regularizer; and only supports an input tensor of rank 4 and a channel
      # dimension on axis 1 and 3.
      # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
      # output back to its original shape accordingly.
      self.fused = not self.renorm and ndim == 4 and axis in [
          1, 3
      ] and self.beta_regularizer is None and self.gamma_regularizer is None

    if self.fused:
      if axis == 1:
        self._data_format = 'NCHW'
      else:
        self._data_format = 'NHWC'

    param_dim = input_shape[axis]
    if not param_dim.value:
      raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                       input_shape)
    self.input_spec = base.InputSpec(ndim=ndim,
                                     axes={self.axis: param_dim.value})

    if self.scale:
      self.gamma = self.add_variable(name='gamma',
                                     shape=(param_dim,),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True)
    else:
      self.gamma = None
      if self.fused:
        self._gamma_const = array_ops.constant(1.0, shape=(param_dim,))

    if self.center:
      self.beta = self.add_variable(name='beta',
                                    shape=(param_dim,),
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint,
                                    trainable=True)
    else:
      self.beta = None
      if self.fused:
        self._beta_const = array_ops.constant(0.0, shape=(param_dim,))

    # Disable variable partitioning when creating the moving mean and variance
    try:
      if self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None
      self.moving_mean = self.add_variable(
          name='moving_mean',
          shape=(param_dim,),
          initializer=self.moving_mean_initializer,
          trainable=False)
      self.moving_variance = self.add_variable(
          name='moving_variance',
          shape=(param_dim,),
          initializer=self.moving_variance_initializer,
          trainable=False)
      self._one_minus_decay = 1.0 - self.momentum
      if self.renorm:
        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = self.add_variable(name=name,
                                  shape=shape,
                                  initializer=init_ops.zeros_initializer(),
                                  trainable=False)
          return var

        with ops.device(None):
          device = ((lambda _: self.moving_mean.device)
                    if context.in_graph_mode() else self.moving_mean.device)
          with ops.device(device):
            self.renorm_mean = _renorm_variable('renorm_mean', (param_dim,))
            self.renorm_mean_weight = _renorm_variable('renorm_mean_weight', ())
          # We initialize renorm_stddev to 0, and maintain the (0-initialized)
          # renorm_stddev_weight. This allows us to (1) mix the average
          # stddev with the minibatch stddev early in training, and (2) compute
          # the unbiased average stddev by dividing renorm_stddev by the weight.
          device = ((lambda _: self.moving_variance.device)
                    if context.in_graph_mode() else self.moving_variance.device)
          with ops.device(device):
            self.renorm_stddev = _renorm_variable('renorm_stddev', (param_dim,))
            self.renorm_stddev_weight = _renorm_variable(
                'renorm_stddev_weight', ())
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_moving_average(self, variable, value, one_minus_decay):
    with ops.name_scope(None, 'AssignMovingAvg',
                        [variable, value, one_minus_decay]) as scope:
      with ops.colocate_with(variable):
        update_delta = math_ops.multiply(
            math_ops.subtract(variable.read_value(), value),
            one_minus_decay)
        if isinstance(variable, resource_variable_ops.ResourceVariable):
          # state_ops.assign_sub does an extra read_variable_op after the
          # assign. We avoid that here.
          return gen_resource_variable_ops.assign_sub_variable_op(
              variable.handle, update_delta, name=scope)
        else:
          return state_ops.assign_sub(variable, update_delta, name=scope)

  def _fused_batch_norm(self, inputs, training):
    """Returns the output of fused batch norm."""
    # TODO(reedwm): Add support for fp16 inputs.
    beta = self.beta if self.center else self._beta_const
    gamma = self.gamma if self.scale else self._gamma_const

    def _fused_batch_norm_training():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          epsilon=self.epsilon,
          data_format=self._data_format)

    def _fused_batch_norm_inference():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=self.moving_mean,
          variance=self.moving_variance,
          epsilon=self.epsilon,
          is_training=False,
          data_format=self._data_format)

    output, mean, variance = utils.smart_cond(
        training, _fused_batch_norm_training, _fused_batch_norm_inference)
    if not self._bessels_correction_test_only:
      # Remove Bessel's correction to be consistent with non-fused batch norm.
      # Note that the variance computed by fused batch norm is
      # with Bessel's correction.
      sample_size = math_ops.cast(
          array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
      factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
      variance *= factor

    training_value = utils.constant_value(training)
    if training_value is None:
      one_minus_decay = _smart_select(training,
                                      lambda: self._one_minus_decay,
                                      lambda: 0.)
    else:
      one_minus_decay = ops.convert_to_tensor(self._one_minus_decay)
    if training_value or training_value is None:
      mean_update = self._assign_moving_average(self.moving_mean, mean,
                                                one_minus_decay)
      variance_update = self._assign_moving_average(self.moving_variance,
                                                    variance, one_minus_decay)
      if context.in_graph_mode():
        # Note that in Eager mode, the updates are already executed when running
        # assign_moving_averages. So we do not need to put them into
        # collections.
        self.add_update(mean_update, inputs=inputs)
        self.add_update(variance_update, inputs=inputs)

    return output

  def _renorm_correction_and_moments(self, mean, variance, training):
    """Returns the correction and update values for renorm."""
    stddev = math_ops.sqrt(variance + self.epsilon)
    # Compute the average mean and standard deviation, as if they were
    # initialized with this batch's moments.
    mixed_renorm_mean = (self.renorm_mean +
                         (1. - self.renorm_mean_weight) * mean)
    mixed_renorm_stddev = (self.renorm_stddev +
                           (1. - self.renorm_stddev_weight) * stddev)
    # Compute the corrections for batch renorm.
    r = stddev / mixed_renorm_stddev
    d = (mean - mixed_renorm_mean) / mixed_renorm_stddev
    # Ensure the corrections use pre-update moving averages.
    with ops.control_dependencies([r, d]):
      mean = array_ops.identity(mean)
      stddev = array_ops.identity(stddev)
    rmin, rmax, dmax = [self.renorm_clipping.get(key)
                        for key in ['rmin', 'rmax', 'dmax']]
    if rmin is not None:
      r = math_ops.maximum(r, rmin)
    if rmax is not None:
      r = math_ops.minimum(r, rmax)
    if dmax is not None:
      d = math_ops.maximum(d, -dmax)
      d = math_ops.minimum(d, dmax)
    # When not training, use r=1, d=0, and decay=1 meaning no updates.
    r = _smart_select(training, lambda: r, lambda: array_ops.ones_like(r))
    d = _smart_select(training, lambda: d, lambda: array_ops.zeros_like(d))
    decay = _smart_select(training, lambda: self.renorm_momentum, lambda: 1.)

    def _update_renorm_variable(var, weight, value):
      """Updates a moving average and weight, returns the unbiased value."""
      # Update the variables without zero debiasing. The debiasing will be
      # accomplished by dividing the exponential moving average by the weight.
      # For example, after a single update, the moving average would be
      # (1-decay) * value. and the weight will be 1-decay, with their ratio
      # giving value.
      # Make sure the weight is not updated until before r and d computation.
      value = array_ops.identity(value)
      with ops.control_dependencies([value]):
        weight_value = array_ops.constant(1., dtype=weight.dtype)
      new_var = moving_averages.assign_moving_average(
          var, value, decay, zero_debias=False)
      new_weight = moving_averages.assign_moving_average(
          weight, weight_value, decay, zero_debias=False)
      return new_var / new_weight

    with ops.colocate_with(self.moving_mean):
      new_mean = _update_renorm_variable(self.renorm_mean,
                                         self.renorm_mean_weight,
                                         mean)
    with ops.colocate_with(self.moving_variance):
      new_stddev = _update_renorm_variable(self.renorm_stddev,
                                           self.renorm_stddev_weight,
                                           stddev)
      # Make sqrt(moving_variance + epsilon) = new_stddev.
      new_variance = math_ops.square(new_stddev) - self.epsilon

    return (r, d, new_mean, new_variance)

  def call(self, inputs, batch_mean, batch_variance, training=False):
    if self.fused:
      return self._fused_batch_norm(inputs, training=training)

    # First, compute the axes along which to reduce the mean / variance,
    # as well as the broadcast shape to be used for all parameters.
    input_shape = inputs.get_shape()
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[self.axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[self.axis] = input_shape[self.axis].value

    # Determines whether broadcasting is needed.
    needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

    scale, offset = self.gamma, self.beta

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = utils.constant_value(training)
    if training_value is not False:
      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      if batch_mean is None and batch_variance is None:
          batch_mean, batch_variance = nn.moments(inputs, reduction_axes)
      mean = _smart_select(training,
                           lambda: batch_mean,
                           lambda: self.moving_mean)
      variance = _smart_select(training,
                               lambda: batch_variance,
                               lambda: self.moving_variance)

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            mean, variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        scale = array_ops.stop_gradient(r, name='renorm_r')
        offset = array_ops.stop_gradient(d, name='renorm_d')
        if self.gamma is not None:
          scale *= self.gamma
          offset *= self.gamma
        if self.beta is not None:
          offset += self.beta
      else:
        new_mean, new_variance = mean, variance

      # Update moving averages when training, and prevent updates otherwise.
      decay = _smart_select(training, lambda: self.momentum, lambda: 1.)
      mean_update = moving_averages.assign_moving_average(
          self.moving_mean, new_mean, decay, zero_debias=False)
      variance_update = moving_averages.assign_moving_average(
          self.moving_variance, new_variance, decay, zero_debias=False)
      if context.in_graph_mode():
        self.add_update(mean_update, inputs=inputs)
        self.add_update(variance_update, inputs=inputs)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    def _broadcast(v):
      if needs_broadcasting and v is not None:
        # In this case we must explicitly broadcast all parameters.
        return array_ops.reshape(v, broadcast_shape)
      return v

    return nn.batch_normalization(inputs,
                                  _broadcast(mean),
                                  _broadcast(variance),
                                  _broadcast(offset),
                                  _broadcast(scale),
                                  self.epsilon), batch_mean, batch_variance


def batch_normalization(inputs, batch_mean, batch_variance,
                        axis=-1,
                        momentum=0.99,
                        epsilon=1e-3,
                        center=True,
                        scale=True,
                        beta_initializer=init_ops.zeros_initializer(),
                        gamma_initializer=init_ops.ones_initializer(),
                        moving_mean_initializer=init_ops.zeros_initializer(),
                        moving_variance_initializer=init_ops.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        beta_constraint=None,
                        gamma_constraint=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99,
                        fused=None):
  """Functional interface for the batch normalization layer.
  Reference: http://arxiv.org/abs/1502.03167
  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"
  Sergey Ioffe, Christian Szegedy
  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:
  ```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```
  Arguments:
    inputs: Tensor input.
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Convolution2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: An optional projection function to be applied to the `beta`
        weight after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    gamma_constraint: An optional projection function to be applied to the
        `gamma` weight after being updated by an `Optimizer`.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (normalized with statistics of the current batch) or in inference mode
      (normalized with moving statistics). **NOTE**: make sure to set this
      parameter correctly, or else your training/inference will not work
      properly.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation if possible.
      If `None`, use the system recommended implementation.
  Returns:
    Output tensor.
  """
  layer = BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      name=name,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs, batch_mean, batch_variance, training=training)


def _smart_select(pred, fn_then, fn_else):
  """Selects fn_then() or fn_else() based on the value of pred.
  The purpose of this function is the same as `utils.smart_cond`. However, at
  the moment there is a bug (b/36297356) that seems to kick in only when
  `smart_cond` delegates to `tf.cond`, which sometimes results in the training
  hanging when using parameter servers. This function will output the result
  of `fn_then` or `fn_else` if `pred` is known at graph construction time.
  Otherwise, it will use `tf.where` which will result in some redundant work
  (both branches will be computed but only one selected). However, the tensors
  involved will usually be small (means and variances in batchnorm), so the
  cost will be small and will not be incurred at all if `pred` is a constant.
  Args:
    pred: A boolean scalar `Tensor`.
    fn_then: A callable to use when pred==True.
    fn_else: A callable to use when pred==False.
  Returns:
    A `Tensor` whose value is fn_then() or fn_else() based on the value of pred.
  """
  pred_value = utils.constant_value(pred)
  if pred_value:
    return fn_then()
  elif pred_value is False:
    return fn_else()
  t_then = array_ops.expand_dims(fn_then(), 0)
  t_else = array_ops.expand_dims(fn_else(), 0)
  pred = array_ops.reshape(pred, [1])
  result = array_ops.where(pred, t_then, t_else)
  return array_ops.squeeze(result, [0])
