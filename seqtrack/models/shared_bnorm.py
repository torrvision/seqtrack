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


def _fused_batch_norm(
    inputs,
    decay=0.999,
    center=True,
    scale=False,
    epsilon=0.001,
    activation_fn=None,
    param_initializers=None,
    updates_collections=ops.GraphKeys.UPDATE_OPS,
    is_training=True,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    data_format=DATA_FORMAT_NHWC,
    zero_debias_moving_mean=False,
    scope=None):
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
      performance.
    center: If True, add offset of `beta` to normalized tensor.  If False,
      `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
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
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean.
    scope: Optional scope for `variable_scope`.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If the rank of `inputs` is neither 2 or 4.
    ValueError: If rank or `C` dimension of `inputs` is undefined.
  """
  # TODO(reedwm): Add support for fp16 inputs.
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    original_shape = inputs.get_shape()
    original_inputs = inputs
    original_rank = original_shape.ndims
    if original_rank is None:
      raise ValueError('Inputs %s has undefined rank' % inputs.name)
    elif original_rank not in [2, 4]:
      raise ValueError('Inputs %s has unsupported rank.'
                       ' Expected 2 or 4 but got %d' % (
                           inputs.name, original_rank))
    if original_rank == 2:
      channels = inputs.get_shape()[-1].value
      if channels is None:
        raise ValueError('`C` dimension must be known but is None')
      new_shape = [-1, 1, 1, channels]
      if data_format == DATA_FORMAT_NCHW:
        new_shape = [-1, channels, 1, 1]
      inputs = array_ops.reshape(inputs, new_shape)
    inputs_shape = inputs.get_shape()
    dtype = inputs.dtype.base_dtype
    if data_format == DATA_FORMAT_NHWC:
      params_shape = inputs_shape[-1:]
    else:
      params_shape = inputs_shape[1:2]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    trainable_beta = trainable and center
    beta_collections = utils.get_variable_collections(variables_collections,
                                                      'beta')
    if not param_initializers:
      param_initializers = {}
    if center:
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variables.model_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=beta_initializer,
          collections=beta_collections,
          trainable=trainable_beta)
    else:
      beta = array_ops.constant(0.0, shape=params_shape)

    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=gamma_initializer,
          collections=gamma_collections,
          trainable=trainable)
    else:
      gamma = array_ops.constant(1.0, shape=params_shape)

    # Create moving_mean and moving_variance variables and add them to the
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables (this needs to be handled carefully, as it may break
    # the checkpoint backward compatibility).
    with variable_scope.variable_scope(
        variable_scope.get_variable_scope()) as local_scope:
      local_scope.set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_mean = variables.model_variable(
          'moving_mean',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_mean_initializer,
          trainable=False,
          collections=moving_mean_collections)
      moving_variance_collections = utils.get_variable_collections(
          variables_collections, 'moving_variance')
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      moving_variance = variables.model_variable(
          'moving_variance',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_variance_initializer,
          trainable=False,
          collections=moving_variance_collections)

    def _fused_batch_norm_training():
      return nn.fused_batch_norm(
          inputs, gamma, beta, epsilon=epsilon, data_format=data_format)
    def _fused_batch_norm_inference():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=epsilon,
          is_training=False,
          data_format=data_format)
    outputs, mean, variance = utils.smart_cond(is_training,
                                               _fused_batch_norm_training,
                                               _fused_batch_norm_inference)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `need_updates` will be true.
    is_training_value = utils.constant_value(is_training)
    need_updates = is_training_value is None or is_training_value
    if need_updates:
      if updates_collections is None:
        no_updates = lambda: outputs
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies(
              [update_moving_mean, update_moving_variance]):
            return array_ops.identity(outputs)
        outputs = utils.smart_cond(is_training, _force_updates, no_updates)
      else:
        moving_vars_fn = lambda: (moving_mean, moving_variance)
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          return update_moving_mean, update_moving_variance
        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)

    outputs.set_shape(inputs_shape)
    if original_shape.ndims == 2:
      outputs = array_ops.reshape(outputs, array_ops.shape(original_inputs))
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def batch_norm(inputs,
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
               fused=None,
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
    return _fused_batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=activation_fn,
        param_initializers=param_initializers,
        updates_collections=updates_collections,
        is_training=is_training,
        reuse=reuse,
        variables_collections=variables_collections,
        outputs_collections=outputs_collections,
        trainable=trainable,
        data_format=data_format,
        zero_debias_moving_mean=zero_debias_moving_mean,
        scope=scope)

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
      layer = normalization_layers.BatchNormalization(
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
      outputs = layer.apply(inputs, training=is_training)

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
      return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

    # Not supported by layer class: batch_weights argument,
    # and custom updates_collections. In that case, use the legacy BN
    # implementation.
    # Custom updates collections are not supported because the update logic
    # is different in this case, in particular w.r.t. "forced updates" and
    # update op reuse.
    if renorm:
      raise ValueError('renorm is not supported with batch_weights, '
                       'updates_collections or zero_debias_moving_mean')
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if batch_weights is not None:
      batch_weights = ops.convert_to_tensor(batch_weights)
      inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
      # Reshape batch weight values so they broadcast across inputs.
      nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
      batch_weights = array_ops.reshape(batch_weights, nshape)

    if data_format == DATA_FORMAT_NCHW:
      moments_axes = [0] + list(range(2, inputs_rank))
      params_shape = inputs_shape[1:2]
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      moments_axes = list(range(inputs_rank - 1))
      params_shape = inputs_shape[-1:]
      params_shape_broadcast = None
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if not param_initializers:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)

    # Create moving_mean and moving_variance variables and add them to the
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables (this needs to be handled carefully, as it may break
    # the checkpoint backward compatibility).
    with variable_scope.variable_scope(
        variable_scope.get_variable_scope()) as local_scope:
      local_scope.set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_mean = variables.model_variable(
          'moving_mean',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_mean_initializer,
          trainable=False,
          collections=moving_mean_collections)
      moving_variance_collections = utils.get_variable_collections(
          variables_collections, 'moving_variance')
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      moving_variance = variables.model_variable(
          'moving_variance',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_variance_initializer,
          trainable=False,
          collections=moving_variance_collections)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      if batch_weights is None:
        if data_format == DATA_FORMAT_NCHW:
          mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.moments(inputs, moments_axes)
      else:
        if data_format == DATA_FORMAT_NCHW:
          mean, variance = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights)

      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    if data_format == DATA_FORMAT_NCHW:
      mean = array_ops.reshape(mean, params_shape_broadcast)
      variance = array_ops.reshape(variance, params_shape_broadcast)
      if beta is not None:
        beta = array_ops.reshape(beta, params_shape_broadcast)
      if gamma is not None:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Compute batch_normalization.
    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                     epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
