# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AdaBelief for TensorFlow.
Modified from tensorflow/tensorflow/python/keras/optimizer_v2/adam.py
https://github.com/andreselizondo-adestech/Adabelief-Optimizer/blob/5d9bc02ae1d125d78380a604def94b298c023f9a/pypi_packages/adabelief_tf/adabelief_tf/AdaBelief_tf.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.AdaBeliefOptimizer')
class AdaBeliefOptimizer(optimizer_v2.OptimizerV2):
    """Optimizer that implements the AdaBelief algorithm.
    References:
    AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients:
      ([pdf](https://arxiv.org/pdf/2010.07468.pdf))
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name="AdaBelief",
               **kwargs):
        """Construct a new Adam optimizer.
        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
          beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability.
          amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond". Defaults to `False`.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
          **kwargs: Keyword arguments. Allowed to be one of
            `"clipnorm"` or `"clipvalue"`.
            `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
            gradients by value.
        """
        super(AdaBeliefOptimizer, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad
        
    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 's')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'shat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaBeliefOptimizer, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
             (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdaBeliefOptimizer, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        s = self.get_slot(var, 's')

        if not self.amsgrad:
            return  training_ops.resource_apply_adam(
                    var.handle,
                    m.handle,
                    s.handle,
                    coefficients['beta_1_power'],
                    coefficients['beta_2_power'],
                    coefficients['lr_t'],
                    coefficients['beta_1_t'],
                    coefficients['beta_2_t'],
                    coefficients['epsilon'],
                    grad,
                    use_locking=self._use_locking)
        else:
            shat = self.get_slot(var, 'shat')
            return  training_ops.resource_apply_adam_with_amsgrad(
                    var.handle,
                    m.handle,
                    s.handle,
                    shat.handle,
                    coefficients['beta_1_power'],
                    coefficients['beta_2_power'],
                    coefficients['lr_t'],
                    coefficients['beta_1_t'],
                    coefficients['beta_2_t'],
                    coefficients['epsilon'],
                    grad,
                    use_locking=self._use_locking)


    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                                use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # s_t = beta2 * s + (1 - beta2) * (g_t - m_t) * (g_t - m_t)
        s = self.get_slot(var, 's')
        s_scaled_g_values = (grad - m_t) * (grad - m_t) * coefficients['one_minus_beta_2_t']
        s_t = state_ops.assign(s, s * coefficients['beta_2_t'],
                                use_locking=self._use_locking)
        with ops.control_dependencies([s_t]):
            s_t = self._resource_scatter_add(s, indices, s_scaled_g_values)

        if not self.amsgrad:
            s_sqrt = math_ops.sqrt(s_t)
            var_update = state_ops.assign_sub(
                var, coefficients['lr'] * m_t / (s_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, s_t])
        else:
            s_hat = self.get_slot(var, 'shat')
            s_hat_t = math_ops.maximum(s_hat, s_t)
            with ops.control_dependencies([s_hat_t]):
                s_hat_t = state_ops.assign(s_hat, s_hat_t, use_locking=self._use_locking)
            s_hat_sqrt = math_ops.sqrt(s_hat_t)
            var_update = state_ops.assign_sub(
                var,
                coefficients['lr'] * m_t / (s_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, s_t, s_hat_t])

    def get_config(self):
        config = super(AdaBeliefOptimizer, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config