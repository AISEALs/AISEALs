import tensorflow.compat.v1 as tf
# fc layer
#param: Layer:input_layer, List:layersizes
def build_deep_layers(input_layer, layersizes, name):
  net = input_layer
  for num in range(0, len(layersizes)):
    net = tf.layers.dense(net,
          units = layersizes[num],
          activation = tf.nn.selu,
          kernel_initializer=tf.glorot_uniform_initializer(),
          name = "%s_fc_%d"%(name, num))
  return net

# mmoe model definition
def mmoe_model_fn(input_layer, layersizes, num_experts, num_tasks, name, debug=False):
  experts = []
  for num in range(0, num_experts):
    expert = build_deep_layers(input_layer, layersizes, "%s_expert_%d"%(name, num))
    experts.append(expert)
    if debug:
      print("%s_expert_%d"%(name, num))
      print(expert.get_shape())
      print("-----")
  ## experts outputs
  expert_concat_layer = tf.stack(experts, axis=2, name="%s_expert_concat_layer"%name)

  ## tower output
  towers = []
  for num in range(0, num_tasks):
    weight = tf.layers.dense(input_layer,
             units=num_experts,
             activation=tf.nn.relu,
             kernel_initializer=tf.glorot_uniform_initializer(),
             name="%s_weight%d"%(name, num))
    gate = tf.nn.softmax(weight, name="%s_gate%d"%(name, num))
    if debug:
      tf.logging.info("gate shape:"%gate.shape)
    # task towers
    tower = tf.multiply(expert_concat_layer, tf.expand_dims(gate, axis=1))
    if debug:
      tf.logging.info(tower.shape)
    tower = tf.reduce_sum(tower, axis=2)
    if debug:
      tf.logging.info("tower shape:"%tower.shape)
    ## reshape to output size
    tower = tf.reshape(tower, [-1, layersizes[-1]])
    if debug:
      tf.logging.info("tower reshape:"%tower.shape)
    towers.append(tower)
  return towers


# mmoe model definition
def mmoe_model_fn_rerank(input_layer, layersizes, num_experts, num_tasks, name, debug=False):
  experts = []
  for num in range(0, num_experts):
    expert = build_deep_layers(input_layer, layersizes, "%s_expert_%d"%(name, num))
    experts.append(expert)
    if debug:
      print("%s_expert_%d"%(name, num))
      print(expert.get_shape())
      print("-----")
  ## experts outputs
  expert_concat_layer = tf.stack(experts, axis=3, name="%s_expert_concat_layer"%name)

  ## tower output
  towers = []
  for num in range(0, num_tasks):
    weight = tf.layers.dense(input_layer,
             units=num_experts,
             activation=tf.nn.relu,
             kernel_initializer=tf.glorot_uniform_initializer(),
             name="%s_weight%d"%(name, num))
    gate = tf.nn.softmax(weight, name="%s_gate%d"%(name, num))
    if debug:
      print("-----")
      print("%s_task_%d"%(name, num))
      print("gate shape:")
      print(gate.get_shape())
    # task towers
    tower = tf.multiply(expert_concat_layer, tf.expand_dims(gate, axis=2))
    if debug:
      tf.logging.info(tower.shape)
      print("tower shape:")
      print(tower.get_shape())
    tower = tf.reduce_sum(tower, axis=3)
    if debug:
      tf.logging.info("tower shape:"%tower.shape)
      print("tower reduce_sum shape:")
      print(tower.get_shape())
    ## reshape to output size
    tower = tf.reshape(tower, [-1, 5, layersizes[-1]])
    if debug:
      tf.logging.info("tower reshape:"%tower.shape)
      print("tower reshape shape:")
      print(tower.get_shape())
    towers.append(tower)
  return towers

## definition of focal loss
def binary_focal_loss(logits, y_true, gamma=2, alpha=0.25):
  epsilon = 1.e-8
  probs = tf.nn.sigmoid(logits)
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.clip_by_value(probs, epsilon, 1.-epsilon)
  p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
  weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
  alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1-alpha)
  focal_loss = -alpha_t * weight * tf.log(p_t)
  return tf.reduce_mean(focal_loss)
