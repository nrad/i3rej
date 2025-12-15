import tfxkit.common.tf_utils as tf_utils
import tensorflow as tf
import logging
import keras
import numpy as np
import re

logger = logging.getLogger(__name__)

DEFAULT_MUON_FEATURES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "dir_x",
    "dir_y",
    "dir_z",
    "radius",
    "log_energy",
]

DEFAULT_ELOSS_FEATURES = [ 
 'type',
 'pos_x',
 'pos_y',
 'pos_z',
 'pos_magnitude',
 'distance',
 'log_energy'
 ]


DEFAULT_N_MUONS = 10

# DEFAULT_MUON_FEATURES = ['log_energy', 'pos_x', 'pos_y', 'pos_z', 'distance']


def df_to_muon_array(
    df, n_muons=10, epsilon_std=1e-5, muon_features=DEFAULT_MUON_FEATURES, muon_prefix="mu"
):
    """
    Converts columns like mu1_pos_x, mu1_pos_y, mu1_pos_z, mu1_dir_x, mu1_dir_y, mu1_dir_z, mu1_energy
    (and similarly for mu2..mu10) into a NumPy array of shape (num_events, n_muons, 7).

    Each "slice" along axis 1 corresponds to one muon, with the 7 features:
      [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy].

    If muon feature values are zero, replaces them with small random numbers sampled
    from a normal distribution around zero with standard deviation `epsilon_std`.
    """
    arr = np.zeros((len(df), n_muons, len(muon_features)), dtype=float)

    for i in range(1, n_muons + 1):
        col_list = [f"{muon_prefix}{i}_{feat}" for feat in muon_features]
        print(f"muon {i}: {col_list}")
        muon_data = df[col_list].values
        if epsilon_std is not None:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.random.normal(
                loc=0, scale=epsilon_std, size=zero_mask.sum()
            )
        else:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.nan

        arr[:, i - 1, :] = muon_data
    return arr


def df_to_muon_plus_event_array(
    df,
    n_muons=10,
    epsilon_std=1e-5,
    muon_features=DEFAULT_MUON_FEATURES,
    event_features=[],
):
    """
    similar to df_to_muon_array but also includes the event information per muon
    """
    # event_features = [k for k in df.columns if not re.findall(r"mu\d+_*", k)]
    # print(event_features)
    arr = np.zeros(
        (len(df), n_muons, len(muon_features) + len(event_features)), dtype=float
    )

    for i in range(1, n_muons + 1):
        col_list = [f"mu{i}_{feat}" for feat in muon_features] + event_features
        print(f"muon {i}: {col_list}")
        muon_data = df[col_list].values
        if epsilon_std is not None:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.random.normal(
                loc=0, scale=epsilon_std, size=zero_mask.sum()
            )
        else:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.nan

        arr[:, i - 1, :] = muon_data
    return arr


def build_muon_branch(
    embedding_dim=16,
    layers_list=[1028, 512, 256, 128, 64],
    hidden_activation="relu",
    dropout=0.2,
    kernel_regularizer=1e-4,
    final_activation="sigmoid",
    n_muon_features=len(DEFAULT_MUON_FEATURES),
    n_muons=DEFAULT_N_MUONS,
    aggregation_method="simple",
    batch_size=None,
    name="MuonBranch",
    **kwargs,
):
    """
    Returns a Sequential sub-model that:
      - Uses TimeDistributed to apply a small MLP to each muon in the bundle.
      - Aggregates embeddings (e.g., by mean, max, etc).
    """


    # muon_input = keras.Input(shape=(None, n_muon_features), name="muon_input")
    muon_input = keras.Input(shape=(n_muons, n_muon_features), name=f"{name}_input")
    muon_mlp = tf_utils.define_mlp(
        n_muon_features,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=embedding_dim,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        batch_size=batch_size,
        build=True,
        name=name,
        **kwargs,
    )
    muon_mlp.summary()

    ##
    ## Apply the same MLP to each muon in the bundle,
    ## i.e. muon_multiplicity is considered as the temporal dimension
    ##
    print(muon_mlp)
    print(type(muon_mlp))
    muon_embeddings = keras.layers.TimeDistributed(muon_mlp)(muon_input)

    # aggregation =
    if not aggregation_method in AGGREGATION_METHODS:
        raise ValueError(
            f"Invalid aggregation method: {aggregation_method}. Must be one of {list(AGGREGATION_METHODS.keys())}"
        )
    agg_layer = AGGREGATION_METHODS[aggregation_method]()
    # if hasattr(agg_layer, "build"):
    #     print(f"building agg_layer with input_shape {muon_embeddings.shape}")
    #     #agg_layer.build(input_shape=muon_embeddings.shape)
    #     agg_layer.build(input_shape=(batch_size, DEFAULT_N_MUONS, n_muon_features) )

    muon_event_embedding = agg_layer(muon_embeddings)

    return keras.Model(
        inputs=muon_input, outputs=muon_event_embedding, name="MuonBranch"
    )


# tf.keras.regularizers.l2(kernel_regularizer)
# mf.features = event_features


def build_event_branch(
    event_feat_dim=10,
    layers_list=[1028, 512, 256, 128, 64],
    hidden_activation="relu",
    n_final_units=64,
    final_activation="sigmoid",
    dropout=0.2,
    kernel_regularizer=1e-4,
    **kwargs,
):
    """
    Returns a Sequential sub-model for the event-level features.
    Example: 2 dense layers, can be adapted as needed.
    """

    event_input = keras.Input(shape=(event_feat_dim,), name="event_input")
    event_mlp = tf_utils.define_mlp(
        features=event_feat_dim,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=n_final_units,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        build=True,
        name="EventMLP",
        **kwargs,
    )
    event_mlp(event_input)
    return event_mlp


def build_combined_model(
    event_feat_dim=10,
    muon_embedding_dim=16,
    muon_feat_dim=len(DEFAULT_MUON_FEATURES),
    layers_list=[1028],
    kernel_regularizer=1e-4,
    dropout=0.2,
    event_branch_kwargs={},
    muon_branch_kwargs={},
    combined_activation="relu",
    batch_size=None,
):

    print(f"{event_branch_kwargs = }")
    event_branch = build_event_branch(
        event_feat_dim=event_feat_dim, **event_branch_kwargs
    )
    print(f"{muon_branch_kwargs = }")
    layers_list = tf_utils.parse_layers_list(layers_list)
    layers_list = layers_list if isinstance(layers_list, list) else [layers_list]
    layers_list = [int(layer) for layer in layers_list]
    dropout_list = tf_utils.broadcast_argument_to_layers(dropout, n_layers=len(layers_list))
    muon_branch = build_muon_branch(
        embedding_dim=int(muon_embedding_dim),
        n_muon_features=muon_feat_dim,
        **muon_branch_kwargs,
    )

    combined = keras.layers.Concatenate(name="merge_branches")(
        [event_branch.output, muon_branch.output]
    )

    x = keras.layers.BatchNormalization()(combined)
    for i, n_units in enumerate(layers_list):
        if i != 0:
            x = keras.layers.Dropout(dropout_list[i - 1])(x)
        x = keras.layers.Dense(
            n_units,
            activation=combined_activation,
            kernel_regularizer=tf_utils.parse_regularizer(kernel_regularizer),
        )(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="classification")(x)
    model = keras.Model(
        inputs=[event_branch.input, muon_branch.input],
        outputs=output,
        name="CombinedModel",
    )

    return event_branch, muon_branch, model

def build_combined_showermu_model(
    event_feat_dim=10,
    muon_embedding_dim=16,
    shower_muon_embedding_dim=16,
    muon_feat_dim=len(DEFAULT_MUON_FEATURES),
    shower_muon_feat_dim=len(DEFAULT_MUON_FEATURES),
    layers_list=[1028],
    kernel_regularizer=1e-4,
    dropout=0.2,
    event_branch_kwargs={},
    muon_branch_kwargs={},
    shower_muon_branch_kwargs={},
    combined_activation="relu",
    batch_size=None,
):

    print(f"{event_branch_kwargs = }")
    event_branch = build_event_branch(
        event_feat_dim=event_feat_dim, **event_branch_kwargs
    )
    print(f"{muon_branch_kwargs = }")
    layers_list = tf_utils.parse_layers_list(layers_list)
    layers_list = layers_list if isinstance(layers_list, list) else [layers_list]
    layers_list = [int(layer) for layer in layers_list]
    dropout_list = tf_utils.broadcast_argument_to_layers(dropout, n_layers=len(layers_list))
    muon_branch = build_muon_branch(
        embedding_dim=int(muon_embedding_dim),
        n_muon_features=muon_feat_dim,
        name="InIceMuonBranch",
        **muon_branch_kwargs,
    )

    shower_muon_branch = build_muon_branch(
        embedding_dim=int(shower_muon_embedding_dim),
        n_muon_features=shower_muon_feat_dim,
        name="ShowerMuonBranch",
        **shower_muon_branch_kwargs,
    )

    combined = keras.layers.Concatenate(name="merge_branches")(
        [event_branch.output, muon_branch.output, shower_muon_branch.output]
    )

    x = keras.layers.BatchNormalization()(combined)
    for i, n_units in enumerate(layers_list):
        if i != 0:
            x = keras.layers.Dropout(dropout_list[i - 1])(x)
        x = keras.layers.Dense(
            n_units,
            activation=combined_activation,
            kernel_regularizer=tf_utils.parse_regularizer(kernel_regularizer),
        )(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="classification")(x)
    model = keras.Model(
        inputs=[event_branch.input, muon_branch.input, shower_muon_branch.input],
        outputs=output,
        name="CombinedModel",
    )
    model.sub_networks = {
        "event_branch": event_branch,
        "muon_branch": muon_branch,
        "shower_muon_branch": shower_muon_branch,
    }

    return event_branch, muon_branch, shower_muon_branch, model


# event_features = [k for k in mf.features if not (re.findall(r"mu\d+_pos_[xyz]", k) or re.findall(r"mu\d+_dir_[xyz]", k) or re.findall("mu\d+_energy", k) or re.findall("mu\d+_log_energy", k)) ]

# n_event_features = len(event_features)


# def xy_maker(df, features, labels, weight=None):
#     """
#     Simple xy_maker function that extracts features and labels from a DataFrame."""
#     X = df[features]
#     y = df[labels]
#     sample_weight = get_weight_column(df, weight)

#     return X, y, sample_weight


def xy_maker_wrapper(muon_prefix="mu", muon_features=DEFAULT_MUON_FEATURES):
    def xy_maker(
        df, features, labels, weight=None
    ):
        muon_pattern = f"{muon_prefix}"+r"\d+_*"
        event_features = [k for k in features if not re.findall(muon_pattern, k)]
        event_feats = df[event_features]
        muon_feats = df_to_muon_array(df, epsilon_std=0, muon_prefix=muon_prefix, muon_features=muon_features)

        sample_weight = tf_utils.get_weight_column(df, weight)
        X = (event_feats, muon_feats)
        y = df[labels]
        return X, y, sample_weight

    return xy_maker

xy_maker_muon_embedding = xy_maker_wrapper(muon_prefix="mu")
xy_maker_shower_muon_embedding = xy_maker_wrapper(muon_prefix="shower_mu")
xy_maker_eloss_embedding = xy_maker_wrapper(muon_prefix="eloss", muon_features=DEFAULT_ELOSS_FEATURES)

def xy_maker_showermu_icemu_embedding(
    df, features, labels, weight=None
):
    muon_pattern = f"mu"+r"\d+_*"
    shower_muon_pattern = f"shower_mu"+r"\d+_*"
    event_features = [k for k in features if not re.findall(muon_pattern, k) and not re.findall(shower_muon_pattern, k)]
    event_feats = df[event_features]
    inice_muon_feats = df_to_muon_array(df, epsilon_std=0, muon_prefix="mu")
    shower_muon_feats = df_to_muon_array(df, epsilon_std=0, muon_prefix="shower_mu")

    sample_weight = tf_utils.get_weight_column(df, weight)
    X = (event_feats, inice_muon_feats, shower_muon_feats)
    y = df[labels]
    return X, y, sample_weight



def define_muemb_model(
    # n_features,
    features,
    n_labels=1,
    event_branch_layers=[64],
    muon_branch_layers=[64],
    combination_layers=[64],
    muon_embedding_dim=16,
    hidden_activation="relu",
    dropout=0.3,
    dropout_muon=None,
    dropout_event=None,
    kernel_regularizer=1e-4,
    aggregation_method="simple",
    batch_size=None,
    muon_prefix="mu",
):
    muon_pattern = f"{muon_prefix}"+r"\d+_*"
    # print(f"{muon_pattern = }")
    # print(f"{features = }")
    # print([ (k, re.findall(muon_pattern, k)) for k in features])
    event_features = [k for k in features if not re.findall(muon_pattern, k)]
    n_event_features = len(event_features)
    print(f"{n_event_features=}, \n{event_features=} \n{features=}")
  
    muon_features = [k for k in features if re.findall(muon_pattern, k)]
    print(f"{n_event_features=}, \n{muon_features=} \n{features=}")
    event_branch_kwargs = dict(
        layers_list=event_branch_layers,
        dropout=dropout_event if dropout_event is not None else dropout,
        kernel_regularizer=kernel_regularizer,
    )
    muon_branch_kwargs = dict(
        layers_list=muon_branch_layers,
        dropout=dropout_muon if dropout_muon is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        aggregation_method=aggregation_method,
        batch_size=batch_size,
    )

    print(muon_branch_kwargs)
    event_branch, muon_branch, model = build_combined_model(
        event_feat_dim=n_event_features,
        muon_embedding_dim=muon_embedding_dim,
        layers_list=combination_layers,
        event_branch_kwargs=event_branch_kwargs,
        muon_branch_kwargs=muon_branch_kwargs,
        dropout=dropout,
        batch_size=batch_size,
    )
    model.event_branch = event_branch
    model.muon_branch = muon_branch

    return model


def define_muemb_showermuemb_model(
    #n_features,
    features,
    n_labels=1,
    event_branch_layers=[64],
    muon_branch_layers=[64],
    shower_muon_branch_layers=[64],
    combination_layers=[64],
    muon_embedding_dim=16,
    shower_muon_embedding_dim=16,
    hidden_activation="relu",
    dropout=0.3,
    dropout_muon=None,
    dropout_event=None,
    kernel_regularizer=1e-4,
    aggregation_method="simple",
    batch_size=None,
):
    muon_pattern = f"mu"+r"\d+_*"
    shower_muon_pattern = f"shower_mu"+r"\d+_*"
    event_features = [k for k in features if not re.findall(muon_pattern, k)]
    n_event_features = len(event_features)
    # muon_features = [k for k in features if re.findall(muon_pattern, k)]
    # shower_muon_features = [k for k in features if re.findall(shower_muon_pattern, k)]

    # print(f"{n_event_features=}, \n{muon_features=} \n{features=}")

    event_branch_kwargs = dict(
        layers_list=event_branch_layers,
        dropout=dropout_event if dropout_event is not None else dropout,
        kernel_regularizer=kernel_regularizer,
    )

    muon_branch_kwargs = dict(
        layers_list=muon_branch_layers,
        dropout=dropout_muon if dropout_muon is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        aggregation_method=aggregation_method,
        batch_size=batch_size,
    )

    shower_muon_branch_kwargs = dict(
        layers_list=shower_muon_branch_layers,
        dropout=dropout_muon if dropout_muon is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        aggregation_method=aggregation_method,
        batch_size=batch_size,
    )


    print(muon_branch_kwargs)
    event_branch, muon_branch, shower_muon_branch, model = build_combined_showermu_model(
        event_feat_dim=n_event_features,
        muon_embedding_dim=muon_embedding_dim,
        shower_muon_embedding_dim=shower_muon_embedding_dim,
        layers_list=combination_layers,
        event_branch_kwargs=event_branch_kwargs,
        muon_branch_kwargs=muon_branch_kwargs,
        shower_muon_branch_kwargs=shower_muon_branch_kwargs,
        dropout=dropout,
        batch_size=batch_size,
    )
    model.event_branch = event_branch
    model.muon_branch = muon_branch
    model.shower_muon_branch = shower_muon_branch

    return model

# def define_custom_embedding_model(
#     # n_features,
#     features,
#     n_labels=1,
#     combination_layers=[64],
#     hidden_activation="relu",
#     dropout=0.3,
#     dropout_event=None,
#     kernel_regularizer=1e-4,
#     aggregation_method="simple",
#     batch_size=None,
# ):
#     muon_pattern = f"mu"+r"\d+_*"
#     shower_muon_pattern = f"shower_mu"+r"\d+_*"
#     event_features = [k for k in features if not re.findall(muon_pattern, k)]
#     n_event_features = len(event_features)
#     # muon_features = [k for k in features if re.findall(muon_pattern, k)]
#     # shower_muon_features = [k for k in features if re.findall(shower_muon_pattern, k)]

#     # print(f"{n_event_features=}, \n{muon_features=} \n{features=}")

#     event_branch_kwargs = dict(
#         layers_list=event_branch_layers,
#         dropout=dropout_event if dropout_event is not None else dropout,
#         kernel_regularizer=kernel_regularizer,
#     )

#     muon_branch_kwargs = dict(
#         layers_list=muon_branch_layers,
#         dropout=dropout_muon if dropout_muon is not None else dropout,
#         kernel_regularizer=kernel_regularizer,
#         aggregation_method=aggregation_method,
#         batch_size=batch_size,
#     )

#     shower_muon_branch_kwargs = dict(
#         layers_list=shower_muon_branch_layers,
#         dropout=dropout_muon if dropout_muon is not None else dropout,
#         kernel_regularizer=kernel_regularizer,
#         aggregation_method=aggregation_method,
#         batch_size=batch_size,
#     )


#     print(muon_branch_kwargs)
#     event_branch, muon_branch, shower_muon_branch, model = build_combined_showermu_model(
#         event_feat_dim=n_event_features,
#         muon_embedding_dim=muon_embedding_dim,
#         shower_muon_embedding_dim=shower_muon_embedding_dim,
#         layers_list=combination_layers,
#         event_branch_kwargs=event_branch_kwargs,
#         muon_branch_kwargs=muon_branch_kwargs,
#         shower_muon_branch_kwargs=shower_muon_branch_kwargs,
#         dropout=dropout,
#         batch_size=batch_size,
#     )
#     model.event_branch = event_branch
#     model.muon_branch = muon_branch
#     model.shower_muon_branch = shower_muon_branch

#     return model


class NeuralNetworkAggregation(keras.layers.Layer):
    def __init__(self, output_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # input_shape is expected to be (batch_size, n_muons, embedding_dim)
        static_shape = list(input_shape)
        n_muons = static_shape[1]
        embedding_dim = static_shape[2]
        if n_muons is None or embedding_dim is None:
            raise ValueError("n_muons and embedding_dim must be fully defined.")
        flatten_dim = n_muons * embedding_dim
        # Create Dense layers with the known flattened dimension
        self.dense1 = keras.layers.Dense(
            64, activation="relu", input_shape=(flatten_dim,)
        )
        self.dense2 = keras.layers.Dense(self.output_dim, activation="relu")
        super().build(input_shape)

    def call(self, inputs):
        # Use tf.shape for dynamic batch size, but use static dimensions for flattening.
        batch_size = tf.shape(inputs)[0]
        static_shape = inputs.get_shape().as_list()
        flatten_dim = (
            static_shape[1] * static_shape[2]
        )  # now fully defined, e.g., 10 * 16 = 160
        flattened = tf.reshape(inputs, (batch_size, flatten_dim))
        return self.dense2(self.dense1(flattened))


# class NeuralNetworkAggregation(keras.layers.Layer):
#     def __init__(self, output_dim=16, **kwargs):
#         super().__init__(**kwargs)
#         self.output_dim = output_dim

#     def build(self, input_shape):
#         # Expecting input_shape = (batch_size, n_muons, embedding_dim)
#         _, n_muons, embedding_dim = input_shape
#         if n_muons is None or embedding_dim is None:
#             raise ValueError("Input dimensions n_muons and embedding_dim must be fully defined")
#         flatten_dim = n_muons * embedding_dim
#         self.dense1 = keras.layers.Dense(64, activation="relu", input_dim=flatten_dim)
#         self.dense2 = keras.layers.Dense(self.output_dim, activation="relu")

#     def call(self, inputs):
#         print("====================================================")
#         print("🔍 Received input_shape in:", inputs, tf.shape(inputs))
#         print("🔍 Received input_shape in:", inputs.shape)
#         batch_size = tf.shape(inputs)[0]  # Get dynamic batch size
#         n_muons = tf.shape(inputs)[1]  # Get dynamic number of muons
#         embedding_dim = tf.shape(inputs)[2]  # Should be 16

#         flatten_dim = n_muons * embedding_dim  # Compute dynamically

#         print(f"🛠 call() -> batch: {batch_size}, muons: {n_muons}, embed_dim: {embedding_dim}")


#         flattened = tf.reshape(inputs, (batch_size, flatten_dim))  # Flatten dynamically
#         return self.dense2(self.dense1(flattened))
class WeightedSumAggregation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agg_weights = None  # This will be created in build()

    def build(self, input_shape):
        # input_shape: (batch_size, n_muons, embedding_dim)
        static_shape = list(input_shape)
        n_muons = static_shape[1]
        if n_muons is None:
            raise ValueError("The muon dimension (n_muons) must be fully defined.")
        # Create one learnable weight per muon.
        self.agg_weights = self.add_weight(
            shape=(n_muons, 1),
            initializer="uniform",
            trainable=True,
            name="aggregation_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Normalize the weights along the muon axis.
        weights = tf.nn.softmax(self.agg_weights, axis=0)
        # Multiply each muon's embedding by its weight and sum over muons.
        weighted_sum = tf.reduce_sum(
            inputs * weights, axis=1
        )  # (batch_size, embedding_dim)
        return weighted_sum


class AttentionAggregation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query = None

    def build(self, input_shape):
        # input_shape: (batch_size, n_muons, embedding_dim)
        static_shape = list(input_shape)
        embedding_dim = static_shape[-1]
        if embedding_dim is None:
            raise ValueError("The embedding dimension must be fully defined.")
        self.query = self.add_weight(
            shape=(embedding_dim,),
            initializer="uniform",
            trainable=True,
            name="query_vector",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores as a dot product with the query.
        scores = tf.reduce_sum(inputs * self.query, axis=-1)  # (batch_size, n_muons)
        attention_weights = tf.nn.softmax(
            scores, axis=1
        )  # Normalize scores along the muon axis.
        # Compute weighted sum of muon embeddings.
        weighted_sum = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1), axis=1
        )
        return weighted_sum  # (batch_size, embedding_dim)


class HybridAggregation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create the submodules without fixed parameters.
        self.nn_agg = (
            NeuralNetworkAggregation()
        )  # You may choose a default output_dim if needed.
        self.attention_agg1 = AttentionAggregation()
        self.attention_agg2 = AttentionAggregation()

    def call(self, inputs):
        # Compute fixed aggregations along the muon axis.
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        # Compute a trainable aggregation using NeuralNetworkAggregation.
        nn_agg_out = self.nn_agg(inputs)
        # Concatenate all aggregated outputs.
        return tf.concat([mean_agg, max_agg, min_agg, nn_agg_out], axis=-1)


# class HybridAggregation(keras.layers.Layer):
#     def __init__(self, embedding_dim, **kwargs):
#         super(HybridAggregation, self).__init__(**kwargs)
#         self.trainable_agg = AttentionAggregation(embedding_dim=embedding_dim)
#         self.nn_agg = NeuralNetworkAggregation(output_dim=embedding_dim)
#         self.attention_agg = AttentionAggregation(embedding_dim=embedding_dim)

#     def call(self, inputs):
#         # Fixed aggregations
#         mean_agg = tf.reduce_mean(inputs, axis=1)
#         max_agg = tf.reduce_max(inputs, axis=1)
#         min_agg = tf.reduce_min(inputs, axis=1)
#         # Trainable aggregation
#         # trainable_agg = self.trainable_agg(inputs)
#         nn_agg = self.nn_agg(inputs)
#         # Concatenate all
#         return tf.concat([mean_agg, max_agg, min_agg, nn_agg], axis=-1)


@keras.saving.register_keras_serializable()
class SimpleAggregation(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    def call(self, inputs):
        # Fixed aggregations
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        return tf.concat([mean_agg, max_agg, min_agg], axis=-1)


@keras.saving.register_keras_serializable()
class SumAggregation(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    def call(self, inputs):
        # Fixed aggregations
        sum_agg = tf.reduce_sum(inputs, axis=1)
        # max_agg = tf.reduce_max(inputs, axis=1)
        # min_agg = tf.reduce_min(inputs, axis=1)
        return tf.concat([sum_agg], axis=-1)


@keras.saving.register_keras_serializable()
class SimpleAggregation3(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    def call(self, inputs):
        # Fixed aggregations
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        sum_agg = tf.reduce_sum(inputs, axis=1)
        std_agg = tf.math.reduce_std(inputs, axis=1)

        return tf.concat([mean_agg, max_agg, min_agg, sum_agg, std_agg], axis=-1)


class SimpleAggregation2(keras.layers.Layer):
    def call(self, inputs):
        # mean_agg = tf.reduce_mean(tf.where(tf.math.is_finite(inputs), inputs, 0), axis=1) #valid_counts
        is_valid = tf.math.is_finite(inputs) & (inputs != 0)
        valid_counts = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=1)

        masked_inputs = tf.where(is_valid, inputs, tf.zeros_like(inputs))
        mean_agg = tf.reduce_sum(masked_inputs, axis=1) / valid_counts
        max_agg = tf.reduce_max(tf.where(is_valid, inputs, -tf.float32.max), axis=1)
        min_agg = tf.reduce_min(tf.where(is_valid, inputs, tf.float32.max), axis=1)
        return tf.concat([mean_agg, max_agg, min_agg], axis=-1)

    # def call(self, inputs):
    #     # Create a mask for valid (finite & non-zero) values
    #     is_valid = tf.math.is_finite(inputs) & (inputs != 0)

    #     # Compute valid count per row (for mean normalization)
    #     valid_counts = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=1, keepdims=True)

    #     # Replace NaNs and zeros with neutral values for aggregation
    #     masked_inputs = tf.where(is_valid, inputs, tf.zeros_like(inputs))

    #     # Mean aggregation (avoid division by zero)
    #     mean_agg = tf.reduce_sum(masked_inputs, axis=1) / tf.maximum(valid_counts, 1.0)

    #     # Max and Min ignoring NaNs (set invalid values to extreme numbers)
    #     max_agg = tf.reduce_max(tf.where(is_valid, inputs, -tf.float32.max), axis=1)
    #     min_agg = tf.reduce_min(tf.where(is_valid, inputs, tf.float32.max), axis=1)

    #     return tf.concat([mean_agg, max_agg, min_agg], axis=-1)

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[2] * 3)


AGGREGATION_METHODS = {
    "simple": SimpleAggregation,
    "sum": SumAggregation,
    "simple3": SimpleAggregation3,
    "simple2": SimpleAggregation2,
    "weighted_sum": WeightedSumAggregation,
    "attention": AttentionAggregation,
    "neural_network": NeuralNetworkAggregation,
    "hybrid": HybridAggregation,
}
