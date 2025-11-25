import tfxkit.common.tf_utils as tf_utils
import tensorflow as tf
import logging
import keras
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Union

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

DEFAULT_N_MUONS = 10


@dataclass
class EmbeddedBranchSpec:
    """
    Configuration container for embedded branches (muons, shower muons, etc.).
    Mirrors the arguments accepted by `build_muon_branch` along with metadata
    needed to discover which features belong to the branch.
    """

    name: str
    n_features: int
    n_objects: int 
    # parameters for the embedding branch
    embedding_dim: int = 16
    layers_list: Sequence[int] = field(default_factory=lambda: [64])
    hidden_activation: str = "relu"
    final_activation: str = "sigmoid"
    dropout: float = 0.3
    kernel_regularizer: float = 1e-4
    aggregation_method: str = "simple3"
    batch_size: Optional[int] = None
    build: bool = True
    extra_branch_kwargs: Dict[str, Any] = field(default_factory=dict)


    def to_branch_kwargs(self) -> Dict[str, Any]:
        branch_kwargs: Dict[str, Any] = dict(
            n_objects=self.n_objects,
            n_features=self.n_features,
            embedding_dim=self.embedding_dim,
            layers_list=list(self.layers_list),
            hidden_activation=self.hidden_activation,
            dropout=self.dropout,
            kernel_regularizer=self.kernel_regularizer,
            final_activation=self.final_activation,
            aggregation_method=self.aggregation_method,
            batch_size=self.batch_size,
            name=self.name,
            build=self.build,
        )
        branch_kwargs.update(self.extra_branch_kwargs)
        return branch_kwargs

def build_embedded_branch(
    *,
    n_objects: int,
    n_features: int,
    embedding_dim: int = 16,
    layers_list=(256, 256),
    hidden_activation: str = "relu",
    final_activation: str = "sigmoid",
    dropout: float = 0.2,
    kernel_regularizer: float = 1e-4,
    aggregation_method: str = "simple3",
    name: str = "EmbeddedBranch",
    time_distributed_layer=keras.layers.TimeDistributed,
    batch_size: Optional[int] = None,
    build: bool = True,
    **mlp_kwargs,
) -> keras.Model:
    """
    General-purpose branch builder that applies the same MLP to each object in a
    sequence and aggregates the resulting embeddings.
    """

    branch_input = keras.Input(
        shape=(n_objects, n_features),
        name=f"{name}_input",
    )

    mlp = tf_utils.define_mlp(
        n_features,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=embedding_dim,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        batch_size=batch_size,
        build=build,
        name=name,
        **mlp_kwargs,
    )

    embedded_sequence = time_distributed_layer(mlp)(branch_input)

    if aggregation_method not in AGGREGATION_METHODS:
        raise ValueError(
            f"Invalid aggregation method: {aggregation_method}. "
            f"Supported values: {list(AGGREGATION_METHODS.keys())}"
        )
    aggregation_layer = AGGREGATION_METHODS[aggregation_method]()
    branch_output = aggregation_layer(embedded_sequence)

    return keras.Model(inputs=branch_input, outputs=branch_output, name=name)



def build_multi_embedded_model(
    *,
    event_feat_dim: int,
    event_branch_kwargs: Dict[str, Any],
    embedded_specs,
    combination_layers,
    combined_activation: str = "relu",
    final_activation: str = "sigmoid",
    kernel_regularizer: float = 1e-4,
    dropout: float = 0.2,
    name="EmbeddedModel"
):
    """
    Builds a model that can consume an arbitrary number of embedded subnetworks.
    """

    event_branch = build_event_branch(
        event_feat_dim=event_feat_dim,
        **event_branch_kwargs,
    )

    layers_list = tf_utils.parse_layers_list(combination_layers)
    dropout_list = tf_utils.broadcast_argument_to_layers(dropout, n_layers=len(layers_list))

    embedded_branches = []
    merged_inputs = [event_branch.input]
    merged_outputs = [event_branch.output]

    for spec in embedded_specs:
        branch = build_embedded_branch(**spec.to_branch_kwargs())
        embedded_branches.append(branch)
        merged_inputs.append(branch.input)
        merged_outputs.append(branch.output)

    combined = keras.layers.Concatenate(name="merge_branches")(merged_outputs)
    x = keras.layers.BatchNormalization()(combined)
    for i, n_units in enumerate(layers_list):
        if i != 0:
            x = keras.layers.Dropout(dropout_list[i - 1])(x)
        x = keras.layers.Dense(
            n_units,
            activation=combined_activation,
            kernel_regularizer=tf_utils.parse_regularizer(kernel_regularizer),
        )(x)

    output = keras.layers.Dense(1, activation=final_activation, name=f"{name}_output")(x)
    model = keras.Model(inputs=merged_inputs, outputs=output, name=name)

    return event_branch, embedded_branches, model


def define_embedded_model(
    n_features,
    features,
    *,
    n_labels: int = 1,
    event_feat_dim: Optional[int] = None,
    embedded_specs: Optional[Iterable[Union[EmbeddedBranchSpec, Dict[str, Any]]]] = None,
    event_branch_layers=(
        64,
        64,
    ),
    event_hidden_activation: str = "relu",
    event_dropout: float = 0.2,
    combination_layers=(
        64,
        64,
    ),
    combined_activation: str = "relu",
    final_activation: str = "sigmoid",
    kernel_regularizer: float = 1e-4,
    dropout: float = 0.2,
    name="EmbeddedModel",
):
    """
    Public wrapper that mirrors `define_muemb_model` but works with arbitrary embedded specs.
    """

    if embedded_specs is None:
        raise ValueError("`embedded_specs` must be provided.")

    normalized_specs = []
    for spec in embedded_specs:
        if isinstance(spec, EmbeddedBranchSpec):
            normalized_specs.append(spec)
        elif isinstance(spec, dict):
            normalized_specs.append(EmbeddedBranchSpec(**spec))
        else:
            raise TypeError(
                f"Unsupported embedded spec type: {type(spec)}. "
                "Use EmbeddedBranchSpec or a dict of its constructor args."
            )

    if not normalized_specs:
        raise ValueError("`embedded_specs` cannot be empty.")

    if event_feat_dim is None:
        raise ValueError("`event_feat_dim` has to be specified explicitly.")

    event_branch_kwargs = dict(
        layers_list=event_branch_layers,
        hidden_activation=event_hidden_activation,
        dropout=event_dropout,
        kernel_regularizer=kernel_regularizer,
    )

    event_branch, embedded_branches, model = build_multi_embedded_model(
        event_feat_dim=event_feat_dim,
        event_branch_kwargs=event_branch_kwargs,
        embedded_specs=normalized_specs,
        combination_layers=combination_layers,
        combined_activation=combined_activation,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        dropout=dropout,
        name=name,
    )

    model.event_branch = event_branch
    model.embedded_branches = {
        spec.name: branch for spec, branch in zip(normalized_specs, embedded_branches)
    }
    model.embedded_specs = normalized_specs

    return model


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
        n_features=event_feat_dim,
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


def xy_maker_wrapper(muon_prefix="mu"):
    def xy_maker(
        df, features, labels, weight=None
    ):
        muon_pattern = f"{muon_prefix}"+r"\d+_*"
        event_features = [k for k in features if not re.findall(muon_pattern, k)]
        event_feats = df[event_features]
        muon_feats = df_to_muon_array(df, epsilon_std=0, muon_prefix=muon_prefix)

        sample_weight = tf_utils.get_weight_column(df, weight)
        X = (event_feats, muon_feats)
        y = df[labels]
        return X, y, sample_weight

    return xy_maker

xy_maker_muon_embedding = xy_maker_wrapper(muon_prefix="mu")
xy_maker_shower_muon_embedding = xy_maker_wrapper(muon_prefix="shower_mu")

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
    n_features,
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
    print(f"{muon_pattern = }")
    print(f"{features = }")
    print([ (k, re.findall(muon_pattern, k)) for k in features])
    event_features = [k for k in features if not re.findall(muon_pattern, k)]
    n_event_features = len(event_features)
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
    n_features,
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

def define_custom_embedding_model(
    n_features,
    features,
    n_labels=1,
    embedded_properties=[{"pattern": r"mu\d+_*", "dim": 16, "layers": [64], "dropout": 0.3, "kernel_regularizer": 1e-4, "aggregation_method": "simple", "batch_size": None}, 
                         {"pattern": r"shower_mu\d+_*", "dim": 16, "layers": [64], "dropout": 0.3, "kernel_regularizer": 1e-4, "aggregation_method": "simple", "batch_size": None}],
    combination_layers=[64],
    hidden_activation="relu",
    dropout=0.3,
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
