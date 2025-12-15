import tfxkit.common.tf_utils as tf_utils
import tensorflow as tf
import logging
import keras
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Union

logger = logging.getLogger(__name__)



@dataclass
class EmbeddedBranchSpec:
    """
    Configuration container for embedded branches (muons, shower muons, energylosses, etc.).
    The dataclass is used by both the `define_embedded_model` function and the `xy_maker_embedded` function.

    Parameters
    ----------
    name: str
        The name of the embedded branch.
    
    Parameters for the xy_maker_embedded function:  
    ---------- 
    feature_patterns: Sequence[str]
        The feature patterns to use for the embedded branch.
    n_objects: int
        The number of objects to embed.
    index_start: int
        The index start for the objects.
    pad_strategy: Optional[str]
        The padding strategy to use for the embedded branch.
    pad_values: Optional[Sequence[float]]
        The padding values to use for the embedded branch.
    epsilon_std: Optional[float]
        The epsilon std to use for the embedded branch.
    embedding_dim: int
        The embedding dimension for the embedded branch.

    Parameters for the define_embedded_model function:
    ----------
    layers_list: Sequence[int]
        The list of layers for the embedded branch.
    hidden_activation: str
        The hidden activation for the embedded branch.
    final_activation: str
        The final activation for the embedded branch.
    dropout: float
        The dropout for the embedded branch.
    kernel_regularizer: float
        The kernel regularizer for the embedded branch.
    aggregation_method: str
        The aggregation method for the embedded branch.
    batch_size: Optional[int]


    """

    name: str
    feature_patterns: Sequence[str]
    n_objects: int = 10
    index_start: int = 1
    epsilon_std: Optional[float] = None
    embedding_dim: int = 16
    layers_list: Sequence[int] = field(default_factory=lambda: [64])
    hidden_activation: str = "relu"
    final_activation: str = "sigmoid"
    dropout: float = 0.3
    kernel_regularizer: float = 1e-4
    aggregation_method: str = "simple3"
    # batch_size: Optional[int] = None
    # build: bool = True
    pad_strategy: Optional[str] = "mean"
    pad_values: Optional[Sequence[float]] = None
    pad_and_mask_empty_objects: bool = False
    extra_branch_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.n_features = len(self.feature_patterns)
        self.layers_list = tf_utils.parse_layers_list(self.layers_list)

    def to_branch_kwargs(self) -> Dict[str, Any]:
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(f"{self.name = }")
        # print(f"{self.feature_patterns = }")
        # print(f"{self.n_features = }")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
            # batch_size=self.batch_size,
            name=self.name,
            # build=self.build,
            pad_and_mask_empty_objects=self.pad_and_mask_empty_objects,
        )
        branch_kwargs.update(self.extra_branch_kwargs)

        # print(f"{branch_kwargs = }")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'{type(self.layers_list) = } {self.layers_list = }')
        print(f'{type(list(self.layers_list)) = } {list(self.layers_list) = }')
        for k,v in branch_kwargs.items():
            print(f"{k = } \t {type(v) = } {v = }")
        return branch_kwargs

    def to_df_to_embedded_array_kwargs(self) -> Dict[str, Any]:
        return dict(
            feature_patterns=self.feature_patterns,
            n_objects=self.n_objects,
            index_start=self.index_start,
            pad_and_mask_empty_objects=self.pad_and_mask_empty_objects,
            pad_strategy=self.pad_strategy,
            pad_values=self.pad_values,
            epsilon_std=self.epsilon_std,
        )

    def get_embedded_columns(self):
        columns = []
        for object_idx in range(self.index_start, self.index_start + self.n_objects):
            columns.extend([pattern % object_idx for pattern in self.feature_patterns])
        return columns

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
    # batch_size: Optional[int] = None,
    # build: bool = True,
    pad_and_mask_empty_objects: bool = True,
    **mlp_kwargs,
) -> keras.Model:
    """
    General-purpose branch builder that applies the same MLP to each object in a
    sequence and aggregates the resulting embeddings.
    
    Parameters
    ----------
    pad_and_mask_empty_objects : bool, default=False
        If True, the last feature dimension is treated as a mask indicator (1.0 for
        real objects, 0.0 for padded). The mask is extracted before processing and
        used to zero out padded objects before aggregation. Set to True when using
        `df_to_embedded_array` with mask support.
    """

    # When pad_and_mask_empty_objects=True, input includes mask as last feature dimension
    input_features = n_features + 1 if pad_and_mask_empty_objects else n_features
    embedded_input = keras.Input(
        shape=(n_objects, input_features),
        name=f"{name}_input",
    )

    # Extract mask and actual features when pad_and_mask_empty_objects=True
    # Use Lambda layers for better readability and control over names
    if pad_and_mask_empty_objects:
        mask = keras.layers.Lambda(lambda t: t[:, :, -1:], name=f"EmbPadMask_{name}")(embedded_input)
        actual_features = keras.layers.Lambda(lambda t: t[:,:, :-1], name=f"ActlFeats_{name}")(embedded_input)
    else:
        mask = None
        actual_features = embedded_input

    # Build MLP for actual features only (excluding mask if present)
    embedded_mlp = tf_utils.define_mlp(
        features=n_features,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=embedding_dim,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        # batch_size=batch_size,
        # build=False,
        name=name,
        layers_prefix=name,
        # **{'kernel_initializer': 'he_normal'}
    )
    embedded_mlp.summary(expand_nested=True)
    # Pass actual_features (without mask) to the MLP
    embedded_sequence = time_distributed_layer(embedded_mlp)(actual_features)

    if aggregation_method not in AGGREGATION_METHODS:
        raise ValueError(
            f"Invalid aggregation method: {aggregation_method}. "
            f"Supported values: {list(AGGREGATION_METHODS.keys())}"
        )
    aggregation_layer = AGGREGATION_METHODS[aggregation_method]()
    
    # Apply mask to zero out padded objects before aggregation (if mask is present)
    if mask is not None:
        # mask shape: (batch, n_objects, 1), broadcasts to (batch, n_objects, embedding_dim)

        # masked_embeddings = embedded_sequence * mask # FIXTHIS
        masked_embeddings = embedded_sequence
        assert False, "NOT MASKING FOR NOW"
        # masked_embeddings = keras.layers.Lambda(lambda t: t * mask, name=f"MaskEmb_{name}")(embedded_sequence)
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # tf.print(f"{masked_embeddings = }")
        # tf.print(f"{mask = }")
        # tf.print(f"{embedded_sequence = }")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(f"{masked_embeddings.shape = } \n {type(masked_embeddings) = } \n {masked_embeddings = }")
        # print(f"{mask.shape = } \n {type(mask) = } \n {mask = }")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        branch_output = aggregation_layer(masked_embeddings)
    else:
        branch_output = aggregation_layer(embedded_sequence)

    model = keras.Model(inputs=embedded_input, outputs=branch_output, name=name)
    return model



def build_multi_embedded_model(
    features,
    *,
    event_branch_kwargs: Dict[str, Any],
    embedded_specs: Sequence[EmbeddedBranchSpec],
    combination_layers: Sequence[int],
    combined_activation: str = "relu",
    final_activation: str = "sigmoid",
    kernel_regularizer: float = 1e-4,
    dropout: float = 0.2,
    batch_size: Optional[int] = None,
    n_labels: int = 1,
    name="EmbeddedModel"
):
    """
    Builds a model that can consume an arbitrary number of embedded subnetworks.
    """

    layers_list = tf_utils.parse_layers_list(combination_layers)
    dropout_list = tf_utils.broadcast_argument_to_layers(dropout, n_layers=len(layers_list))

    merged_inputs = []
    merged_outputs = []
    embedded_branches = []
    embedded_features = []

    embedded_features = [spec.get_embedded_columns() for spec in embedded_specs]

    event_features = [k for k in features if not k in embedded_features]
    n_event_features = len(event_features)
    event_branch = build_event_branch(
        event_feat_dim=n_event_features,
        batch_size=batch_size,
        **event_branch_kwargs,
    )


    for spec in embedded_specs:        
        branch = build_embedded_branch(**spec.to_branch_kwargs(), features=features ) 
        embedded_branches.append(branch)
        n_input_features = spec.n_features + 1 if spec.pad_and_mask_empty_objects else spec.n_features
        tf_utils.build_model(branch, batch_size=batch_size, n_input_features=n_input_features)                
        merged_inputs.append(branch.input)
        merged_outputs.append(branch.output)

    merged_inputs = [event_branch.input] + merged_inputs
    merged_outputs = [event_branch.output] + merged_outputs

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

    output = keras.layers.Dense(n_labels, activation=final_activation, name=f"{name}_output")(x)
    model = keras.Model(inputs=merged_inputs, outputs=output, name=name)

    return event_branch, embedded_branches, model


def get_normalized_specs(embedded_specs):
    normalized_specs = []
    for spec in embedded_specs:
        if isinstance(spec, EmbeddedBranchSpec):
            normalized_specs.append(spec)
        elif hasattr(spec, "keys"):
            normalized_specs.append(EmbeddedBranchSpec(**dict(spec)))
        else:
            raise TypeError(f"Unsupported spec type: {type(spec)}")
    return normalized_specs

def define_embedded_model(
    features: Optional[Sequence[str]] = None,
    *,
    n_labels: int = 1,
    # event_feat_dim: Optional[int] = None,
    embedded_specs: Optional[Iterable[Union[EmbeddedBranchSpec, Dict[str, Any]]]] = None,
    event_branch_layers=(
        64,
        64,
    ),
    event_embedding_dim: int = 64,
    event_hidden_activation: str = "relu",
    event_final_activation: str = "sigmoid",
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

    normalized_specs = get_normalized_specs(embedded_specs)

    if not normalized_specs:
        raise ValueError("`embedded_specs` cannot be empty.")

    event_branch_kwargs = dict(
        layers_list=event_branch_layers,
        embedding_dim=event_embedding_dim,
        hidden_activation=event_hidden_activation,
        dropout=event_dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=event_final_activation,
    )

    event_branch, embedded_branches, model = build_multi_embedded_model(
        features=features,
        event_branch_kwargs=event_branch_kwargs,
        embedded_specs=normalized_specs,
        combination_layers=combination_layers,
        combined_activation=combined_activation,
        final_activation=final_activation,
        kernel_regularizer=kernel_regularizer,
        dropout=dropout,
        name=name,
        n_labels=n_labels,
    )

    model.event_branch = event_branch
    model.embedded_branches = {
        spec.name: branch for spec, branch in zip(normalized_specs, embedded_branches)
    }
    model.embedded_specs = normalized_specs

    return model


def build_event_branch(
    event_feat_dim: int,
    layers_list: Sequence[int] = [1028, 512, 256, 128, 64],
    hidden_activation="relu",
    embedding_dim=64,
    final_activation="sigmoid",
    dropout=0.2,
    kernel_regularizer=1e-4,
    batch_size=None,
    **kwargs,
):
    """
    Returns a Sequential sub-model for the event-level features.
    Example: 2 dense layers, can be adapted as needed.
    """

    event_input = keras.Input(shape=(event_feat_dim,), name="event_input")
    event_mlp = tf_utils.define_mlp(
        features=None,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=embedding_dim,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        build=False,
        # build=True,
        name="EventMLP",
        layers_prefix="Evt",
        **kwargs,
    )
    tf_utils.build_model(event_mlp, batch_size=batch_size, n_input_features=event_feat_dim)
    event_mlp(event_input)
    return event_mlp



##
## XY Maker Functions
##


def xy_maker_embedded(
    df,
    features,
    labels,
    weight: Optional[str] = None,
    *,
    model_parameters: Optional[Dict[str, Any]] = None,
):
    """
    Generalized xy_maker for embedded models that extracts event features and
    multiple embedded object arrays.
    
    Parameters
    ----------
    embedded_specs:
        List of EmbeddedBranchSpec (or dicts) defining each embedded branch.
    model_parameters:
        Dictionary containing the model parameters. It must contain the following keys:
        - embedded_specs: List of EmbeddedBranchSpec (or dicts) defining each embedded branch.

    """

    normalized_specs = get_normalized_specs(model_parameters.get("embedded_specs"))
    if not normalized_specs:
        raise ValueError("`embedded_specs` cannot be empty in the model_parameters")

    # embedded_feature_lists

    embedded_arrays = []
    embedded_features = []
    for spec in normalized_specs:
        embedded_features.extend(spec.get_embedded_columns())
        embedded_arr = df_to_embedded_array(
            df,
            **spec.to_df_to_embedded_array_kwargs(),
        )
        embedded_arrays.append(embedded_arr)

    event_features = [k for k in features if not k in embedded_features] 
    logger.info(f"Event features: {event_features}")
    event_array = df[event_features].values

    # Build X tuple: (event_array, embedded_array_1, embedded_array_2, ...)
    X = (event_array,) + tuple(embedded_arrays)
    
    y = df[labels].values
    sample_weight = tf_utils.get_weight_column(df, weight)
    
    return X, y, sample_weight



def df_to_embedded_array(
    df,
    *,
    feature_patterns,
    n_objects: int,
    index_start: int = 1,
    epsilon_std: Optional[float] = None,
    pad_values: Optional[Sequence[float]] = None,
    pad_strategy: Optional[str] = "mean",
    pad_and_mask_empty_objects: bool = True,
):
    """
    Generalized function that extracts per-object features using format strings
    (e.g., "mu%s_pos_x"). Indices run from `index_start` for `n_objects` steps.
    
    Returns
    -------
    arr : np.ndarray
        Array of shape (n_events, n_objects, n_features + 1) containing the extracted
        features plus a mask indicator. The first `n_features` dimensions contain
        the actual features (padded objects are filled according to `pad_values`,
        `pad_strategy`, or `epsilon_std`). The last dimension is a mask indicator:
        1.0 for real objects, 0.0 for padded objects. This mask can be used by
        aggregation layers to ignore padded objects.

    Parameters
    ----------
    pad_values:
        Optional list of per-feature values used to fill padded objects. If not
        provided and `pad_strategy` is "mean" or "median", the values are
        computed from the DataFrame. If both `pad_values` and `pad_strategy` are
        None, falls back to the `epsilon_std` behavior.
    """

    pad_strategies = {
        "mean": np.nanmean,
        "median": np.nanmedian,
    }

    feature_patterns = list(feature_patterns)
    n_features = len(feature_patterns)

    if pad_values is not None:
        pad_values = list(pad_values)
        if len(pad_values) != n_features:
            raise ValueError(
                "Length of pad_values must match number of feature_patterns"
            )
            
    elif pad_and_mask_empty_objects:
        if pad_strategy not in pad_strategies:
            raise ValueError(f"Invalid pad_strategy: {pad_strategy}. Supported values: {list(pad_strategies.keys())}")
        pad_values = []
        for pattern in feature_patterns:
            cols = [
                col
                for col in (
                    pattern % idx
                    for idx in range(index_start, index_start + n_objects)
                )
                if col in df.columns
            ]
            if not cols:
                raise ValueError(
                    f"No columns found in DataFrame for pattern '{pattern}'"
                )
            print(f"{cols = }")
            values = df[cols].values
            pad_values.append(float(pad_strategies[pad_strategy](values)))

    if pad_and_mask_empty_objects:
        n_features_with_mask = n_features + 1         
    else:
        n_features_with_mask = n_features

    arr = np.zeros((len(df), n_objects, n_features_with_mask), dtype=float)

    if epsilon_std is not None and pad_values is not None:
        raise ValueError("Cannot use both epsilon_std and pad_values/pad_strategy. Please provide only one.")


    for offset, object_idx in enumerate(range(index_start, index_start + n_objects)):
        columns = [pattern % object_idx for pattern in feature_patterns]
        object_data = df[columns].values
        
        # Check if this object is real (has at least one non-zero, non-NaN value)
        is_real = np.any((object_data != 0) & ~np.isnan(object_data), axis=1)
        
        zero_mask = object_data == 0
        if pad_values is not None:
            for feat_idx in range(n_features):
                if np.any(zero_mask[:, feat_idx]):
                    object_data[zero_mask[:, feat_idx], feat_idx] = pad_values[feat_idx]
        elif epsilon_std is not None:
            if zero_mask.any():
                object_data[zero_mask] = np.random.normal(
                    loc=0,
                    scale=epsilon_std,
                    size=zero_mask.sum(),
                )
        else:
            object_data[zero_mask] = np.nan

        # Store the actual features
        if pad_and_mask_empty_objects:
            arr[:, offset, :n_features] = object_data
            # Append mask as the last feature: 1.0 for real objects, 0.0 for padded
            arr[:, offset, n_features] = is_real.astype(float)
        else:
            arr[:, offset, :] = object_data

    return arr



##
## Aggregation Methods
##

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
    def call(self, inputs):
        sum_agg = tf.reduce_sum(inputs, axis=1)
        return tf.concat([sum_agg], axis=-1)


@keras.saving.register_keras_serializable()
class SimpleAggregation3(keras.layers.Layer):
    def call(self, inputs):
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        sum_agg = tf.reduce_sum(inputs, axis=1)
        std_agg = tf.math.reduce_std(inputs, axis=1)
        return tf.concat([mean_agg, max_agg, min_agg, sum_agg, std_agg], axis=-1)

@keras.saving.register_keras_serializable()
class SimpleAggregation2(keras.layers.Layer):
    def call(self, inputs):
        # mean_agg = tf.reduce_mean(tf.where(tf.math.is_finite(inputs), inputs, 0), axis=1) #valid_counts
        is_valid = tf.math.is_finite(inputs) & (inputs != 0)
        valid_counts = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=1)

        masked_inputs = tf.where(is_valid, inputs, tf.zeros_like(inputs))
        mean_agg = tf.reduce_sum(masked_inputs, axis=1) / valid_counts
        max_agg = tf.reduce_max(tf.where(is_valid, inputs, -tf.float32.max), axis=1)
        min_agg = tf.reduce_min(tf.where(is_valid, inputs, tf.float32.max), axis=1)
        
        # To check the values in mean_agg, max_agg, min_agg, you can use tf.print,
        # which works within TensorFlow's graph mode (unlike regular Python print)
        # tf.print("mean_agg:", mean_agg)
        # tf.print("max_agg:", max_agg)
        # tf.print("min_agg:", min_agg)
        # print('--------------------------------')
        
        return tf.concat([mean_agg, max_agg, min_agg], axis=-1)

@keras.saving.register_keras_serializable()
class AggregationFactory(keras.layers.Layer):
    def __init__(self, method_name: str = "simple"):
        super().__init__()
        self.method_name = method_name
        self.aggregation_layer = AGGREGATION_METHODS[method_name]

    def call(self, inputs):
        return self.aggregation_layer(inputs)


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
