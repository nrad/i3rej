import tfxkit.common.base_utils as utils
import tfxkit.common.tf_utils as tf_utils
from tfxkit.common.base_utils import combine_weight_columns
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import pylab as plt
import matplotlib as mpl
from uncertainties import ufloat
import time
from scipy.signal import savgol_filter
import scipy
from copy import deepcopy
import textwrap


def make_labels(label_name):
    labels = {
        "n_photons_passed": r"$N_{%s}^{\gamma}$" % (label_name),
        "n_photons_accepted": r"$N_{NN}^{\gamma}$",
        "n_events_eff": r"$N^{NN}_{eff} (pred*gen weights)$",
        "n_events_eff_pred": r"$N^{%s}_{eff}$" % (label_name),
        "n_events_eff_pred_sel": r"$N^{%s}_{eff}$" % (label_name),
        "n_events_eff_gen": r"$N^{%s}_{eff} (gen weight)$" % (label_name),
        "n_events_accepted": r"$N^{NN}_{events}$",
        "n_photons_simulated_sel": r"$N^{simulated}_{\gamma}$ (sel)",
        "n_photons_simulated": "$<N_{\gamma}^{sim}>$",
        "n_photons_simulated_0": "$<N_{simulated\ showers}>$",
        "n_photons_simulated_2": "$<N_{\gamma}^{sim} (n_{\gamma}^2)>$",
        "n_eff_passed": "$<N_{eff}^{%s}>$" % (label_name),
        "n_eff_sampled_expected": r"$<N^{%s}_{eff,raw}>$" % (label_name),
        "n_eff_sampled_gen_expected": r"$<N^{%s}_{eff}>$" % (label_name),
        "n_photons_sampled_expected": r"$<N_{\gamma}>$",
        "min_pred": "Minimum Acceptance Probability",
        "fake_pred": "Ideal",
        "pred": "model",
        #'uniform_pred':'Dystopian',
        "uniform_pred": "Uniform",
    }
    return labels


# labels = make_labels("L3")

infos = {

    "fake_pred": dict(color="green", label="Optimistic", ls=":", lw=2),
    "uniform_pred": dict(color="red", label="Uniform Sampling", ls=":", lw=2),
    "pseudo_pred_gauss": dict(color="green", label="Optimistic Sampling", ls=":", lw=2),
    "pseudo_pred_2gauss": dict(color="green", label="2Gauss", ls="--", lw=2),
    "pseudo_pred_uniform": dict(color="red", label="Uniform Sampling", ls=":", lw=2),
    "pred": dict(color="blue", label="Model", ls="", lw=2),

}


def apply_selection(df, selection_dict={}, selection_weight_column="selection_weights"):
    """
    Apply selection to a dataframe and return the resulting dataframe.

    """
    dfs = {}
    len_df = len(df)

    for sel, frac in selection_dict.items():
        df_ = df.query(sel)  # .sample(frac)
        dfs[sel] = df_
    del df

    sum_lengths = sum(map(len, dfs.values()))
    sum_values = sum(selection_dict.values())
    print(sum_lengths, sum_values)

    selection_weights = {}
    for sel, frac in selection_dict.items():
        if frac > 1:
            raise ValueError(
                "Fractions given as values in selection_dict must be smaller than 1."
            )
        n = int(len(dfs[sel]) * frac)
        print(f"{sel}: selecting {n} out of {len(dfs[sel])}")
        dfs[sel] = dfs[sel].sample(n)
        selection_weights[sel] = 1.0 / frac if frac else 0

    df = pd.concat(dfs.values()).sample(frac=1)

    if selection_weight_column is not None:
        df[selection_weight_column] = 0
        for sel, weight in selection_weights.items():
            df.loc[df.query(sel).index, selection_weight_column] = weight
    return df


def get_sample_weights_from_reference(y, y_ref, normalize=True):

    n = len(y)
    n_ones = np.count_nonzero(y)
    n_zeros = n - n_ones

    n_ref = len(y_ref)
    n_ones_ref = np.count_nonzero(y_ref)
    n_zeros_ref = n_ref - n_ones_ref

    weights_reference = np.ones(n_ref)
    weights = np.ones(n)

    weights[y == 0] = (n_zeros_ref / n_ref) * (n / n_zeros)
    weights[y == 1] = (n_ones_ref / n_ref) * (n / n_ones)

    if normalize:
        weights /= n
        weights_reference /= n_ref

    return weights, weights_reference


# calc_livetime = lambda w: w.sum()/(w*w).sum()
# n_eff = lambda w: w.sum()*w.sum()/(w*w).sum()
calc_rel_uncert = lambda w: np.sqrt((w * w).sum()) / w.sum()
calc_n_eff = lambda w: w.sum() ** 2 / (w**2).sum()


def calc_livetime(df, query_or_mask=None, weight_col="weights", return_dict=False):
    if query_or_mask:
        if isinstance(query_or_mask, str):
            df = df.query(query_or_mask)
        elif query_or_mask.dtype == bool:
            df = df[query_or_mask]
        else:
            df = df.loc[query_or_mask]
    if weight_col and weight_col in df:
        w = df[weight_col]
    else:
        w = df

    R = w.sum()
    DeltaR2 = (w * w).sum()
    livetime = R / DeltaR2
    DeltaR = np.sqrt(DeltaR2)
    uncert = DeltaR / R
    ret_dict = dict(
        R=R,
        DeltaR2=DeltaR2,
        DeltaR=DeltaR,
        livetime=livetime,
        total_rate=ufloat(R, round(DeltaR, 5)),
        ulivetime=ufloat(livetime, round(uncert, 5)),
        N_eff=(R * R / DeltaR2),
    )
    ret = livetime if not return_dict else ret_dict
    return ret


def get_weight_from_pred(pred, eps=1e-11):
    weights = 1.0 / (pred + eps)
    size = len(pred)
    rand = np.random.rand(size)
    mask = rand < pred
    # print(weights.sum(), factor)
    return weights, mask


def get_class_probability(
    df,
    label="Level3",
    pred="pred",
    min_val=1e-5,
    weight_columns=None,
    bins=50,
    return_func=False,
    add_prob=True,
    normalize=True,
    # normalize
):
    """
    Get the binned probabilities to have the positive class as a function of the prediction variable
    """
    weights = combine_weight_columns(df, weight_columns)
    if normalize:
        vals = normalize_range(df[pred], target=(1e-5, 1))

    cls0 = df[label] == 0
    cls1 = df[label] == 1
    h0, be = np.histogram(vals[cls0], weights=weights[cls0], bins=bins)
    h1, be = np.histogram(vals[cls1], weights=weights[cls1], bins=bins)

    # n = 1
    # factor = h0[-n:].sum() / h1[-n:].sum()
    # h0 = h0 * factor

    prob = h1 / (h0 + h1)
    if min_val is None:
        min_val = min([p for p in prob if p > 0])

    prob = np.where(prob, prob, min_val)
    b = (be[1:] + be[:-1]) / 2

    if return_func in [False, None]:
        return prob, (be[1:] + be[:-1]) / 2
    elif return_func == "interp":
        func = lambda x: np.interp(x, b, prob, left=prob[0], right=prob[-1], period=5)
    elif return_func == "polyfit":
        degree = 3
        coefs = np.polyfit(b, np.log(prob), degree)
        func = lambda x: np.exp(np.poly1d(coefs)(x))
        print(f"{coefs = }")
    else:
        raise ValueError(f"return_func={return_func} not recognized")

    if add_prob:
        vals = func(df[pred])
        if normalize:
            vals = normalize_range(vals, target=(1e-5, 1))
        prob_name = add_prob if isinstance(add_prob, str) else f"{pred}_prob"
        df[prob_name] = vals
    return func


def get_pred_summary(
    df,
    pred="pred",
    passed="passed",
    col_n_photons="n_photons",
    col_flux_weights="weights",
    col_sel_weights="selection_weights",
    min_pred_range=np.linspace(0, 1, 100),
    full=False,
):
    """
    calculate summary information for a given prediction
    """
    steps = []
    # pred = df_[pred_name]
    pred = df[pred] if isinstance(pred, str) else pred
    passed = df[passed] if isinstance(passed, str) else passed
    w_gen = df[col_flux_weights] if isinstance(col_flux_weights, str) else col_flux_weights
    w_sel = df[col_sel_weights] if isinstance(col_sel_weights, str) else col_sel_weights
    n_photons = df[col_n_photons] if isinstance(col_n_photons, str) else col_n_photons

    for min_pred in min_pred_range:
        step_pred = np.where(pred <= min_pred, min_pred, pred)
        w_pred, mask_sample = get_weight_from_pred(step_pred)
        w_passed_sampled = w_pred[mask_sample & passed]

        w_pred_passed = w_pred[passed]
        w_gen_passed = w_gen[passed]
        step_pred_passed = step_pred[passed]

        n_photons_w_sell = n_photons * w_sel
        n_photons_w_sell_pred = n_photons_w_sell * step_pred

        n_photons_sampled_expected = (n_photons_w_sell_pred).sum()
        n_photons_sampled_gen_expected = (n_photons_w_sell_pred * w_gen).sum()

        n_eff_sampled_passed = calc_n_eff(w_passed_sampled)
        n_eff_sampled_expected = calc_expected_n_eff(w_pred_passed)

        n_eff_sampled_gen_expected = calc_expected_n_eff(
            w_pred_passed * w_gen_passed, p=step_pred_passed
        )
        n_eff_sampled_gen = calc_n_eff(w_passed_sampled * w_gen[mask_sample & passed])

        steps_dict = dict(
            min_pred=min_pred,
            n_photons_sampled_expected=n_photons_sampled_expected,
            n_photons_sampled_gen_expected=n_photons_sampled_gen_expected,
            n_eff_sampled_passed=n_eff_sampled_passed,
            n_eff_sampled_expected=n_eff_sampled_expected,
            n_eff_sampled_gen_expected=n_eff_sampled_gen_expected,
            n_eff_sampled_gen=n_eff_sampled_gen,
        )

        if full:
            pass

        steps.append(steps_dict)

    sdf = pd.DataFrame(steps)
    return sdf


def calc_expected_n_eff(w, p=None):
    # w = combine_weight_columns(df, weights)
    # prob = prob if prob is not None else np.ones(size=len(w))
    # w.sum()**2/(w**2).sum()
    if p is None:
        expected_n_eff = (len(w) * len(w)) / (w).sum()
    else:
        expected_n_eff = (p * w).sum() ** 2 / (p * (w) ** 2).sum()
    # print(w.sum(), len(w), expected_n_eff)
    return expected_n_eff


def calc_expected_livetime(w, p=None):
    if p is None:
        expected_lt = (len(w)) / (w).sum()
    else:
        expected_lt = (p * w).sum() / (p * (w) ** 2).sum()
    return expected_lt


def get_pred_speed(
    pred, passed=None, n_photons=None, flux_weights=None, sel_weights=None, full=False
):
    pred_passed = pred[passed]
    w_pred, mask_sample = get_weight_from_pred(pred)

    n_eff_passed = calc_expected_n_eff(
        w_pred[passed] * flux_weights[passed] * sel_weights[passed], pred[passed]
    )
    n_eff_noflux_passed = calc_expected_n_eff(
        w_pred[passed] * sel_weights[passed], pred[passed]
    )
    # n_eff_passed = calc_n_eff( (w_pred * flux_weights * sel_weights)[passed & mask_sample] )
    n_photons_simulated = (n_photons * sel_weights * pred).sum()
    speed = n_eff_passed / n_photons_simulated
    results = dict(
        n_eff_passed=n_eff_passed,
        n_photons_simulated=n_photons_simulated,
        n_eff_noflux_passed=n_eff_noflux_passed,
        speed=speed,
    )

    if full:
        n_eff_passed_nom = (
            pred[passed] * w_pred[passed] * flux_weights[passed] * sel_weights[passed]
        ).sum() ** 2
        n_eff_passed_denom = (
            pred[passed]
            * (w_pred[passed] * flux_weights[passed] * sel_weights[passed]) ** 2
        ).sum()

        livetime_passed_nom = (
            pred[passed] * w_pred[passed] * flux_weights[passed] * sel_weights[passed]
        ).sum()
        livetime_passed_denom = n_eff_passed_denom
        results.update(
            n_eff_passed_nom=n_eff_passed_nom,
            n_eff_passed_denom=n_eff_passed_denom,
            livetime_passed_nom=livetime_passed_nom,
            liveitme_passed_denom=livetime_passed_denom,
            livetime_passed=livetime_passed_nom / livetime_passed_denom,
            n_photons_rejected=(n_photons * sel_weights * (1 - pred)).sum(),
            n_photons_passed=(n_photons * sel_weights * pred)[passed].sum(),
            n_photons_simulated_flux=(
                n_photons * sel_weights * pred * flux_weights
            ).sum(),
            n_photons_passed_flux=(n_photons * sel_weights * pred * flux_weights)[
                passed
            ].sum(),
        )
    return results


def get_pred_speedup(
    df,
    pred="pred",
    passed="passed",
    col_n_photons="n_photons",
    col_flux_weights="weights",
    col_sel_weights="selection_weights",
    min_pred_range=np.linspace(0, 1, 100),
    full=False,
    step_pred_func=None,
):
    """
    calculate summary information for a given prediction
    """
    steps = []

    pred = df[pred] if isinstance(pred, str) else pred
    passed = df[passed].astype(bool) if isinstance(passed, str) else passed
    # print(passed)
    w_gen = df[col_flux_weights] if isinstance(col_flux_weights, str) else col_flux_weights
    w_sel = df[col_sel_weights] if isinstance(col_sel_weights, str) else col_sel_weights
    n_photons = df[col_n_photons] if isinstance(col_n_photons, str) else col_n_photons

    speed_dict_nominal = get_pred_speed(
        np.ones_like(pred),
        passed=passed,
        n_photons=n_photons,
        flux_weights=w_gen,
        sel_weights=w_sel,
    )

    speed_nominal = speed_dict_nominal["speed"]

    n_eff_passed_nominal = speed_dict_nominal["n_eff_passed"]
    n_photons_simulated_nominal = speed_dict_nominal["n_photons_simulated"]

    for min_pred in min_pred_range:
        if step_pred_func is not None:
            if not callable(step_pred_func):
                raise ValueError(f"step_pred_func={step_pred_func} is not callable!")
            step_pred = step_pred_func(pred, min_pred)
        else:
            step_pred = np.where(pred <= min_pred, min_pred, pred)

        speed_dict = get_pred_speed(
            step_pred,
            passed=passed,
            n_photons=n_photons,
            flux_weights=w_gen,
            sel_weights=w_sel,
            full=full,
        )

        n_eff_passed = speed_dict["n_eff_passed"]
        n_photons_simulated = speed_dict["n_photons_simulated"]
        speed = speed_dict["speed"]

        n_eff_passed_normed = n_eff_passed / n_eff_passed_nominal
        n_photons_simulated_normed = n_photons_simulated / n_photons_simulated_nominal
        speed_normed = speed / speed_nominal

        steps_dict = dict(
            min_pred=min_pred,
            **speed_dict,
            n_eff_passed_normed=n_eff_passed_normed,
            n_photons_simulated_normed=n_photons_simulated_normed,
            speed_normed=speed_normed,
        )

        steps.append(steps_dict)

    sdf = pd.DataFrame(steps)
    return sdf


def get_max_speedup(model, df, features, truth, batch_size, key="speed_normed"):
    """
    Get the maximum speed up for a given ModelFactory instance
    """
    # df = mf.df_test
    # df['pred'] = mf.model.predict(df[mf.features], batch_size=mf.batch_size)
    X_test = df[features]
    y_test = df[truth]
    df["pred"] = model.predict(X_test, batch_size=batch_size)
    df["passed"] = y_test
    speedup_normed = get_pred_speedup(
        df,
        pred="pred",
        col_flux_weights="flux_weights",
        col_sel_weights="selection_weights",
    )[key]
    return speedup_normed.max()


def plot_computation_cost(
    df,
    pred_name="pred",
    sim_levels=["generated", "triggered", "filtered", "Level3"],
    key="sim_level",
):
    fig, ax = plt.subplots()
    colors = {"generated": "C1", "triggered": "C2", "filtered": "C5", "Level3": "C0"}
    if key not in df.columns:
        utils.label_sim_levels(df, sim_levels=sim_levels, key=key)
    for level in sim_levels:
        df_ = df.query(f'{key}=="{level}"')
        # vals = df_[pred_name] * df_['selection_weight'] * df_['n_photons']
        vals = df_["selection_weights"] * df_["n_photons"]
        ax.scatter(
            df_[pred_name],
            vals,
            s=0.1,
            alpha=0.1,
            label=level,
            color=colors.get(level, None),
        )

        ax.set_yscale("log")
        ax.set_xlabel("sampling probability ($p_{i}$)")
        ax.set_ylabel(
            "weighted comp. cost ($w^{sel.}_{i} \cdot p_{i} \cdot n_{\gamma,\ i}$)",
            fontsize=14,
        )

    leg = ax.legend()
    for handle in leg.legend_handles:
        handle.set_alpha(1)
        handle.set_sizes([20])
    return fig, ax


"""
    Functions for creating "pseudo" predictions
"""

from scipy.stats import truncnorm


def normalize_range(vals, target=(1e-5, 1)):
    """
    normalize values <vals> to the target range <target>
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val) * (target[1] - target[0]) + target[0]


def truncnorm_wrapper(a_trunc, b_trunc, loc, scale):
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    return truncnorm(a, b, loc, scale)


def add_gaussian_preds(
    df,
    truth_label="Level3",
    name="fake_pred",
    loc_pos=1,
    scale_pos=0.1,
    loc_neg=0.1,
    scale_neg=0.1,
):
    x_range = (0, 1)
    x = np.linspace(*x_range, 100)
    pdf_pos = truncnorm_wrapper(*x_range, loc=loc_pos, scale=scale_pos)
    pdf_neg = truncnorm_wrapper(*x_range, loc=loc_neg, scale=scale_neg)

    cls0 = df[truth_label] == 0
    cls1 = df[truth_label] == 1

    df.loc[cls0, name] = pdf_neg.rvs(cls0.sum())
    df.loc[cls1, name] = pdf_pos.rvs(cls1.sum())


def add_double_gaussian_preds(
    df,
    truth_label="Level3",
    name="2g_pred",
    loc_pos=0.9,
    scale_pos=0.3,
    loc_neg=0.1,
    scale_neg=0.4,
):
    x_range = (0, 1)
    x = np.linspace(*x_range, 100)
    pdf_pos = truncnorm_wrapper(*x_range, loc=loc_pos, scale=scale_pos)
    pdf_neg = truncnorm_wrapper(*x_range, loc=loc_neg, scale=scale_neg)

    cls0 = df[truth_label] == 0
    cls1 = df[truth_label] == 1

    n0 = cls0.sum()
    n1 = cls1.sum()

    pdf_pos1 = pdf_pos.rvs(n1)
    pdf_pos2 = truncnorm_wrapper(*x_range, loc=1, scale=0.05).rvs(n1)
    mask = np.random.choice([True, False], size=n1)
    pdf_pos = np.where(mask, pdf_pos1, pdf_pos2)

    pdf_neg1 = pdf_neg.rvs(n0)
    pdf_neg2 = truncnorm_wrapper(*x_range, loc=0, scale=0.05).rvs(n0)
    mask = np.random.choice([True, False], size=n0)
    pdf_neg = np.where(mask, pdf_neg1, pdf_neg2)

    df.loc[cls0, name] = pdf_neg
    df.loc[cls1, name] = pdf_pos


def add_uniform_preds(df, truth_label="Level3", name="uniform_pred"):
    cls0 = df[truth_label] == 0
    cls1 = df[truth_label] == 1
    df.loc[cls1, name] = np.random.rand(sum(cls1))
    df.loc[cls0, name] = np.random.rand(sum(cls0))


def add_coin_flip_preds(df, truth_label="Level3", name="coin_flip_pred", p=0.5):
    cls0 = df[truth_label] == 0
    cls1 = df[truth_label] == 1
    # df.loc[cls1, name] = np.random.rand(sum(cls1)) < p
    df.loc[cls1, name] = p
    df.loc[cls0, name] = p


def add_pseudo_preds(
    df,
    truth_label="Level3",
    pred_prefix="pseudo_pred",
    loc_pos=1,
    scale_pos=0.3,
    loc_neg=0.1,
    scale_neg=0.3,
    p=0.5,
):
    add_gaussian_preds(
        df,
        truth_label=truth_label,
        name=f"{pred_prefix}_gauss",
        loc_pos=loc_pos,
        scale_pos=scale_pos,
        loc_neg=loc_neg,
        scale_neg=scale_neg,
    )
    add_double_gaussian_preds(
        df,
        truth_label=truth_label,
        name=f"{pred_prefix}_2gauss",
        loc_pos=loc_pos,
        scale_neg=scale_neg,
        loc_neg=loc_neg,
        scale_pos=scale_pos,
    )
    add_uniform_preds(df, truth_label=truth_label, name=f"{pred_prefix}_uniform")
    add_coin_flip_preds(
        df, truth_label=truth_label, name=f"{pred_prefix}_coin_flip", p=p
    )


###
### Custom Activation Function using logistic function
###


def generalized_logistic_function(x, a, b, c):
    # inspired by https://en.wikipedia.org/wiki/Logistic_function#Modeling_early_COVID-19_cases
    return 1 / ((1 + a * np.exp(-c * (x - b))) ** (1 / a))


def logistic_activation(x, a=1.0, b=0.0, c=1.0):
    return 1 / (tf.pow((1 + a * tf.exp(-c * (x - b))), 1 / a))


@tf.keras.utils.register_keras_serializable(
    package="I3K", name="LogisticActivationLayer"
)
class LogisticActivationLayer(tf.keras.layers.Layer):
    """
    Tensorflow Activation layer with the logistic shape as free parameters
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        # Initialize parameters a, b, c as trainable weights
        self.logistic_shape = self.add_weight(
            name="logistic_shape",
            shape=(3,),
            initializer=tf.constant_initializer([1.0, 0.0, 1.0]),
            trainable=True,
        )

    def call(self, inputs):
        a, b, c = tf.unstack(self.logistic_shape)
        return 1 / (tf.pow((1 + a * tf.exp(-c * (inputs - b))), 1 / a))


def remove_final_activation(model):
    import tensorflow as tf
    import keras

    """
        based on the underrated answer of @carlos-bermudez in: 
        https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
    """

    final_layer_name = model.layers[-1].name

    def f(layer):
        config = layer.get_config()
        # if not isinstance(layer, Activation) and layer.name in model.output_names:
        if not isinstance(layer, keras.layers.Activation) and layer.name in [final_layer_name]:
            print(f"removing activation in {layer}")
            config.pop("activation", None)
        layer_copy = layer.__class__.from_config(config)
        return layer_copy

    copy_model = tf.keras.models.clone_model(model, clone_function=f)
    copy_model.build(model.input_shape)
    copy_model.set_weights(model.get_weights())
    return copy_model


def model_modifier_logistic_activation(model, **kwargs):
    custom_act_model = remove_final_activation(model)
    custom_act_model.add(LogisticActivationLayer("logistic_activation"))
    return custom_act_model



###
### Functions for plotting
###


def plot_nom_denom_ratio(
    axs,
    sdf,
    nom,
    denom,
    x="min_pred",
    label=None,
    norm_idx=-1,
    infos=infos,
    label_name="CscdBDT",
    label_wrap_width=30,
    title=None,
):
    labels = make_labels(label_name)
    ax = axs[0]
    nom_label = labels.get(nom, nom)
    x_label = labels.get(x, x)
    # ls = None if label in ['pred'] else '--'
    # ls = '--'
    # color = colors.get(label,None)
    plot_kw = deepcopy(infos.get(label, {}))
    plot_kw.setdefault("alpha", 0.8)
    plot_kw.setdefault("ls", None)
    plot_kw.setdefault("lw", 2)
    print(label, plot_kw)

    label = plot_kw.pop("label", label)
    label = "\n".join(textwrap.wrap(label, label_wrap_width))

    ax.plot(sdf[x], sdf[nom], **plot_kw)
    ax.set_xlabel(x_label)
    ax.set_ylabel(nom_label)

    ax = axs[1]
    denom_label = labels.get(denom, denom)
    ax.plot(sdf[x], sdf[denom], **plot_kw)
    ax.set_ylabel(denom_label)
    ax.set_xlabel(x_label)

    ax = axs[2]
    # x = 'min_pred'
    # y = 'n_events_eff'
    y = sdf[nom] / sdf[denom]
    if not norm_idx is None:
        y /= sdf[nom].iloc[norm_idx] / sdf[denom].iloc[norm_idx]
    ax.plot(sdf[x], y, label=label, **plot_kw)
    ax.set_ylabel(r"%s/%s" % (nom_label, denom_label))
    ax.set_xlabel(
        x_label,
    )
    if title:
        axs[1].set_title(title)
    return axs


def make_tri_plot(
    nom,
    denom,
    x="min_pred",
    sdfs=None,
    norm_idx=-1,
    save_path=None,
    legend=True,
    infos=infos,
    label_name="CscdBDT",
    label_wrap_width=30,
    title=None,
):
    model_names = list(sdfs)
    fig, axs = plt.subplots(
        ncols=3, nrows=1, squeeze=True, figsize=(15, 5), constrained_layout=True,
    )
    fig.tight_layout(pad=3)
    axs = axs.flatten()
    for pred_name in model_names:
        sdf = sdfs[pred_name]
        plot_nom_denom_ratio(
            axs,
            sdf,
            nom,
            denom,
            x,
            label=pred_name,
            # label=pred_name,
            infos=infos,
            norm_idx=norm_idx,
            label_name=label_name,
            label_wrap_width=label_wrap_width,
            title=title,
        )
    if legend:
        # fig.legend(ncol=min(4, len(model_names)), loc='upper center', prop={'size':19})
        legend_kwargs = dict(
            ncol=min(4, len(model_names)),
            fontsize=15,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
        )
        if isinstance(legend, dict):
            legend_kwargs.update(legend)
        fig.legend(**legend_kwargs)

    if save_path:
        utils.savefig(fig, save_path)
    fig.tight_layout()
    return fig, axs

def plot_speedup(plotter, 
                df=None,
                #  label='CscdBDT',
                 pred='pred',
                 truth_label='CscdBDT',
                 n_photons='n_photons',
                 flux_weights='flux_weights',
                 selection_weights='selection_weights',
                 plot_name="speedup",
                 nom='n_eff_passed',
                 denom='n_photons_simulated',
                 ylim=(0, 3.0),
                 selection_dict={},
                 model_name=None,
                 save_results=False,
                 train=False,
                 ):

    if df is None:
        if train:
            df = plotter.data.df_train
        else:
            df = plotter.data.df_test
    else:
        df = df

    sdfs = {}
    model_name = model_name if model_name else plotter.config.info.model_name
    selection_dict.update({model_name:None})
    if selection_dict:
        for label, selection in selection_dict.items():
            if selection is not None:
                df_ = df.query(selection)
            else:
                df_ = df
            sdfs[label] = get_pred_speedup(df_, 
                                           pred=pred, 
                                           passed=truth_label, 
                                           col_n_photons=n_photons, 
                                           col_flux_weights=flux_weights, 
                                           col_sel_weights=selection_weights)
    add_uniform_preds(df, truth_label=truth_label)

    pseudo_preds = ['uniform_pred',]
    for pred in pseudo_preds:
        sdfs[pred] = get_pred_speedup(df, 
                                      pred=pred, 
                                      passed=truth_label, 
                                      col_n_photons=n_photons, 
                                      col_flux_weights=flux_weights, 
                                      col_sel_weights=selection_weights)
    tri_plot = make_tri_plot(nom, 
                                denom, 
                                x="min_pred", 
                                sdfs=sdfs)
    tri_plot[-1][-1].set_ylim(*ylim)
    if plot_name:
        plotter.save_fig(tri_plot[0],  f"{plot_name}"+ ("_train" if train else ""))
        if save_results:
            import pickle
            result_path = plotter.resolve_plot_path(f'{plot_name}.pkl')
            with open(result_path, 'wb') as f:
                pickle.dump(sdfs, f)
    return tri_plot, sdfs
  
