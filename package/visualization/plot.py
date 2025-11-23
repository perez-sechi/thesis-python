import shap
import numpy as np
import matplotlib.pylab as pl
import matplotlib.ticker as mtick

from package.exploration.schema import is_categorical


def shap_stats_at_variable_percentile(
    variable_index,
    X_variable,
    shap_values,
    abs_shap_values,
    starting_percentile,
    ending_percentile
):
    starting_percentile_value = np.percentile(
        X_variable, starting_percentile
    )
    ending_percentile_value = np.percentile(
        X_variable, ending_percentile
    )
    idx_survivors = np.where(
        (X_variable >= starting_percentile_value)
        & (X_variable <= ending_percentile_value)
    )[0]

    abs_all_shap_sum = np.sum(abs_shap_values[idx_survivors], axis=(0, 1))
    abs_variable_shap_sum = np.sum(
        abs_shap_values[idx_survivors, variable_index]
    )
    abs_variable_shap_mean = np.mean(
        abs_shap_values[idx_survivors, variable_index]
    )
    variable_shap_mean = np.mean(
        shap_values[idx_survivors, variable_index]
    )

    return {
        "abs_all_shap_sum": abs_all_shap_sum,
        "abs_variable_shap_sum": abs_variable_shap_sum,
        "abs_variable_shap_mean": abs_variable_shap_mean,
        "variable_shap_mean": variable_shap_mean,
        "variable_label": (ending_percentile_value + starting_percentile_value) / 2
    }


def shap_interaction_percentile_distribution_value(
    starting_percentile,
    ending_percentile,
    shap_interaction_values,
    variable_A_index,
    variable_B_index
):
    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_A_index, variable_B_index),
        (variable_B_index, variable_A_index)
    ]
    starting_percentile_value = np.percentile(
        variable_shap_interaction_values, starting_percentile
    )
    ending_percentile_value = np.percentile(
        variable_shap_interaction_values, ending_percentile
    )
    idx_survivors = np.where(
        (variable_shap_interaction_values >= starting_percentile_value)
        & (variable_shap_interaction_values < ending_percentile_value)
    )[0]

    sum_ineraction_shap_values = np.sum(
        shap_interaction_values[idx_survivors], axis=(0, 1, 2)
    )
    top_variable_interaction_shap_values = variable_shap_interaction_values[idx_survivors]

    return 100 * np.sum(top_variable_interaction_shap_values) / sum_ineraction_shap_values


def shap_stats_at_variable_category(
    category,
    variable_index,
    X_variable,
    shap_values,
    abs_shap_values
):
    idx_survivors = np.where(
        X_variable == category
    )[0]

    abs_all_shap_sum = np.sum(abs_shap_values[idx_survivors], axis=(0, 1))
    abs_variable_shap_sum = np.sum(
        abs_shap_values[idx_survivors, variable_index]
    )
    abs_variable_shap_mean = np.mean(
        abs_shap_values[idx_survivors, variable_index]
    )
    variable_shap_mean = np.mean(
        shap_values[idx_survivors, variable_index]
    )

    return {
        "abs_all_shap_sum": abs_all_shap_sum,
        "abs_variable_shap_sum": abs_variable_shap_sum,
        "abs_variable_shap_mean": abs_variable_shap_mean,
        "variable_shap_mean": variable_shap_mean,
        "sample_length_ratio": len(idx_survivors) / len(X_variable)
    }


def plot_shap_categorical_distribution(variable_name, X, shap_values):
    X_variable = X[variable_name]

    if not is_categorical(X_variable):
        raise ValueError(f"The variable {variable_name} is not categorical")


    X_unique = X_variable.unique()
    n_bins = len(X_unique)
    n_ticks = np.arange(n_bins)

    abs_shap_values = np.abs(shap_values)
    abs_shap_relevance = np.zeros(n_bins)
    abs_shap_mean_vs_max = np.zeros(n_bins)
    shap_mean = np.zeros(n_bins)
    variable_index = X.columns.get_loc(variable_name)
    variable_sample_percentage = np.zeros(n_bins)
    abs_variable_shap_max = np.max(np.abs(shap_values[:, variable_index]))


    i = 0
    for category in X_unique:
        result = shap_stats_at_variable_category(
            category,
            variable_index,
            X_variable,
            shap_values,
            abs_shap_values
        ) 
        abs_shap_relevance[i] = 100 * (
            result["abs_variable_shap_sum"] / result["abs_all_shap_sum"]
        )
        abs_shap_mean_vs_max[i] = 100 * (
            result["abs_variable_shap_mean"] / abs_variable_shap_max
        )
        shap_mean[i] = result["variable_shap_mean"]
        variable_sample_percentage[i] = 100 * result["sample_length_ratio"]
        i += 1

    _, ax1 = pl.subplots()
    width = 0.5

    cmap = shap.plots.colors._colors.red_blue

    # Normalización que respeta el signo de los valores SHAP:
    # Valores negativos -> mitad inferior del colormap [0, 0.5]
    # Valor cero -> punto medio del colormap (0.5)
    # Valores positivos -> mitad superior del colormap [0.5, 1.0]
    max_abs_shap = max(abs(np.min(shap_mean)), abs(np.max(shap_mean)))
    if max_abs_shap > 0:
        norm_shap_mean = 0.5 + (shap_mean / (2 * max_abs_shap))
    else:
        norm_shap_mean = np.full_like(shap_mean, 0.5)
    colors = cmap(norm_shap_mean)


    ticks = [str(x) for x in X_unique]
    p = ax1.bar(ticks, abs_shap_relevance, width, color=colors)
    ax1.bar_label(
        p, label_type="center", color="white",
        fontsize=8, fmt="%.1f%%"
    )
    ax1.set_ylabel(f"SHAP Relevance (%)", color="blue")
    ax1.set_xlabel(f"{variable_name} categories")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        ticks, abs_shap_mean_vs_max, color="black",
        label=f"{variable_name} SHAP mean vs max"
    )
    ax2.set_ylabel(f"SHAP Mean (%)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Add a new x axis with labels in variable_label list
    ax3_labels = [f"{str(value)}%" for value in variable_sample_percentage]

    ax3 = ax1.twiny()
    ax3.set_xticks(n_ticks)
    ax3.set_xticklabels(ax3_labels)
    ax3.set_xlabel(f"{variable_name} category sample size (%)", color="black")
    ax3.tick_params(axis="x", labelcolor="black")
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.set_xlim(ax1.get_xlim())

    pl.show()

    return {
        "abs_shap_relevance": abs_shap_relevance,
        "abs_shap_mean_vs_max": abs_shap_mean_vs_max,
        "variable_label": ax3_labels,
        "norm_shap_mean": norm_shap_mean,
    }


def plot_shap_numerical_distribution(variable_name, X, shap_values, n_bins):
    bins = np.arange(n_bins)
    X_variable = X[variable_name]

    abs_shap_values = np.abs(shap_values)
    abs_shap_relevance = np.zeros(n_bins)
    abs_shap_mean_vs_max = np.zeros(n_bins)
    shap_mean = np.zeros(n_bins)
    variable_label = np.zeros(n_bins)
    variable_index = X.columns.get_loc(variable_name)
    abs_variable_shap_max = np.max(np.abs(shap_values[:, variable_index]))

    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins
        result = shap_stats_at_variable_percentile(
            variable_index,
            X_variable,
            shap_values,
            abs_shap_values,
            starting_percentile,
            ending_percentile
        )

        abs_shap_relevance[i] = 100 * (
            result["abs_variable_shap_sum"] / result["abs_all_shap_sum"]
        )
        abs_shap_mean_vs_max[i] = 100 * (
            result["abs_variable_shap_mean"] / abs_variable_shap_max
        )
        shap_mean[i] = result["variable_shap_mean"]
        variable_label[i] = result["variable_label"]

    _, ax1 = pl.subplots()

    cmap = shap.plots.colors._colors.red_blue

    # Normalización que respeta el signo de los valores SHAP:
    # Valores negativos -> mitad inferior del colormap [0, 0.5]
    # Valor cero -> punto medio del colormap (0.5)
    # Valores positivos -> mitad superior del colormap [0.5, 1.0]
    max_abs_shap = max(abs(np.min(shap_mean)), abs(np.max(shap_mean)))
    if max_abs_shap > 0:
        norm_shap_mean = 0.5 + (shap_mean / (2 * max_abs_shap))
    else:
        norm_shap_mean = np.full_like(shap_mean, 0.5)
    colors = cmap(norm_shap_mean)

    bar_width = 100 / n_bins * 0.95  # 95% of bin width, leaving 5% margin
    # Position bars at the center of each bin (percentile range)
    bar_positions = bins * 100 / n_bins + (100 / n_bins) / 2
    ax1.bar(
        bar_positions, abs_shap_relevance, width=bar_width, color=colors,
        label=f"{variable_name} SHAP relevance vs all", align='center'
    )
    ax1.set_xlabel(f"{variable_name} percentile")
    ax1.set_ylabel(f"SHAP Relevance (%)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xlim(0, 100)  # Ensure x-axis shows 0-100 percentile range
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        bar_positions, abs_shap_mean_vs_max, color="black",
        label=f"{variable_name} SHAP mean vs max"
    )
    ax2.set_ylabel(f"SHAP Mean (%)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Determine tick spacing based on number of bins
    tick_spacing = max(1, n_bins // 5) if n_bins <= 20 else 20
    ax3_indices = np.arange(0, n_bins, tick_spacing)
    if ax3_indices[-1] != n_bins - 1:  # Ensure last bin is included
        ax3_indices = np.append(ax3_indices, n_bins - 1)

    ax3_labels = [f"{variable_label[i]:.1f}" for i in ax3_indices]
    ax3_tick_positions = bar_positions[ax3_indices]

    ax3 = ax1.twiny()
    ax3.set_xticks(ax3_tick_positions)
    ax3.set_xticklabels(ax3_labels)
    ax3.set_xlabel(f"{variable_name} value", color="black")
    ax3.tick_params(axis="x", labelcolor="black")
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.set_xlim(ax1.get_xlim())

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines + lines2, labels + labels2,
        loc="upper left", bbox_to_anchor=(0, -0.2)
    )

    pl.show()

    return {
        "abs_shap_relevance": abs_shap_relevance,
        "abs_shap_mean_vs_max": abs_shap_mean_vs_max,
        "variable_label": variable_label,
        "norm_shap_mean": norm_shap_mean,
    }


def plot_shap_distribution(variable_name, X, shap_values, n_bins):
    if is_categorical(X[variable_name]):
        return plot_shap_categorical_distribution(
            variable_name, X, shap_values
        )
    else:
        return plot_shap_numerical_distribution(
            variable_name, X, shap_values, n_bins
        )


def plot_shap_interaction_numerical_numerical_distribution(
    variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
):
    A_variable = X[variable_A_name]
    B_variable = X[variable_B_name]

    if is_categorical(A_variable) or is_categorical(B_variable):
        raise ValueError(
            "Both variables must be numerical to plot the distribution"
        )

    A_max = np.max(A_variable)
    B_max = np.max(B_variable)
    shap_interaction_distribution = np.zeros(n_bins)
    variable_A_distribution = np.zeros(n_bins)
    variable_A_index = X.columns.get_loc(variable_A_name)
    variable_B_distribution = np.zeros(n_bins)
    variable_B_index = X.columns.get_loc(variable_B_name)

    bins = np.arange(n_bins)

    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_A_index, variable_B_index),
        (variable_B_index, variable_A_index)
    ]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins

        shap_interaction_distribution[i] = shap_interaction_percentile_distribution_value(
            starting_percentile,
            ending_percentile,
            shap_interaction_values,
            variable_A_index,
            variable_B_index
        )
        starting_percentile_value = np.percentile(
            variable_shap_interaction_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_interaction_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_interaction_values >= starting_percentile_value)
            & (variable_shap_interaction_values < ending_percentile_value)
        )[0]
        variable_A_mean = A_variable.iloc[idx_survivors].mean()
        variable_A_distribution[i] = 100 * variable_A_mean / A_max
        variable_B_mean = B_variable.iloc[idx_survivors].mean()
        variable_B_distribution[i] = 100 * variable_B_mean / B_max

    fig, ax = pl.subplots()

    ax.bar(
        bins * 100 / n_bins, shap_interaction_distribution
    )
    ax.plot(bins * 100 / n_bins, variable_A_distribution, color="green")
    ax.plot(bins * 100 / n_bins, variable_B_distribution, color="red")

    pl.legend([
        f"{variable_A_name} mean value percentage",
        f"{variable_B_name} mean value percentage",
        "SHAP Interaction percentage in current percentile"
    ])
    pl.xlabel("SHAP Interaction percentile")
    pl.ylabel(f"Percentage")
    pl.title(
        f"{variable_A_name}, {variable_B_name} SHAP Interaction percentage and value distribution"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    pl.show()


def plot_shap_interaction_numerical_categorical_distribution(
    variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
):
    A_variable = X[variable_A_name]
    B_variable = X[variable_B_name]

    if is_categorical(A_variable) and not is_categorical(B_variable):
        variable_N_name = variable_B_name
        variable_C_name = variable_A_name
    elif is_categorical(B_variable) and not is_categorical(A_variable):
        variable_N_name = variable_A_name
        variable_C_name = variable_B_name
    else:
        raise ValueError(
            "One variable must be categorical and the other numerical"
        )

    N_variable = X[variable_N_name]
    variable_N_index = X.columns.get_loc(variable_N_name)
    C_variable = X[variable_C_name]
    variable_C_index = X.columns.get_loc(variable_C_name)

    bins = np.arange(n_bins)
    N_max = np.max(N_variable)
    C_unique = C_variable.unique()

    variable_N_distribution = np.zeros(n_bins)
    shap_interaction_distribution = dict(
        [(str(category), np.zeros(n_bins)) for category in C_unique]
    )

    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_N_index, variable_C_index),
        (variable_C_index, variable_N_index)
    ]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins

        shap_interaction_distribution_value = shap_interaction_percentile_distribution_value(
            starting_percentile,
            ending_percentile,
            shap_interaction_values,
            variable_N_index,
            variable_C_index
        )

        starting_percentile_value = np.percentile(
            variable_shap_interaction_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_interaction_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_interaction_values >= starting_percentile_value)
            & (variable_shap_interaction_values < ending_percentile_value)
        )[0]

        variable_N_mean = N_variable.iloc[idx_survivors].mean()
        variable_N_distribution[i] = 100 * variable_N_mean / N_max

        survivors = C_variable.iloc[idx_survivors]
        all_bin_elements = len(survivors)
        for category in C_unique:
            if all_bin_elements == 0:
                category_distribution_value = 0
            else:
                category_bin_elements = np.sum(survivors == category)
                category_distribution_value = shap_interaction_distribution_value * \
                    (category_bin_elements / all_bin_elements)

            shap_interaction_distribution[
                str(category)
            ][i] = category_distribution_value

    fig, ax = pl.subplots()
    bottom = np.zeros(n_bins)
    width = 0.5

    ax.plot(bins * 100 / n_bins, variable_N_distribution, color="green")

    for category, distribution in shap_interaction_distribution.items():
        p = ax.bar(bins, distribution, width, label=category, bottom=bottom)
        bottom += distribution

    pl.legend([f"Variable {variable_N_name} mean value percentage"] + [
        f"{variable_C_name} = {category}" for category in C_unique
    ])

    pl.xlabel("SHAP Interaction percentile")
    pl.ylabel(f"Percentage")
    pl.title(
        f"{variable_A_name}, {variable_B_name} SHAP Interaction percentage and value distribution"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    pl.show()


def plot_shap_interaction_categorical_categorical_distribution(
    variable_A_name, variable_B_name, X, shap_interaction_values, n_bins, partial_sum=True
):
    A_variable = X[variable_A_name]
    B_variable = X[variable_B_name]

    if not is_categorical(A_variable) or not is_categorical(B_variable):
        raise ValueError(
            "Both variables must be categorical to plot the distribution"
        )

    variable_A_index = X.columns.get_loc(variable_A_name)
    variable_B_index = X.columns.get_loc(variable_B_name)

    bins = np.arange(n_bins)
    A_unique = A_variable.unique()
    B_unique = B_variable.unique()
    T_unique = [
        (category_A, category_B)
        for category_B in B_unique for category_A in A_unique
    ]
    variable_distribution = np.zeros(n_bins)
    shap_interaction_distribution = dict([
        (f"{str(category_A)}-{str(category_B)}", np.zeros(n_bins))
        for (category_A, category_B) in T_unique
    ])

    variable_shap_interaction_values = shap_interaction_values[
        :,
        (variable_A_index, variable_B_index),
        (variable_B_index, variable_A_index)
    ]
    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins

        shap_interaction_distribution_value = shap_interaction_percentile_distribution_value(
            starting_percentile,
            ending_percentile,
            shap_interaction_values,
            variable_A_index,
            variable_B_index
        )

        starting_percentile_value = np.percentile(
            variable_shap_interaction_values, starting_percentile
        )
        ending_percentile_value = np.percentile(
            variable_shap_interaction_values, ending_percentile
        )
        idx_survivors = np.where(
            (variable_shap_interaction_values >= starting_percentile_value)
            & (variable_shap_interaction_values < ending_percentile_value)
        )[0]

        survivors = X[[variable_A_name, variable_B_name]].iloc[idx_survivors]
        all_bin_elements = len(survivors)
        for (category_A, category_B) in T_unique:
            if all_bin_elements == 0:
                category_distribution_value = 0
            else:
                category_bin_elements = np.sum(
                    (survivors[variable_A_name] == category_A)
                    & (survivors[variable_B_name] == category_B)
                )
                category_distribution_value = shap_interaction_distribution_value * \
                    (category_bin_elements / all_bin_elements)

            shap_interaction_distribution[
                f"{str(category_A)}-{str(category_B)}"
            ][i] = category_distribution_value

    fig, ax = pl.subplots()
    bottom = np.zeros(n_bins)
    width = 0.5

    for category, distribution in shap_interaction_distribution.items():
        p = ax.bar(bins, distribution, width, label=category, bottom=bottom)
        bottom += distribution

    pl.legend([
        f"{variable_A_name} = {category_A}, {variable_B_name} = {category_B}, " for (category_A, category_B) in T_unique
    ])

    pl.xlabel("SHAP Interaction percentile")
    pl.ylabel(f"Percentage")
    pl.title(
        f"{variable_A_name}, {variable_B_name} SHAP Interaction percentage and value distribution"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    pl.show()


def plot_shap_interaction_distribution(variable_A_name, variable_B_name, X, shap_interaction_values, n_bins):
    variable_A_is_categorical = is_categorical(X[variable_A_name])
    variable_B_is_categorical = is_categorical(X[variable_B_name])

    if variable_A_is_categorical and variable_B_is_categorical:
        plot_shap_interaction_categorical_categorical_distribution(
            variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
        )
    elif (variable_A_is_categorical and not variable_B_is_categorical) or (not variable_A_is_categorical and variable_B_is_categorical):
        plot_shap_interaction_numerical_categorical_distribution(
            variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
        )
    else:
        plot_shap_interaction_numerical_numerical_distribution(
            variable_A_name, variable_B_name, X, shap_interaction_values, n_bins
        )


def shap_acumulated_importance_value(
    X_variable,
    variable_shap_values,
    starting_percentile,
    ending_percentile
):
    starting_percentile_value = np.percentile(variable_shap_values, starting_percentile)
    ending_percentile_value = np.percentile(variable_shap_values, ending_percentile)
    idx_survivors_percentile = np.where(
        (variable_shap_values >= starting_percentile_value) &
        (variable_shap_values <= ending_percentile_value)
    )[0]
    idx_survivors_accumulated = np.where(
        variable_shap_values <= ending_percentile_value
    )[0]

    percentile_shap_values = variable_shap_values[idx_survivors_accumulated]
    percentile_x_values = X_variable.iloc[idx_survivors_percentile]

    return {
        "percentile_shap_values": percentile_shap_values,
        "percentile_x_values": percentile_x_values,
    }

def plot_shap_lorenz_curve(variable_name, X, shap_values, n_bins):
    bins = np.arange(n_bins)
    X_variable = X[variable_name]

    lorenz_values = np.zeros(n_bins)
    mean_x_values = np.zeros(n_bins)
    variable_index = X.columns.get_loc(variable_name)
    variable_shap_values = shap_values[:, variable_index]
    shap_sum = np.sum(shap_values[:, variable_index])

    for i in bins:
        starting_percentile = i * 100 / n_bins
        ending_percentile = (i + 1) * 100 / n_bins
        result = shap_acumulated_importance_value(
            X_variable,
            variable_shap_values,
            starting_percentile,
            ending_percentile
        )
        lorenz_values[i] = 100 * np.sum(result["percentile_shap_values"]) / shap_sum
        mean_x_values[i] = np.mean(result["percentile_x_values"])

    _, ax = pl.subplots()

    cmap = shap.plots.colors._colors.red_blue

    # Normalización que respeta el signo de los valores SHAP:
    # Valores negativos -> mitad inferior del colormap [0, 0.5]
    # Valor cero -> punto medio del colormap (0.5)
    # Valores positivos -> mitad superior del colormap [0.5, 1.0]
    max_abs_shap = max(abs(np.min(mean_x_values)), abs(np.max(mean_x_values)))
    if max_abs_shap > 0:
        norm_shap_mean = 0.5 + (mean_x_values / (2 * max_abs_shap))
    else:
        norm_shap_mean = np.full_like(mean_x_values, 0.5)
    colors = cmap(norm_shap_mean)

    ax.bar(
        bins * 100 / n_bins, lorenz_values, color=colors
    )
    ax.plot(bins * 100 / n_bins, bins * 100 / n_bins, color="green")

    pl.legend([
        f"Perfect equality",
        f"Accumulated SHAP"
    ])
    pl.xlabel(f"SHAP percentile")
    pl.ylabel(f"Percentage")
    pl.title(f"{variable_name} SHAP Lorenz curve")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    pl.show()