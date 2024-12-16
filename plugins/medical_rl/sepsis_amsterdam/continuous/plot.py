# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.latex import latex_gamma, latex_lambda

from dice_rl_TU_Vienna.plot.continuous import get_plot_logs as get_plot_logs_general

# ---------------------------------------------------------------- #

def get_plot_logs(
        get_behavior_policy_value,
        #
        outputs_dir,
        hparam_str_policy, hparam_str_dataset,
        estimator_name, hparam_str_evaluation,
        #
        title=None,
        xlim=None,
        ylim_1=None, ylim_2=None, ylim_3=None,
        n_ma_1=None, n_ma_2=None, n_ma_3=None,
        #
        save_dir=None, file_name=None,
    ):

    def get_suptitle(gammas):
        return f"Medical Continuous"

    def get_pv_baselines(gamma):
        pv = get_behavior_policy_value(gamma)

        return [
            {
                "label": "OnPE original",
                "value": pv,
                "linestyle": "dotted",
            },
        ]

    error_tags = None
    plot_types = None

    hparams_title = [
        "gamma",
        "batch-size",
        "hidden-dimensions",
        "mlp-regularizer",
    ]

    return get_plot_logs_general(
        get_suptitle, get_pv_baselines,
        #
        outputs_dir,
        hparam_str_policy, hparam_str_dataset,
        estimator_name, hparam_str_evaluation,
        #
        error_tags, plot_types,
        #
        title,
        xlim,
        ylim_1, ylim_2, ylim_3,
        n_ma_1, n_ma_2, n_ma_3,
        #
        save_dir, file_name,
        hparams_title,
    )

# ---------------------------------------------------------------- #
