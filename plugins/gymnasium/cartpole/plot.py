# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.plot.continuous import get_plot_logs as get_plot_logs_general

# ---------------------------------------------------------------- #

def get_plot_logs(
        get_policy_value,
        #
        outputs_dir,
        hparam_str_policy, hparam_str_dataset,
        estimator_name, hparam_str_evaluation,
        #
        ylim_1=None, ylim_2=None, ylim_3=None,
        n_ma_1=None, n_ma_2=None, n_ma_3=None,
        #
        save_dir=None, file_name=None,
        hparams_title=None,
    ):

    def get_suptitle(gammas):
        return "Cartpole"

    def get_pv_baselines(gamma):
        pv_b = get_policy_value["b"](gamma)
        pv_e = get_policy_value["e"](gamma)

        return [
            {
                "label": "OnPE behavior",
                "value": pv_b,
                "linestyle": "dashed",
            },
            {
                "label": "OnPE evaluation",
                "value": pv_e,
                "linestyle": "dotted",
            },
        ]

    error_tags = []
    plot_types = None

    return get_plot_logs_general(
        get_suptitle, get_pv_baselines,
        #
        outputs_dir,
        hparam_str_policy, hparam_str_dataset,
        estimator_name, hparam_str_evaluation,
        #
        error_tags, plot_types,
        #
        ylim_1, ylim_2, ylim_3,
        n_ma_1, n_ma_2, n_ma_3,
        #
        save_dir, file_name,
        hparams_title,
    )

# ---------------------------------------------------------------- #
