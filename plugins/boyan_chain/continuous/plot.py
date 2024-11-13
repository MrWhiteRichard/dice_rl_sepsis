# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.plot.continuous import get_plot_logs as get_plot_logs_general

# ---------------------------------------------------------------- #

def get_plot_logs(
        analytical_solver,
        #
        outputs_dir,
        hparam_str_policy, hparam_str_dataset,
        estimator_name, hparam_str_evaluation,
        #
        ylim_1=None, ylim_2=None, ylim_3=None,
        n_ma_1=None, n_ma_2=None, n_ma_3=None,
        #
        save_dir=None, file_name=None,
    ):

    def get_suptitle(gammas):
        if 1 in gammas:
            assert len( set(gammas) ) == 1
            kind = "continuing"
        else:
            kind = "episodic"

        return f"Boyan Chain Continuous - {kind}"

    def get_pv_baselines(gamma):
        pv, *_ = analytical_solver.solve(gamma, primal_dual="dual")

        return [
            {
                "label": "analytical",
                "value": pv,
                "linestyle": "dotted",
            },
        ]

    error_tags = ["pv_error", "sdc_L2_error", "norm_error"]
    plot_types = ["plot", "semilogy", "plot"]

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
    )


# ---------------------------------------------------------------- #
