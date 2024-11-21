# ---------------------------------------------------------------- #

import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #

def plot(
        gammas,
        get_pv_DICE, get_pv_OnPE_e, get_pv_OnPE_b,
        one_minus_gamma=False, include_undiscounted=False):

    pvs_DICE = []
    pvs_OnPE_e = []
    pvs_OnPE_b = []

    for gamma in gammas:

        pv_DICE = get_pv_DICE(gamma)
        pv_OnPE_e = get_pv_OnPE_e(gamma)
        pv_OnPE_b = get_pv_OnPE_b(gamma)

        pvs_DICE.append( float(pv_DICE) )
        pvs_OnPE_e.append( float(pv_OnPE_e) )
        pvs_OnPE_b.append( float(pv_OnPE_b) )

    _, ax = plt.subplots(ncols=2, figsize=(15, 5))

    x = gammas if not one_minus_gamma else 1 - gammas

    y_1 = pvs_DICE
    y_2 = pvs_OnPE_e
    y_3 = pvs_OnPE_b

    ax[0].set_title("PVs")
    ax[0].plot(x, y_1, label="DICE",            color="blue",   marker=".")
    ax[0].plot(x, y_2, label="OnPE evaluation", color="orange", marker="+")
    ax[0].plot(x, y_3, label="OnPE behavior",   color="green",  marker="x")
    ax[0].set_ylabel("policy value")
    ax[0].legend()
    ax[0].grid(linestyle=":")

    y = [ abs(pv_1 - pv_2) for pv_1, pv_2 in zip(pvs_DICE, pvs_OnPE_e) ]

    ax[1].set_title("PV Errors")
    ax[1].plot(x, y)
    ax[1].set_ylabel("error")
    ax[1].set_yscale("log")
    ax[1].set_ylim([1e-3, 1e1])
    ax[1].grid(linestyle=":")

    if one_minus_gamma:
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
        ax[0].invert_xaxis()
        ax[1].invert_xaxis()

    if include_undiscounted:
        pv_DICE = get_pv_DICE(1.0)
        pv_OnPE_e = get_pv_OnPE_e(1.0)

        ax[0].axhline(y=pv_DICE,   color="blue",   marker=".", linestyle=":")
        ax[0].axhline(y=pv_OnPE_e, color="orange", marker="+", linestyle=":")

        ax[1].axhline(y=abs(pv_DICE - pv_OnPE_e), linestyle=":")

    plt.xlabel("gamma" if not one_minus_gamma else "1 - gamma")

    plt.show()

# ---------------------------------------------------------------- #
