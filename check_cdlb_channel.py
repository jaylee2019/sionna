import numpy as np
import tensorflow as tf
from sionna.phy.channel.tr38901 import CDL, PanelArray
from gen_cdlb_channel import generate_cdl_b_channel

# Reference values from 3GPP TR 38.901 for CDL-B
CDL_POWERS_B_DB = np.array([
    0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0,
    -8.9, -9.0, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4,
    -14.9, -9.2, -11.3
])

CDL_DELAYS_B = np.array([
    0.0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055,
    0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842,
    2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834
])

DELAY_SPREAD = 100e-9  # s
CARRIER_FREQUENCY = 3.5e9  # Hz

MAX_ERR_REL = 3e-2
MAX_ERR_ABS = 1e-4


def check_cdlb_compliance(batch_size=10000):
    """Return True if CDL-B statistics match 3GPP reference."""
    tx = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQUENCY,
        precision="double",
    )
    rx = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQUENCY,
        precision="double",
    )

    cdl = CDL(
        "B",
        delay_spread=DELAY_SPREAD,
        carrier_frequency=CARRIER_FREQUENCY,
        ut_array=rx,
        bs_array=tx,
        direction="downlink",
        precision="double",
    )
    a, tau = cdl(batch_size, 1, 100e6)
    a = a[:, 0, 0, 0, 0, :, 0].numpy()
    tau = tau.numpy()[0, 0, 0]

    p = np.mean(np.abs(a) ** 2, axis=0)
    ref_p = 10 ** (CDL_POWERS_B_DB / 10.0)
    ref_p = ref_p / np.sum(ref_p)
    order = np.argsort(CDL_DELAYS_B)
    ref_p = ref_p[order]

    delays = tau / DELAY_SPREAD
    ref_delays = np.sort(CDL_DELAYS_B)

    max_err_p = np.max(np.abs(ref_p - p) / ref_p)
    max_err_tau = np.max(np.abs(ref_delays - delays))

    return max_err_p <= MAX_ERR_REL and max_err_tau <= MAX_ERR_ABS, max_err_p, max_err_tau


def main():
    h = generate_cdl_b_channel(num_ues=4)
    print("Channel shape:", h.shape)
    avg_power = tf.reduce_mean(tf.abs(h) ** 2).numpy()
    print(f"Average normalized power: {avg_power:.4f}")

    ok, err_p, err_tau = check_cdlb_compliance()
    print(f"Max relative power error: {err_p:.2e}")
    print(f"Max absolute delay error: {err_tau:.2e}")
    if ok and np.isclose(avg_power, 1.0, rtol=1e-2):
        print("CDL-B channel complies with 3GPP reference within tolerance.")
    else:
        print("CDL-B channel does NOT meet 3GPP reference.")


if __name__ == "__main__":
    main()

