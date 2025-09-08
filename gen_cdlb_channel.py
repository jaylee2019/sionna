import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid


def generate_cdl_b_channel(num_ues=4):
    """Generate a CDL-B channel matrix with shape (4, 64, 768, num_ues).

    Returns a TensorFlow tensor of complex64 values corresponding to
    (rx_ant, tx_ant, subcarrier, UE).
    """
    carrier_freq = 3.5e9

    bs_array = PanelArray(
        num_rows_per_panel=8,
        num_cols_per_panel=4,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="omni",
        carrier_frequency=carrier_freq,
    )

    ut_array = PanelArray(
        num_rows_per_panel=2,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="omni",
        carrier_frequency=carrier_freq,
    )

    rg = ResourceGrid(
        num_ofdm_symbols=1,
        fft_size=768,
        subcarrier_spacing=15e3,
    )

    cdl = CDL(
        "B",
        delay_spread=100e-9,
        carrier_frequency=carrier_freq,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
    )

    ofdm_channel = GenerateOFDMChannel(cdl, rg, normalize_channel=True)
    h_freq = ofdm_channel(batch_size=num_ues)
    h_freq = tf.squeeze(h_freq, axis=[1, 3, 5])  # [UE, rx_ant, tx_ant, subcarrier]
    h_freq = tf.transpose(h_freq, perm=[1, 2, 3, 0])  # (rx_ant, tx_ant, subcarrier, UE)
    return h_freq


if __name__ == "__main__":
    h = generate_cdl_b_channel(num_ues=4)
    print("Channel shape:", h.shape)
    np.save("cdl_b_channel.npy", h.numpy())
