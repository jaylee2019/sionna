# Plan to generate CDL-B channel matrix

1. **Define antenna arrays**
   - Create a `PanelArray` for the base station (BS) with 64 transmit antennas using 8×4 dual-polarized (cross) elements.
   - Create a `PanelArray` for the user terminals (UEs) with 4 receive antennas using 2×1 dual-polarized (cross) elements.

2. **Configure resource grid**
   - Instantiate a `ResourceGrid` with one OFDM symbol, FFT size 768, and desired subcarrier spacing (e.g., 15 kHz).

3. **Instantiate CDL-B channel model**
   - Use Sionna's `CDL` class with model "B", setting parameters such as delay spread, carrier frequency, arrays, and downlink direction.

4. **Generate frequency-domain channel**
   - Use `GenerateOFDMChannel` with the CDL-B model and resource grid.
   - Sample the channel with a batch size equal to the number of UEs (4).

5. **Reshape to desired format**
   - Squeeze singleton dimensions and transpose to obtain a tensor of shape `(4, 64, 768, 4)` corresponding to `(rx_ant, tx_ant, subcarrier, UE)`.

6. **Store or process the result**
   - Optionally save the complex tensor to a `.npy` file for later use.

7. **Verify**
   - Run the script to confirm the generated tensor has the expected shape and properties.

8. **Add compliance check**
   - Implement `check_cdlb_channel.py` to statistically compare tap powers and delays against 3GPP CDL-B reference values.
   - Report the maximum deviation and flag if tolerances are exceeded.

