"""
The idea is roughly:

1. Load the Grand averaged evoked
2. get the data into numpy
3. average across channels
4. get the amplitude of the maximum negative peak e.g. N1 ("Signal")
5. Get the sample Stdev across the baseline period ("Noise")
6. SNR = signal / noise
"""

import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns


def main():
    palette = sns.color_palette()
    sns.set_style("darkgrid")
    assets_fpath = Path(__file__).parent.parent / "assets" / "insar_2025"
    
    evoked_fpaths = assets_fpath.glob("*_ave.fif")
    derivative = []
    pos_amps = []
    pos_inds = []
    neg_amps = []
    neg_inds = []
    max_amps = []
    max_inds = []
    stdevs = []
    snrs = []
    datas = np.zeros(shape=(4, 951))
    for ii, ev_fpath in enumerate(evoked_fpaths):
        evoked: list[mne.Evoked] = mne.read_evokeds(ev_fpath)
        assert len(evoked) == 1
        evoked: mne.Evoked = evoked[0]
        evoked.set_eeg_reference(["VREF"])
        data = evoked.get_data().mean(0).squeeze() # average across channels
        # Find Peaks
        peak_ind_pos, peak_amp_pos = mne.preprocessing.peak_finder(data, extrema=1)
        max_amp_idx = np.argmax(peak_amp_pos)
        peak_ind_pos, peak_amp_pos = peak_ind_pos[max_amp_idx], peak_amp_pos[max_amp_idx]     

        peak_ind_neg, peak_amp_neg = mne.preprocessing.peak_finder(data, extrema=-1)
        max_amp_idx = np.argmin(peak_amp_neg)
        peak_ind_neg, peak_amp_neg = peak_ind_neg[max_amp_idx], peak_amp_neg[max_amp_idx]

        # Get absolute maximum peak
        amps = [peak_amp_pos, peak_amp_neg]
        abs_amps = [abs(peak_amp_pos), abs(peak_amp_neg)]
        inds = [peak_ind_pos, peak_ind_neg]
        max_ind = np.argmax(abs_amps)
        max_amp, max_ind = amps[max_ind], inds[max_ind]

        assert isinstance(peak_amp_pos, float)
        assert isinstance(peak_amp_neg, float)
        assert isinstance(max_amp, float)
        assert isinstance(peak_ind_pos, np.int64)
        assert isinstance(peak_ind_neg, np.int64)
        assert isinstance(max_ind, np.int64)

        derivative.append(ev_fpath.stem)
        pos_amps.append(peak_amp_pos)
        neg_amps.append(peak_amp_neg)
        pos_inds.append(peak_ind_pos)
        neg_inds.append(peak_ind_neg)
        max_amps.append(max_amp)
        max_inds.append(max_ind)
        # Calculate Noise
        noise = data[:200].std(ddof=1)
        stdevs.append(noise)
        # Calculate SNR
        snr = abs(peak_amp_neg) / noise
        snrs.append(snr)
        datas[ii, :] = data

    assert (
        len(pos_amps) ==
        len(neg_amps) ==
        len(pos_inds) ==
        len(neg_inds) ==
        len(max_amps) ==
        len(max_inds) ==
        len(stdevs) ==
        len(snrs) == 4
    )
    df = pd.DataFrame(
        {
            "derivative": derivative,
            "positive_peak": pos_amps,
            "positive_indice": pos_inds,
            "negative_peak": neg_amps,
            "negative_indice": neg_inds,
            "absolute_maximum_peak": max_amps,
            "absolute_maximum_ind": max_inds,
            "noise": stdevs,
            "snr": snrs,
            }
    )
    df.to_csv(assets_fpath / "snr.csv", index=False)

    fig, ax = plt.subplots(constrained_layout=True)
    for dat, deriv, pos, posii, neg, negii in zip(
        datas, derivative, pos_amps, pos_inds, neg_amps, neg_inds
        ):
        ax.plot(dat, label=deriv)
        # ax.scatter(posii, pos, color=palette[-2])
        ax.scatter(negii, neg, color=palette[-1])
    ax.legend()
    fig.savefig(assets_fpath / "snr.png")

    return df

if __name__ == "__main__":
    df = main()