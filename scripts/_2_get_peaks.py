import argparse

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
sns.set_style("darkgrid")


def parse_args():
    parser = argparse.ArgumentParser(description="This script will load each evoked file from derivatives/evoked, average across channels, and compute the most negative trough.")
    parser.add_argument(
        "--derivative",
        dest="derivative",
        type=str,
        required=True,
        choices=["raw", "pylossless", "QCR", "pyprep"],
        help="Which processed derivative of the data to use. The respective evoked will be read from derivatives/evoked"
    )
    return parser.parse_args()


def main(derivative):
    dpath = Path(__file__).parent.parent / "derivatives" / "evoked" / "run-01"
    assert dpath.exists()

    palette = sns.color_palette()
    df = pd.DataFrame({"file": [], "trough": [], "trough_indice": [], "baseline_noise": []})
    dpaths = dpath.rglob(f"*_proc-{derivative}_task-phonemes_ave.fif")
    for dpath in dpaths:
        evoked: list[mne.Evoked] = mne.read_evokeds(dpath)
        assert len(evoked) == 1
        evoked = evoked[0]
        evoked.set_eeg_reference(["VREF"])
        baseline = abs(evoked.baseline[1] - evoked.baseline[0]) * evoked.info["sfreq"]
        assert int(baseline) == 200, baseline
        baseline = int(baseline)
        data = evoked.get_data().mean(0).squeeze() # average across channels
        inds, troughs = mne.preprocessing.peak_finder(
            data[baseline:], extrema=-1
            )
        min_idx = np.argmin(troughs)
        trough, ind = troughs[min_idx], inds[min_idx]

        # Compute baseline noise
        assert baseline == 200
        noise = np.std(data[:baseline], ddof=1)

        new_row = pd.DataFrame(
            {"file": [dpath.name], "trough": [trough], "trough_indice": [ind], "baseline_noise": [noise]}
        )
        df = pd.concat(
            [df, new_row], ignore_index=True
        )

        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(data)
        ax.scatter(x=ind+baseline, y=trough, color=palette[3])
        out_dir = dpath.parent.parent.parent.parent / "peaks" / "run-01"
        assert out_dir.exists()
        # Lets make subfolders just for the figures since there are so many
        out_dir_deriv = out_dir / derivative
        out_dir_deriv.mkdir(exist_ok=True, parents=False)
        fig.savefig(out_dir_deriv / f"{dpath.stem}.png")

    df["snr"] = df["trough"].abs() / df["baseline_noise"]
    df.to_csv(out_dir / f"n1_troughs_{derivative}.csv", index=False)
    return df

if __name__ == "__main__":
    args = parse_args()
    derivative = args.derivative
    main(derivative)