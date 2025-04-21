import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import mne

import numpy as np

import pandas as pd

import seaborn as sns

def main(derivative):
    WANT_EVENTS = ["Tone/Standard", "Tone/Deviant"]
    ROI = [
        "E7",
        "E106",
        "E80",
        "E55",
        "E31",
        "E30",
        "E13",
        "E6",
        "E112",
        "E111",
        "E105",
        "E87",
        "E79",
        "E54",
        "E37",
        ]
    dpath = Path(__file__).parent.parent / "derivatives"
    dpath = dpath / derivative / "run-01"
    assert dpath.exists()
    dpaths = list(dpath.rglob("*_proc-cleaned_raw.fif"))
    assert dpaths
    assets_fpath = Path(__file__).parent.parent / "assets" / "insar_2025"

    tmin = -.2
    tmax = .75
    dfs = {ev: [] for ev in WANT_EVENTS}
    for ii, dpath in enumerate(dpaths):
        dpath = Path(dpath)
        raw = mne.io.read_raw(dpath, preload=True)
        events_fname = dpath.parent / "events.csv"
        if not events_fname.exists():
            continue
        df = pd.read_csv(events_fname)

        # Set the condition specific annotations + bad annotations
        bad_annots = raw.annotations[
            np.where(
                np.char.startswith(
                    raw.annotations.description, "BAD_"
                )
            )[0]
        ].copy()
        event_annots = mne.Annotations(
            onset=df["onset"],
            duration=df["duration"],
            description=df["description"],
            orig_time=raw.info["meas_date"],
        )
        raw.set_annotations(bad_annots + event_annots)
       
        # Now clean the raw data
        ica_fpath = dpath.parent / dpath.name.replace("-cleaned_raw", "-ica")
        if not ica_fpath.exists():
            continue
        ica = mne.preprocessing.read_ica(ica_fpath)
        ica.apply(raw)
        raw.interpolate_bads()

        for jj, event in enumerate(WANT_EVENTS):
            evoked = mne.Epochs(
                raw,
                tmin=tmin,
                tmax=tmax,
                event_id=event,
                ).average() # .pick(ROI)
            df_ = evoked.to_data_frame().set_index("time").mean(1).to_frame(name="voltage")  # channels
            df_["file"] = dpath.name
            dfs[event].append(df_)
    
    # Average across participants
    assert 1 == 0
    df_standard = pd.concat(
        dfs["Tone/Standard"],
        axis=0
    )
    df_deviant = pd.concat(
        dfs["Tone/Deviant"],
        axis=0
        )
    del dfs
    df_standard["condition"] = "Standard"
    df_deviant["condition"] = "Deviant"
    df_full = pd.concat([df_standard, df_deviant], axis=0)


    sns.set_style("darkgrid")
    fig, ax = plt.subplots(constrained_layout=True)
    sns.lineplot(df_full, x="time", y="voltage", hue="condition", ax=ax)
    
    ax.set_xlim(-.2, .75)
    ax.legend()

    fig.savefig(assets_fpath / f"grand_avg_evoked_by_condition_{derivative}.png", dpi=300)
    df_full.reset_index().to_csv(assets_fpath / f"grand_avg_evoked_by_condition_{derivative}.csv", index=False)
    return df_full


def parse_args():
    parser = argparse.ArgumentParser(description="This script makes Grand Averaged Evoked figures for LISTEN data.")
    parser.add_argument(
        "--derivative",
        dest="derivative",
        required=True,
        type=str,
        choices=["pylossless", "QCR", "pyprep"],
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    derivative = args.derivative
    df = main(derivative)

        
        