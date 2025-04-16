import argparse

from pathlib import Path
import pandas as pd

import mne

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--derivative_fpath",
        dest="derivative_fpath",
        type=str,
        required=True,
        help="filepath to the directory containing the pylossless, QCR, or pyprep derivative"
    )
    return parser.parse_args()

def main(derivative_fpath):
    derivative_fpath = Path(derivative_fpath).expanduser().resolve()
    dfs = []
    fpaths = list(derivative_fpath.rglob("*_proc-cleaned_raw.fif"))
    for fpath in fpaths:
        if "pyprep" in derivative_fpath.parts:
            raw = read_pyprep_derivative(fpath)
        else:
            raw = mne.io.read_raw_fif(fpath)
        raw.pick("eeg")
        n_bads = len(list(set(raw.info["bads"])))
        n_channels = len(raw.ch_names)
        bad_annots_inds = np.where(
            np.char.startswith(
                raw.annotations.description,
                prefix="BAD_"
            )
        )[0]
        if "pyprep" in derivative_fpath.parts:
            epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)
            epochs.drop_bad(reject={"eeg": 250e-6})
            assert np.isclose(epochs.info["sfreq"], 1000.0)
            dropped = [ep for ep in epochs.drop_log if any([elem.startswith("E") for elem in ep])]
            n_bad_seconds = len(dropped)
            n_seconds = raw.times[-1]
        else:
            bad_durations = raw.annotations.duration[bad_annots_inds]
            n_bad_seconds = (bad_durations).sum()
            n_seconds = raw.times[-1]

        ica_fpath = fpath.with_name(fpath.name.replace("-cleaned_raw.fif", "-ica.fif"))
        if not ica_fpath.exists():
            continue
        ica = mne.preprocessing.read_ica(ica_fpath)
        n_components = ica.n_components_
        n_bad_components = len(ica.exclude)
        df = pd.DataFrame(
            {
                "filename": [raw.filenames[0].name],
                "n_bad_channels": [n_bads],
                "n_channels": [n_channels],
                "n_bad_seconds": [n_bad_seconds],
                "n_seconds": [n_seconds],
                "n_bad_components": [n_bad_components],
                "n_components": [n_components],
            }
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    parts = derivative_fpath.parts
    pipeline = (
        "lossless" if "pylossless" in parts
        else "QCR" if "QCR" in parts
        else "pyprep" if "pyprep" in parts
        else derivative_fpath.name
    )
    df["pipeline"] = pipeline
    out_fpath = Path(__file__).parent / f"{pipeline}_descriptives.csv"
    df.to_csv(out_fpath, index=False)
    return df
    

def read_pyprep_derivative(fpath):
    raw = mne.io.read_raw(fpath)
    # we need to exclude any channels that were interpolated or marked bad by pyprep
    pyprep_sub_dir = raw.filenames[0].parent
    bads_fname = pyprep_sub_dir / "bad_channels.csv"
    bads_orig_fname = pyprep_sub_dir / "bad_channels_original.csv"
    bads_post_interp_fname = pyprep_sub_dir / "bad_channels_after_interpolation.csv"
    
    bads = pd.read_csv(bads_fname)["ch_name"].tolist()
    bads_orig = pd.read_csv(bads_orig_fname)["ch_name"].tolist()
    bads_interp = pd.read_csv(bads_post_interp_fname)["ch_name"].tolist()

    # drop duplicates
    bads_final = list(set(raw.info["bads"] + bads + bads_orig + bads_interp))
    raw.info["bads"] = bads_final
    return raw


if __name__ == "__main__":
    args = parse_args()
    derivative_fpath = args.derivative_fpath
    descriptives_df = main(derivative_fpath=derivative_fpath)