import argparse

from pathlib import Path

import mne

import numpy as np

from scipy.stats import zscore

def main(derivative):
    # Since we dont have a "raw" derivative, we'll just load pylosslesse's and then ignore bad chs/annots
    deriv = "pylossless" if derivative == "raw" else derivative
    dpath = Path(__file__).parent.parent / "derivatives"
    dpath = dpath / deriv / "run-01"
    assert dpath.exists()
    dpaths = list(dpath.rglob("*_proc-cleaned_raw.fif"))
    assert dpaths
    assets_fpath = Path(__file__).parent.parent / "assets" / "insar_2025"

    # Let's Pre-allocate a block of memory to be efficient
    want_shape = (len(dpaths), 129, 951) # n_participants, n_channels, n_times
    tmin = -.2
    tmax = .75
    evoked_data = np.zeros(shape=want_shape)
    for ii, dpath in enumerate(dpaths):
        raw = mne.io.read_raw(dpath, preload=True)
        # If we just want the raw data, then get the data without rejecting annotations or channels
        epochs_kwargs = dict(tmin=tmin, tmax=tmax, event_id="stm+")
        if derivative == "raw":
            epochs_kwargs.update(reject_by_annotation=False)
        elif derivative in ["pyprep", "pylossless", "QCR"]:
            # clean the raw data
            ica_fpath = dpath.parent / dpath.name.replace("-cleaned_raw", "-ica")
            if not ica_fpath.exists():
                continue
            ica = mne.preprocessing.read_ica(ica_fpath)
            ica.apply(raw)
            raw.interpolate_bads()
        else:
            raise RuntimeError(f"Got unknown derivative: {derivative}.")

        # Now make the evoked object
        evoked = mne.Epochs(raw, **epochs_kwargs).average()
        save_evoked(evoked, dpath, derivative)

        del raw
        evoked_data[ii, :, :] = evoked.get_data(picks="eeg")
 
    # Average across participants
    evoked_data = evoked_data.mean(0)
    evoked_grand_avg = mne.EvokedArray(evoked_data, evoked.info, tmin)
    
    # Finally, for plotting purposes, exclude any large amplitude channels from plot
    bad_chs = get_outlier_chs(evoked_grand_avg)
    
    evoked_grand_avg.drop_channels(bad_chs)
    
    fig = evoked_grand_avg.plot()
    basename = f"grand_avg_evoked_{derivative}_ave"
    fig.savefig(assets_fpath / f"{basename}.png", dpi=300)
    evoked.save(assets_fpath / f"{basename}.fif", overwrite=True)


def save_evoked(evoked, dpath, derivative):
    basename = dpath.stem.replace("_proc-cleaned_raw", "")
    fname = dpath.name
    evoked_dir = dpath.parent.parent.parent.parent / "evoked" / "run-01"
    assert evoked_dir.exists()
    out_dir = evoked_dir / basename
    out_dir.mkdir(exist_ok=True, parents=False)
    out_fname = fname.replace("_raw.fif", "_task-phonemes_ave.fif")
    out_fname = out_fname.replace("proc-cleaned", f"proc-{derivative}")
    out_fpath = out_dir / out_fname
    evoked.save(out_fpath, overwrite=True)
    return out_fpath

def get_outlier_chs(evoked):
    zscores = zscore(
        np.abs(
            evoked.get_data().mean(1)
        )
    )
    outlier_idxs = np.where(zscores > 3)[0]
    bad_chs = [evoked.ch_names[ii] for ii in outlier_idxs]
    return bad_chs


def parse_args():
    parser = argparse.ArgumentParser(description="This script makes Grand Averaged Evoked figures for LISTEN data.")
    parser.add_argument(
        "--derivative",
        dest="derivative",
        required=True,
        type=str,
        choices=["raw", "pylossless", "QCR", "pyprep"],
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    derivative = args.derivative
    main(derivative)

        
        