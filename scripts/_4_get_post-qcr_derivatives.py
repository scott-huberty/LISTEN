import argparse

from pathlib import Path

import mne

import numpy as np

import pandas as pd


def main(derivative):
    WANT_EVENTS = ["Tone/Standard", "Tone/Deviant"]
    dpath = Path(__file__).parent.parent / "derivatives"
    dpath = dpath / derivative / "run-01"
    assert dpath.exists()
    dpaths = list(dpath.rglob("*_proc-cleaned_raw.fif"))
    assert dpaths

    tmin = -.2
    tmax = .75
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

        freqs = np.arange(4, 51, 1)
        n_cycles = freqs / 2
        tfr_method = "morlet"
        epochs = mne.Epochs(
            raw,
            tmin=tmin,
            tmax=tmax,
            event_id=WANT_EVENTS,
            )
        evokeds: list[mne.Evoked] = epochs.average(by_event_type=True)
        spectrum = epochs.compute_psd(fmin=freqs[0], fmax=freqs[-1], method="welch")
        tfr_dict, itc_dict = dict(), dict()
        for ev in WANT_EVENTS:
            tfr, itc = epochs[WANT_EVENTS].compute_tfr(
                method=tfr_method, average=True, freqs=freqs, n_cycles=n_cycles, return_itc=True
                )
            tfr_dict[ev] = tfr
            itc_dict[ev] = itc
        
        derivative_dir = dpath.parent.parent.parent.parent.resolve()
        assert derivative_dir.name == "derivatives"
        this_derivative_dir = derivative_dir / f"{derivative}_features"
        this_derivative_dir.mkdir(exist_ok=True, parents=False)
        sub_dir = this_derivative_dir / dpath.stem.split("_proc")[0]
        sub_dir.mkdir(exist_ok=True, parents=False)
        out_name = dpath.stem.replace("_raw", "")

        epochs.save(sub_dir / f"{out_name}-epo.fif", overwrite=True)
        mne.write_evokeds(sub_dir / f"{out_name}-ave.fif", evokeds, overwrite=True)
        spectrum.event_id = {key.split("/")[1]: val for key, val in spectrum.event_id.items()}
        spectrum.save(sub_dir / f"{out_name}-psd.hdf5", overwrite=True)
        for event_id in WANT_EVENTS:
            # HD5 can't handle keys with a slash in them so change Tone/Standard -> Standard
            event = event_id.split("/")[1]
            tfr = tfr_dict[event_id]
            itc = tfr_dict[event_id]
            tfr.save(sub_dir / f"{out_name}_{event}-tfr.hdf5", overwrite=True)
            itc.save(sub_dir / f"{out_name}_{event}-itc.hdf5", overwrite=True)
    return


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
    main(derivative)

        
        