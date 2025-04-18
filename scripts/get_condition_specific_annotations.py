import argparse

from glob import glob
from pathlib import Path

import mne
import mffpy
import numpy as np 



def parse_args():
    parser = argparse.ArgumentParser("This script will take a derivative fif file, the original MFF file, identify the conditions (standard, deviant) for each stm+ annotation in the derivative file")
    parser.add_argument(
        "--derivative_fpath",
        dest="derivative_fpath",
        type=str,
        required=True,
        help="Path to the derivaitve, e.g. ~/path/to/listen/derivatives/QCR/run-01"
    )
    parser.add_argument(
        "--sourcedata_fpath",
        dest="sourcedata_fpath",
        type=str,
        required=True,
        help="Path to the directory containing the MFF files."
    )
    return parser.parse_args()


def main(*, derivative_fpath, sourcedata_fpath):    
    
    derivative_fpath = Path(derivative_fpath)
    sourcedata_fpath = Path(sourcedata_fpath)

    mff_fpaths = glob(f"{sourcedata_fpath}/2???/**/*.mff", recursive=True)
    dpaths = derivative_fpath.rglob("*_proc-cleaned_raw.fif")

    for dpath in dpaths:
        dpath_name = dpath.stem.replace("_proc-cleaned_raw", "")
        if dpath_name in ["phonemes_2054_20240620_112330", "phonemes_2054_20240620_112330"]:
            continue
            # unfortunately in this file I cant line up the mffpy events and the raw annotations
        mff_fnames = [
            Path(mff_path).stem.replace(" ", "_")
            for mff_path
            in mff_fpaths
        ]
        mff_fpath = [
            mff_path
            for mff_path
            in mff_fpaths
            if Path(mff_path).stem.replace(" ", "_") == dpath_name
            ]
        assert len(mff_fpath) == 1
        mff_fpath = Path(mff_fpath[0])
        
        events_xmls = list(mff_fpath.glob("Events_ECI*.xml"))

        events_xmls = [fname.name for fname in events_xmls]
        assert len(events_xmls) == 1
        event_file = events_xmls[0]

        categories_dict = {}
        categories = mffpy.XML.from_file(mff_fpath / event_file)
        categories_dict[event_file] = categories.get_content()["event"]

        events_eci = [k for k in categories_dict.keys() if "ECI" in k]
        assert len(events_eci) == 1
        events_eci = categories_dict[events_eci[0]]
        condition_mapping = {1: "Standard", 2: "Deviant"}

        raw = mne.io.read_raw_fif(dpath)
        bad_annots = np.char.startswith(raw.annotations.description, "BAD_")
        bad_annots_idx = np.where(bad_annots)[0]
        raw.annotations.delete(bad_annots_idx)
        din_annots_idx = np.where(raw.annotations.description == "DIN8")[0]
        if len(din_annots_idx):
            raw.annotations.delete(din_annots_idx)
        assert len(raw.annotations) == len(events_eci)

        annots_to_rename = {"Standard": [], "Deviant": []}
        for ii, (annot, event) in enumerate(zip(raw.annotations, events_eci)):
            if annot["description"] == "stm+":
                assert event["code"] == "stm+"
                condition_number = int(event["keys"]["cel#"])
                condition = condition_mapping[condition_number]
                annots_to_rename[condition].append(ii)
        for key in annots_to_rename:
            raw.annotations.description[annots_to_rename[key]] = f"Tone/{key}"
        df = raw.annotations.to_data_frame(time_format=None)
        out_dir = raw.filenames[0].parent
        out_fpath = out_dir / "events.csv"
        df.to_csv(out_fpath, index=False) 



def get_cel_map(events_eci):
    """Return a dictionary mapping from CEL codes to human readable conditions.

    For example for the semantics task, this should return a dictionary like:
    {"1": "match", "2": "mismatch"}. For the phonemes task, this should return
    {"1": "Standard", "2": "Deviant"}.
    """
    at_cell_events = False
    cel_map = {}
    for event in events_eci:
        if at_cell_events and event["code"] != "CELL":
            return cel_map
        if event["code"] == "CELL":
            at_cell_events = True
        if at_cell_events:
            cel_map[int(event["keys"]["cel#"])] = event["label"]
    return cel_map


if __name__ == "__main__":
    args = parse_args()
    derivative_fpath = args.derivative_fpath
    sourcedata_fpath = args.sourcedata_fpath
    main(derivative_fpath=derivative_fpath, sourcedata_fpath=sourcedata_fpath)