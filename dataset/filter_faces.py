"""Face SFT data, step 1: pick single-person candidates from OpenVid-1M metadata.

Same input as filter_openvid.py (the parquet from fetch_openvid_meta.py). Captions only give
candidates (one real person is the subject); encode_faces.py then runs a face detector and keeps
only clips with exactly one face that is large enough (--min_face) and stays in the crop, so
"close-up" does not have to be in the caption (--closeup_only makes it required).

    python dataset/filter_faces.py --meta data/openvid_meta --out data/faces.parquet
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.filter_openvid import head, parse_path

# with --closeup_only the caption head must also say it is a close-up / portrait
CLOSEUP = re.compile(r"close[- ]?up|closeup|portrait|headshot|\bface\b|facial", re.I)
PERSON = re.compile(r"\b(?:man|woman|girl|boy|lady|guy|person|child|kid|teen(?:ager)?|"
                    r"gentleman|actress|actor|model|businessman|businesswoman|baby|toddler|student|"
                    r"doctor|nurse|chef|worker|singer|dancer|athlete|player|bride|groom|father|mother|"
                    r"daughter|son|grandmother|grandfather|influencer|vlogger|speaker|presenter)\b", re.I)
# more than one person, or not real footage
MULTI = re.compile(r"\b(?:two|three|several|group|couple|crowd|people|men|women|friends|family|"
                   r"children|kids|they|their faces|each other)\b", re.I)
NOT_REAL = re.compile(r"\b(?:animat\w+|cartoon|cgi|render\w*|3d|digital art|illustration|painting|"
                      r"drawing|mannequin|statue|sculpture|doll|mask|text|logo|watermark|subtitles?)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/openvid_meta")
    ap.add_argument("--out", default="data/faces.parquet")
    ap.add_argument("--min_seconds", type=float, default=2.5, help="same as filter_openvid.py")
    ap.add_argument("--all_sources", action="store_true",
                    help="also keep non-stock sources (default: pixabay / pexels / mixkit only)")
    ap.add_argument("--closeup_only", action="store_true", help="caption must say close-up / portrait / face")
    ap.add_argument("--max_clips_per_video", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(args.meta)
    funnel = [("all", len(df))]
    parsed = df.video_path.map(parse_path)
    df["source"], df["video_id"] = parsed.str[0], parsed.str[1]
    if not args.all_sources:
        df = df[df.source.isin({"pixabay", "pexels", "mixkit"})]
        funnel.append(("stock sources", len(df)))

    df["head"] = df.caption.map(head)
    df = df[df["head"].str.contains(PERSON)]
    funnel.append(("person in caption", len(df)))
    if args.closeup_only:
        df = df[df["head"].str.contains(CLOSEUP)]
        funnel.append(("close-up / portrait", len(df)))
    df = df[~df["head"].str.contains(MULTI) & ~df["head"].str.contains(NOT_REAL)]
    funnel.append(("single, real person", len(df)))

    df = df[df.seconds >= args.min_seconds]
    funnel.append((f"seconds>={args.min_seconds}", len(df)))
    # drop near-frozen, shaky and cut clips; a moving camera is left to the detector's drift check
    df = df[(df.temporal_consistency_score >= 0.97) & (df.motion_score.between(0.2, 30))]
    funnel.append(("temporal/motion ok", len(df)))

    df = (df.sort_values("aesthetic_score", ascending=False)
            .groupby("video_id").head(args.max_clips_per_video))
    funnel.append((f"<= {args.max_clips_per_video} clips/video", len(df)))

    print("=== funnel ===")
    for name, n in funnel:
        print(f"{name:28s}{n:>9,d}")
    print("\n=== sources ===")
    print(df.source.value_counts().head(15).to_string())
    print("\n=== camera ===")
    print(df.camera_motion.value_counts().to_string())
    df["scene"] = "face"
    df.drop(columns=["head"]).to_parquet(args.out)
    print(f"wrote {len(df):,d} rows -> {args.out}")


if __name__ == "__main__":
    main()
