"""Face SFT data, step 1: pick single-person face close-ups from OpenVid-1M metadata.

Same input as filter_openvid.py (the parquet from fetch_openvid_meta.py). Captions only give
candidates; encode_faces.py then runs a face detector and drops clips without one clear face.

    python dataset/filter_faces.py --meta data/openvid_meta --out data/faces.parquet
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.filter_openvid import SLOW, STATIC, head, parse_path

# the caption head must say it is a close-up / portrait of one person
CLOSEUP = re.compile(r"close[- ]?up|closeup|portrait|headshot|\bface\b|facial", re.I)
PERSON = re.compile(r"\b(?:man|woman|girl|boy|lady|guy|person|child|kid|teen(?:ager)?|"
                    r"gentleman|actress|actor|model|businessman|businesswoman)\b", re.I)
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
    ap.add_argument("--max_clips_per_video", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_parquet(args.meta)
    funnel = [("all", len(df))]
    parsed = df.video_path.map(parse_path)
    df["source"], df["video_id"] = parsed.str[0], parsed.str[1]
    if not args.all_sources:
        df = df[df.source.isin({"pixabay", "pexels", "mixkit"})]
        funnel.append(("stock sources", len(df)))

    df["head"] = df.caption.map(head)
    df = df[df["head"].str.contains(CLOSEUP) & df["head"].str.contains(PERSON)]
    funnel.append(("close-up of a person", len(df)))
    df = df[~df["head"].str.contains(MULTI) & ~df["head"].str.contains(NOT_REAL)]
    funnel.append(("single, real person", len(df)))

    df = df[df.seconds >= args.min_seconds]
    funnel.append((f"seconds>={args.min_seconds}", len(df)))
    # faces move more than scenery: keep some motion, drop shaky / cut clips
    df = df[(df.temporal_consistency_score >= 0.98) & (df.motion_score.between(0.5, 20))]
    funnel.append(("temporal/motion ok", len(df)))
    df = df[df.camera_motion.isin(STATIC | SLOW)]
    funnel.append(("static or slow camera", len(df)))

    df = (df.sort_values("aesthetic_score", ascending=False)
            .groupby("video_id").head(args.max_clips_per_video))
    funnel.append((f"<= {args.max_clips_per_video} clips/video", len(df)))

    print("=== funnel ===")
    for name, n in funnel:
        print(f"{name:28s}{n:>9,d}")
    print(df.source.value_counts().to_string())
    df["scene"] = "face"
    df.drop(columns=["head"]).to_parquet(args.out)
    print(f"wrote {len(df):,d} rows -> {args.out}")


if __name__ == "__main__":
    main()
