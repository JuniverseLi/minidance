"""S1 step 2: filter OpenVid-1M metadata down to nature-scenery clips for T2V.

Input : parquet produced by fetch_openvid_meta.py (merged)
Output: filtered parquet + a printed funnel / scene distribution.

    python dataset/filter_openvid.py --meta data/openvid_meta --out data/scenery.parquet
"""
import argparse
import re

import pandas as pd

# Stock-footage sites only: clear licenses, no YouTube re-hosting.
SOURCES = {"pixabay", "pexels", "mixkit"}

# Category names (from file names) we never want, even if the caption looks fine.
BAD_CATEGORY = re.compile(
    r"city|urban|skyline|village|monument|heritage|cobblestone|wildlife|garden|park|"
    r"underwater|reef|savanna", re.I)

# Scene tags: the first match (in this order) on the caption head names the scene.
SCENES = {
    "aurora":     r"\baurora|northern lights",
    "night_sky":  r"night sky|starry|stars\b|milky way|galaxy",
    "fire":       r"campfire|bonfire|\bflames?\b|fireplace",
    "waterfall":  r"waterfall|cascade",
    "ocean":      r"\bocean|\bsea\b|seascape|\bwaves?\b|beach|coast|shore",
    "river_lake": r"\briver|\blake|stream|creek|pond",
    "snow":       r"\bsnow|glacier|\bice\b|winter",
    "rain_storm": r"\brain\b|raining|storm|lightning",
    "fog":        r"\bfog|\bmist|haze",
    "sunset":     r"sunset|sunrise|dusk|dawn|golden hour",
    "clouds":     r"\bclouds?\b|cloudy|sky\b|timelapse|time-lapse",
    "mountain":   r"mountain|peak|cliff|canyon|valley|hill|rock formation|desert",
    "forest":     r"forest|\btrees?\b|woods|jungle|meadow|field|grass|flower|landscape|countryside",
}
SCENE_RE = {k: re.compile(v, re.I) for k, v in SCENES.items()}

# Things a small model draws badly or that carry portrait / branding risk.
BAD_CONTENT = re.compile(
    r"\b(?:person|people|man|men|woman|women|child|children|boy|girl|kid|face|hands?|"
    r"crowd|tourists?|hikers?|surfers?|swimmers?|couple|someone|figure|silhouette of a|"
    r"cars?|vehicles?|trucks?|road traffic|boats?|sailboats?|ships?|yachts?|kayaks?|canoes?|"
    r"airplane|drone|"
    r"text|logo|watermark|words|letters|sign|"
    r"buildings?|houses?|city|street|bridge|tower)\b", re.I)
# Checked on the caption head only: what the clip is *about* (not background mentions).
BAD_SUBJECT = re.compile(
    r"\b(?:digital|animation|animated|render(?:ed|ing)?|cgi|computer[- ]generated|graphics?|"
    r"abstract|bokeh|particles?|illustration|cartoon|painting|metallic|3d|constellations?|illusion|"
    r"animals?|birds?(?!['’]s?[- ]?eye)|seagulls?|gulls?|ducks?|swans?|geese|sheep|goats?|deer|"
    r"horses?|cows?|cattle|dogs?|cats?|fish|trout|salmon|snails?|insects?|butterfl\w+|bees?|"
    r"spiders?|lizards?|turtles?|frogs?|squirrels?|bears?|whales?|dolphins?|seals?|jellyfish|"
    r"lions?|lioness|tigers?|leopards?|elephants?|giraffes?|zebras?|monkeys?|foxes|fox|wolf|wolves|"
    r"rabbits?|window|indoors?|room|interior)\b",
    re.I)
# Sentences like "There are no people in the scene" must not trigger BAD_CONTENT.
NEGATION = re.compile(r"\b(no|not|without|absence|devoid|free of|nor|any)\b", re.I)

STATIC = {"static"}
SLOW = {"pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out"}


def parse_path(path):
    fn = path.rsplit("/", 1)[-1][:-4]
    src = fn.split("_", 1)[0]
    video_id = re.sub(r"_\d+$", "", fn)  # strip clip index -> source video
    category = re.sub(r"_?\d+$", "", re.sub(r"^[a-z]+_", "", video_id))
    return src, video_id, category


def head(caption, n=2):
    return " ".join(re.split(r"(?<=[.!?])\s+", caption.strip())[:n])


def has_bad_content(caption):
    for sent in re.split(r"(?<=[.!?])\s+", caption):
        if BAD_CONTENT.search(sent) and not NEGATION.search(sent):
            return True
    return False


def scene_of(text):
    for k, r in SCENE_RE.items():
        if r.search(text):
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/openvid_meta", help="parquet file or the folder from fetch_openvid_meta.py")
    ap.add_argument("--out", default="data/scenery.parquet")
    ap.add_argument("--min_seconds", type=float, default=2.5)
    ap.add_argument("--max_clips_per_video", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(args.meta)
    funnel = [("all", len(df))]

    parsed = df.video_path.map(parse_path)
    df["source"] = parsed.str[0]
    df["video_id"] = parsed.str[1]
    df["category"] = parsed.str[2]
    df = df[df.source.isin(SOURCES)]
    funnel.append(("stock sources", len(df)))

    df = df[~df.category.str.contains(BAD_CATEGORY)]
    funnel.append(("category ok", len(df)))

    df["head"] = df.caption.map(head)
    df["scene"] = df["head"].map(scene_of)
    df = df[df.scene.notna()]
    funnel.append(("scenery caption", len(df)))

    df = df[~df.caption.map(has_bad_content)]
    funnel.append(("no people/cars/text", len(df)))

    df = df[~df["head"].str.contains(BAD_SUBJECT)]
    funnel.append(("no CG/animal subject", len(df)))

    df = df[df.seconds >= args.min_seconds]
    funnel.append((f"seconds>={args.min_seconds}", len(df)))

    df = df[(df.temporal_consistency_score >= 0.993) & (df.motion_score.between(0.3, 30))]
    funnel.append(("temporal/motion ok", len(df)))

    df = df[df.camera_motion.isin(STATIC | SLOW)]
    funnel.append(("static or slow camera", len(df)))

    # Diversity: keep the best-looking few clips of each source video.
    df = (df.sort_values("aesthetic_score", ascending=False)
            .groupby("video_id").head(args.max_clips_per_video))
    funnel.append((f"<= {args.max_clips_per_video} clips/video", len(df)))

    print("=== funnel ===")
    for name, n in funnel:
        print(f"{name:28s}{n:>9,d}")
    print(f"unique source videos: {df.video_id.nunique():,d}")

    print("\n=== aesthetic thresholds ===")
    for q in [0, .25, .5]:
        t = df.aesthetic_score.quantile(q)
        print(f"aesthetic >= {t:.2f} (top {1 - q:.0%}): {(df.aesthetic_score >= t).sum():,d}")

    print("\n=== scenes ===")
    print(df.scene.value_counts().to_string())
    print("\n=== sources ===")
    print(df.source.value_counts().to_string())
    print("\n=== camera ===")
    print(df.camera_motion.value_counts().to_string())

    df.drop(columns=["head"]).to_parquet(args.out)
    print(f"\nwrote {len(df):,d} rows -> {args.out}")


if __name__ == "__main__":
    main()
