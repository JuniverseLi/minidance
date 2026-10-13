"""S1 step 1: dump OpenVid-1M (Lance) metadata to local parquet, one file per fragment.

Only reads small columns (no video bytes, no embeddings). Resumable: finished
fragments are skipped on rerun.

    python dataset/fetch_openvid_meta.py --out data/openvid_meta
"""
import argparse
import os
import time

import lance
import pyarrow as pa
import pyarrow.parquet as pq

URI = "hf://datasets/lance-format/Openvid-1M/data/train.lance"
COLUMNS = ["video_path", "caption", "aesthetic_score", "motion_score",
           "temporal_consistency_score", "camera_motion", "frame", "fps", "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/openvid_meta")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = lance.dataset(URI)
    frags = ds.get_fragments()
    t0 = time.time()
    for i, frag in enumerate(frags):
        path = os.path.join(args.out, f"frag_{frag.fragment_id:05d}.parquet")
        if os.path.exists(path):
            continue
        for attempt in range(5):
            try:
                tbl = frag.to_table(columns=COLUMNS, with_row_id=True)
                break
            except Exception as e:  # network hiccups / rate limits
                print(f"  frag {frag.fragment_id} attempt {attempt} failed: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"fragment {frag.fragment_id} failed 5 times")
        pq.write_table(tbl, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"[{i + 1}/{len(frags)}] frag {frag.fragment_id}: {tbl.num_rows} rows, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)

    print("done")


if __name__ == "__main__":
    main()
