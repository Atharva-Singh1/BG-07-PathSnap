"""Run the HMM matcher on a few trips and print the matched sequence."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hmm_matcher import HMMConfig, build_matcher, load_trip, quick_eval

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder holding trip dirs, e.g. 'Student folder/student_test'")
ap.add_argument("--trips", nargs="*", help="trip ids; default first 3")
ap.add_argument("--config", help="json config (HMMConfig fields)")
ap.add_argument("--max-points", type=int, default=400, help="demo: truncate trace")
a = ap.parse_args()
cfg = HMMConfig.from_json(a.config) if a.config else HMMConfig()
root = Path(a.data)
ids = a.trips or sorted(p.name for p in root.iterdir() if p.is_dir())[:3]
for t in ids:
    trip = load_trip(root / t)
    trip["track"] = trip["track"][: a.max_points]
    m = build_matcher(trip, cfg)
    t0 = time.time(); res = m.match(trip["track"]); dt = time.time() - t0
    seq = [int(x) for x in res.arc_ids]
    dedup = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
    print(f"\n=== trip {t}: {len(seq)} pts, {dt:.1f}s, breaks={len(res.breaks)}, "
          f"no-candidate pts={(res.n_candidates==0).sum()}")
    print("matched arc sequence (deduped):", dedup[:40], "..." if len(dedup) > 40 else "")
    # reference restricted to the truncated part is unknown, so compare to full route prefix coverage
    ev = quick_eval(res.arc_ids, trip["route"], m.g)
    print("continuity=%.2f  precision(vs full route)=%.2f  pred arcs=%d" %
          (ev["continuity"], ev["arc_precision"], ev["n_pred_arcs"]))