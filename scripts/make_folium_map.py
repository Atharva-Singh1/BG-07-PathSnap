"""Report map (section 6): GPS points, road network, reference route, predicted route.

Uses the reference route, so it is for the REPORT only (never the UI/API).
python scripts/make_folium_map.py --data-dir "<path>/student_test" --trip 00000007 --out results/maps/trip_00000007.html
"""
import argparse
import sys
from pathlib import Path

import folium
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_data  # noqa: E402
from src.evaluate import load_matcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--trip", required=True)
ap.add_argument("--matcher", default="src.hmm_matcher:match")
ap.add_argument("--out", required=True)
a = ap.parse_args()

t = load_data(a.data_dir)[a.trip]
trace, nodes, arcs, route = t["trace"], t["road_network"]["nodes"], t["road_network"]["arcs"], t["route"]
pred = [int(x) for x in load_matcher(a.matcher)(trace, t["road_network"]) if int(x) >= 0]
seg = lambda i: [(nodes[arcs[i, 0], 1], nodes[arcs[i, 0], 0]), (nodes[arcs[i, 1], 1], nodes[arcs[i, 1], 0])]

lat, lon = trace[:, 1], trace[:, 0]
m = folium.Map(tiles="CartoDB positron", control_scale=True)
m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
pad = 0.003
near = np.flatnonzero((nodes[arcs[:, 0], 1] > lat.min() - pad) & (nodes[arcs[:, 0], 1] < lat.max() + pad)
                      & (nodes[arcs[:, 0], 0] > lon.min() - pad) & (nodes[arcs[:, 0], 0] < lon.max() + pad))
layers = [("Road network (nearby arcs)", near, "#9e9e9e", 1, 0.5, False),
          ("Reference route", sorted(set(route.tolist())), "#1a9641", 7, 0.6, True),
          ("Predicted route (" + a.matcher.split(":")[0].split(".")[-1] + ")", sorted(set(pred)), "#d7191c", 3, 0.9, True)]
for name, ids, color, w, op, show in layers:
    g = folium.FeatureGroup(name=name, show=show)
    for i in ids:
        folium.PolyLine(seg(i), color=color, weight=w, opacity=op).add_to(g)
    g.add_to(m)
gps = folium.FeatureGroup(name="GPS observations")
for la, lo in zip(lat, lon):
    folium.CircleMarker((la, lo), radius=2, color="#2b83ba", fill=True, weight=0, fill_opacity=0.8).add_to(gps)
gps.add_to(m)
folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="black")).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.get_root().html.add_child(folium.Element(
    '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:6px 10px;'
    'border:1px solid #999;font-size:12px"><b>Trip ' + a.trip + '</b><br>'
    '<span style="color:#2b83ba">&#9679;</span> GPS observations<br>'
    '<span style="color:#9e9e9e">&#9644;</span> Road network<br>'
    '<span style="color:#1a9641">&#9644;</span> Reference route<br>'
    '<span style="color:#d7191c">&#9644;</span> Predicted route</div>'))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
m.save(a.out)
print("wrote", a.out)
