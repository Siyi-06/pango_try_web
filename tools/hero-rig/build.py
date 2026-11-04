# Builds the layered pangolin rig from the flat artwork.
#   in : public/brand/mark-dark.png (untouched)
#   out: public/rig/{base.png, atlas.png, rig.json} + tools/hero-rig/preview/* (debug, git-ignored)
import json, pathlib
import cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]  # = the website folder
SRC = ROOT / "public/brand/mark-dark.png"
OUT = ROOT / "public/rig"; OUT.mkdir(parents=True, exist_ok=True)
PRE = ROOT / "tools/hero-rig/preview"; PRE.mkdir(parents=True, exist_ok=True)

im = cv2.imread(str(SRC)); H, W = im.shape[:2]
f = im.astype(np.float32)
B, G, R = f[..., 0], f[..., 1], f[..., 2]
lum = (R * .3 + G * .55 + B * .15)

# ---------------------------------------------------------------- regions
e = cv2.Canny(cv2.GaussianBlur(im, (0, 0), 1.5), 30, 80)
e = cv2.dilate(e, np.ones((3, 3), np.uint8))
fg = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) > 40).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(((e == 0) & (fg > 0)).astype(np.uint8), 4)
comp = lambda i: (lab == i).astype(np.uint8)
def union(ids):
    m = np.zeros((H, W), np.uint8)
    for i in ids: m |= comp(i)
    return m
def dil(m, r): return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) if r > 0 else m
def fill_holes(m):
    inv = (1 - m).astype(np.uint8); ff = inv.copy(); msk = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, msk, (0, 0), 2); return ((ff != 2) | (m > 0)).astype(np.uint8)

# shield: convex hull of its gold rim (component 86)
ys, xs = np.where(lab == 86)
hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
shield = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(shield, hull, 1)
shield |= dil(comp(84), 2) & (lum > 60).astype(np.uint8) | comp(84)     # the shield's top edge curling up behind the neck is shield too (fixed)
SH_AXIS = float(xs.min() + xs.max()) / 2

# ring: the fitted ellipse; core band only (its glow stays with whatever is underneath)
ex, ey, eth, erx, ery = .5042 * W, .5450 * H, .0462, .4706 * W, .1712 * H
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
dx, dy = xx - ex, yy - ey; c, s = np.cos(eth), np.sin(eth)
u, v = (c * dx + s * dy) / erx, (-s * dx + c * dy) / ery
rr = np.sqrt(u * u + v * v)
ring_px = np.abs(rr - 1) * np.minimum(erx, ery * 1.0)        # rough distance in px
ring_ids = [101, 93, 96, 104, 106, 107, 197, 253, 298, 286]
pin = cv2.imread(str(ROOT / "public/brand/mark-dark-pin.png"), 0).astype(np.float32) / 255
core = (union(ring_ids) | ((lum > 228) & (pin > .8)).astype(np.uint8))
core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
ring = (dil(core, 3) & (pin > .5)).astype(np.uint8)
nr, rl, rs, _ = cv2.connectedComponentsWithStats(ring, 8)
ring = np.isin(rl, [k for k in range(1, nr) if rs[k, 4] > 2500]).astype(np.uint8)       # drop stray highlights
# refit the ellipse to the actual ring core (used for the travelling light)
ys_, xs_ = np.where((core > 0) & (pin > .8))
(ex, ey), (d1, d2), ang = cv2.fitEllipse(np.stack([xs_, ys_], 1).astype(np.float32))
erx, ery, eth = max(d1, d2) / 2, min(d1, d2) / 2, np.deg2rad(ang - 90 if d1 < d2 else ang)
print("ellipse", ex, ey, erx, ery, np.rad2deg(eth))
sphere = (((xx - 944) ** 2 + (yy - 382) ** 2) < 47 ** 2).astype(np.uint8) | dil(union([109, 135]), 2)
ring |= sphere
# the front of the ring also passes over the shield bottom: that part is ring
static_top = ring

# head (+ eye, nose), arm + claws, tail
head = fill_holes(dil(union([42, 34, 74]), 2)) & (1 - shield)
clawcol = ((R - G > 28) & (G - B < 38) & (lum > 72) & (lum < 175)).astype(np.uint8)    # claw tone only: not the gold rim, not the shadow
clawzone = (((xx - 545) ** 2 + (yy - 380) ** 2) < 62 ** 2).astype(np.uint8)
claws = cv2.morphologyEx(clawcol & clawzone, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) | comp(115)
# keep only the three claws themselves (one connected blob), not the brown shadow below the paw or rim specks
kc, kl, ks, kcen = cv2.connectedComponentsWithStats(claws, 8)
claws = np.isin(kl, [j for j in range(1, kc) if ks[j, 4] > 400 and kcen[j][1] < 405]).astype(np.uint8)
claws = dil(claws, 1) & clawzone
# each claw as one whole shape (its own convex outline), so the darker underside comes with it
kc2, kl2, ks2, _ = cv2.connectedComponentsWithStats(cv2.erode(claws, np.ones((3, 3), np.uint8)), 8)
cl_full = np.zeros((H, W), np.uint8)
for j in range(1, kc2):
    if ks2[j, 4] < 60: continue
    ys_, xs_ = np.where(kl2 == j)
    cv2.fillConvexPoly(cl_full, cv2.convexHull(np.stack([xs_, ys_], 1).astype(np.int32)), 1)
claws = dil(cl_full, 2) & ((lum > 22).astype(np.uint8))
# the three claws, traced by hand from the original artwork (image px) — exact shapes, no colour guessing
CLAWS = [
  [(531.3,347.5),(541.3,346.3),(550,348.8),(558.8,355),(567.5,363.8),(575,372.5),(580,381.3),(582.5,389.4),(578.8,387.5),(571.3,381.3),(560,375),(550,371.3),(541.3,367.8),(535,362.5),(531.3,353.8)],
  [(523.8,372.5),(530,370.6),(540,370.6),(550,372.5),(558.8,377.5),(567.5,387.5),(575,397.5),(580,408.8),(575,406.9),(565,402.5),(555,398.1),(545,393.8),(535,390.6),(528.8,387.5),(524.4,378.8)],
  [(511.9,395),(516.3,391.3),(525,390),(535,391.3),(543.8,395),(551.3,400),(557.5,407.5),(564.4,418.8),(561.3,417.5),(553.8,415),(543.8,412.5),(532.5,410),(522.5,407.5),(515,404.4),(511.9,400)],
]
claws = np.zeros((H, W), np.uint8)
for poly in CLAWS: cv2.fillPoly(claws, [np.round(np.array(poly) * 4).astype(np.int32)], 1, shift=2)
claws = dil(claws, 1)
print("claws traced", len(CLAWS))
arm = (fill_holes(dil(comp(73), 2)) & (1 - shield)) | claws        # fur + the three claws, nothing between or below them
arm = fill_holes(arm) & ((1 - shield) | claws)                     # no little holes between the fur and the claws
arm = arm | (cv2.morphologyEx(arm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) & dil(claws, 4) & (1 - shield) | claws)   # pinholes where claws meet the fur
# tail: the whole tail area as one piece (every spike, sliver and edge line), so nothing in it stays behind
tailzone = np.zeros((H, W), np.uint8)
cv2.fillPoly(tailzone, [np.array([[728, 392], [800, 372], [842, 430], [846, 540], [816, 566], [742, 566], [716, 520]], np.int32)], 1)
tail = ((dil(union([157, 236, 299]), 2) | (tailzone & (lum > 38).astype(np.uint8))) & (1 - dil(shield, 7))).astype(np.uint8)
tail = cv2.morphologyEx(tail, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) & (1 - dil(shield, 7))      # never take shield pixels

# scales: every other sizeable component on the body
used = {42, 34, 74, 73, 115, 157, 236, 299, 86, 84, 91, 127, 109, 135, *ring_ids}
head_kids = {21, 14, 4, 5, 35}
BODY_C = np.array([481., 401.])
scales = []
for i in range(1, n):
    if i in used or st[i, 4] < 300: continue
    m = comp(i)
    if (m & shield).sum() > .5 * m.sum(): continue
    mo = cv2.moments(m, True)
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    th = .5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
    ax = np.array([np.cos(th), np.sin(th)])
    ys, xs = np.where(m > 0); pr = (xs - cx) * ax[0] + (ys - cy) * ax[1]
    p1 = np.array([cx, cy]) + ax * pr.min(); p2 = np.array([cx, cy]) + ax * pr.max()
    # C-shaped body: the fixed end of a scale is the end nearer the head along the body (head → back → tail)
    def along(q):   # 0 at the head, grows round the back and down to the tail
        a = np.degrees(np.arctan2(q[1] - BODY_C[1], q[0] - BODY_C[0]))
        return (-55 - a) % 360
    s1, s2 = along(p1), along(p2)
    near_c = min(np.linalg.norm(p1 - BODY_C), np.linalg.norm(p2 - BODY_C)) < 120
    if abs(s1 - s2) > 4 and abs(s1 - s2) < 180 and not near_c:
        root, tip = (p1, p2) if s1 < s2 else (p2, p1)
    else:
        root, tip = (p1, p2) if np.linalg.norm(p1 - BODY_C) < np.linalg.norm(p2 - BODY_C) else (p2, p1)
    pivot = root + (np.array([cx, cy]) - root) * .12
    out = float(np.linalg.norm(np.array([cx, cy]) - BODY_C) / 330)
    scales.append(dict(id=int(i), mask=dil(m, 2), pivot=pivot.tolist(), tip=tip.tolist(), area=int(st[i, 4]),
                       out=round(out, 3), parent="head" if i in head_kids else "body"))
# draw outer scales first: inner scales overlap the roots of the outer ones
scales.sort(key=lambda d: -d["out"])
print("scales", len(scales))

# ---------------------------------------------------------------- base plate
moving_parts = (head | arm | tail).astype(np.uint8)
base = cv2.inpaint(im, dil(moving_parts, 4), 6, cv2.INPAINT_TELEA)
# behind the head, paw and tail it is in shadow: whatever a moving part uncovers reads as shadow, never a bright smear
mp = cv2.GaussianBlur(dil(moving_parts, 4).astype(np.float32), (0, 0), 2)[..., None]
base = base.astype(np.uint8)   # no dark halo behind the head / paw / tail (it made them look detached)
smask = np.zeros((H, W), np.uint8)
for d in scales:
    if d["parent"] != "head": smask |= d["mask"]
# under every scale: a deeper layer of scales (the artwork shifted toward the body centre, a little darker),
# so a lifting tip reveals more scales underneath, not a hole
ox, oy = xx - BODY_C[0], yy - BODY_C[1]; od = np.sqrt(ox * ox + oy * oy) + 1e-3
deep = cv2.remap(base, (xx + ox / od * 11).astype(np.float32), (yy + oy / od * 11).astype(np.float32), cv2.INTER_LINEAR)
shade = cv2.GaussianBlur(deep, (0, 0), 1.2).astype(np.float32)
# no bright highlights underneath: tone it into a dark, even bronze (the shaded scales below), never whitish
sl = (shade[..., 2] * .3 + shade[..., 1] * .55 + shade[..., 0] * .15)[..., None]
shade = shade * np.minimum(1., 92. / np.maximum(sl, 1.)) * .9
shade = shade * .75 + np.array([28., 52., 92.]) * .25 * (sl / 255. + .4)       # warm bronze tint (BGR)
sm = cv2.GaussianBlur(smask.astype(np.float32), (0, 0), 1.5)[..., None]
base = base.astype(np.uint8)            # scales sit on the artwork itself
# no "second skin": under the flexible half of each scale, erase its outline so a moving tip leaves no ghost edge
band = np.zeros((H, W), np.uint8)
for d in scales:
    if d["parent"] == "head": continue
    m = d["mask"]; piv = np.array(d["pivot"]); tip = np.array(d["tip"]); ax_ = tip - piv; L = np.linalg.norm(ax_) + 1e-3; ax_ /= L
    kk = ((xx - piv[0]) * ax_[0] + (yy - piv[1]) * ax_[1]) / L
    ring_ = (dil(m, 5) & (1 - cv2.erode(m, np.ones((9, 9), np.uint8)))).astype(np.uint8)
    band |= (ring_ & (kk > .3).astype(np.uint8))
band &= (1 - dil(ring, 3)) & (1 - dil(shield, 2)) & (lum > 30).astype(np.uint8)
band &= cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))     # inside the body only: the outer curve stays crisp
base = cv2.inpaint(base, band, 4, cv2.INPAINT_TELEA)
base[static_top > 0] = im[static_top > 0]

# shield layer: original shield, with the claws removed (rebuilt from the mirrored rim, matched in colour)
sh_img = im.copy()
cl = (dil(arm, 3) & shield).astype(np.uint8)                   # everything the arm/claws cover on the shield
hd = cv2.distanceTransform(shield, cv2.DIST_L2, 5)              # px from the shield's outer edge
known_rim = ((hd < 15) & (cl == 0) & (shield > 0)).astype(np.uint8)
X0, Y0, X1, Y1 = 470, 250, 810, 620
rim_img = im.copy()
rim_img[Y0:Y1, X0:X1] = cv2.inpaint(im[Y0:Y1, X0:X1], (1 - known_rim[Y0:Y1, X0:X1]).astype(np.uint8), 4, cv2.INPAINT_TELEA)
inner_known = ((hd >= 19) & (cl == 0)).astype(np.uint8)
int_img = cv2.inpaint(im, (((1 - inner_known) & shield) | cl).astype(np.uint8), 9, cv2.INPAINT_TELEA)
t_ = np.clip((hd - 15.) / 4., 0, 1)[..., None]
rebuilt = (rim_img.astype(np.float32) * (1 - t_) + int_img.astype(np.float32) * t_).astype(np.uint8)
# the part of the shield hidden under the paw sits in the paw's shadow: a revealed sliver reads as shadow, never a bright line
shadow = (rebuilt.astype(np.float32) * .32).astype(np.uint8)
sh_img[cl > 0] = shadow[cl > 0]

# ---------------------------------------------------------------- sprites + atlas
def feather(m, r=.8): return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), r) * 1.6 - .3, 0, 1)
sprites = []
def add(name, mask, img=im, pivot=(0, 0), parent="body", kind=0, extra=None, alpha=None):
    ys, xs = np.where(mask > 0)
    x0, y0, x1, y1 = max(xs.min() - 3, 0), max(ys.min() - 3, 0), min(xs.max() + 4, W), min(ys.max() + 4, H)
    a = ((feather(mask) if alpha is None else alpha) * 255).astype(np.uint8)[y0:y1, x0:x1]
    rgba = np.dstack([img[y0:y1, x0:x1], a])
    sprites.append(dict(name=name, rect=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)], pivot=[float(p) for p in pivot],
                        parent=parent, kind=kind, rgba=rgba, **(extra or {})))

for d in scales:
    add(f"s{d['id']}", dil(d["mask"], 1), pivot=d["pivot"], parent=d["parent"], alpha=feather(dil(d["mask"], 1), .5),   # solid to its outline
        extra=dict(tip=d["tip"], out=d["out"], area=d["area"]))
add("tail", tail, pivot=(752, 578), parent="body")
add("head", head, pivot=(515, 290), parent="body", kind=1)
add("shield", shield, img=sh_img, kind=2)
armA = feather(dil(arm, 2), 1.0)
edge = ((dil(arm, 2) - arm) > 0)        # only the outside fringe; inside the paw stays solid
armA[edge] *= np.clip((lum[edge] - 28.) / 70., 0, 1)
armA = cv2.GaussianBlur(armA, (0, 0), .6)
solid = cv2.erode(cv2.morphologyEx(dil(arm, 2), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), np.ones((5, 5), np.uint8)) > 0
armA[solid] = 1.                                                      # fully solid inside the paw: no see-through pinholes
add("arm", dil(arm, 2), pivot=(446, 268), parent="body", alpha=armA)
add("ring", static_top, kind=3)
# order: scales (outer→inner) · tail · head-kid scales go after the head
order = [s for s in sprites if s["name"].startswith("s") and s["parent"] == "body"] + \
        [s for s in sprites if s["name"] == "tail"] + [s for s in sprites if s["name"] == "head"] + \
        [s for s in sprites if s["name"].startswith("s") and s["parent"] == "head"] + \
        [s for s in sprites if s["name"] in ("shield", "arm", "ring")]

AW, pad = 2048, 3; x = y = rowh = 0
for s in sorted(order, key=lambda s: -s["rect"][3]):
    w, h = s["rect"][2], s["rect"][3]
    if x + w + pad > AW: x, y, rowh = 0, y + rowh + pad, 0
    s["at"] = [x, y]; x += w + pad; rowh = max(rowh, h)
AH = y + rowh
atlas = np.zeros((AH, AW, 4), np.uint8)
for s in order:
    ax_, ay_ = s["at"]; w, h = s["rect"][2], s["rect"][3]
    rgba = s["rgba"].copy(); rgba[..., :3] = rgba[..., [2, 1, 0]]
    atlas[ay_:ay_ + h, ax_:ax_ + w] = rgba
cv2.imwrite(str(OUT / "atlas.png"), cv2.cvtColor(atlas, cv2.COLOR_RGBA2BGRA))
cv2.imwrite(str(OUT / "base.png"), base)
meta = dict(W=W, H=H, AW=AW, AH=AH, eye=[668, 205], mouth=[695, 249], nose=[764, 264], noseRoot=[728, 245],
            knuckle=[540, 372], body=BODY_C.tolist(), ellipse=[ex, ey, eth, erx, ery],
            sprites=[{k: v for k, v in s.items() if k != "rgba"} for s in order])
(OUT / "rig.json").write_text(json.dumps(meta))
print("atlas", AW, AH, "sprites", len(order))

# debug: layer map
dbg = (base.astype(np.float32) * .35).astype(np.uint8)
rng = np.random.default_rng(1)
for s in order:
    x0, y0, w, h = s["rect"]; a = s["rgba"][..., 3:4] / 255.
    col = {"head": (60, 160, 255), "arm": (60, 255, 120), "tail": (255, 80, 200), "shield": (200, 200, 200), "ring": (0, 230, 255)}.get(s["name"], rng.integers(80, 255, 3))
    reg = dbg[y0:y0 + h, x0:x0 + w].astype(np.float32)
    dbg[y0:y0 + h, x0:x0 + w] = (reg * (1 - a * .7) + np.array(col, np.float32) * a * .7).astype(np.uint8)
    if "pivot" in s and s["parent"] in ("body", "head") and s["name"] not in ("shield", "ring"):
        cv2.circle(dbg, tuple(int(v) for v in s["pivot"]), 3, (0, 0, 255), -1)
cv2.imwrite(str(PRE / "layers.png"), dbg)
cv2.imwrite(str(PRE / "base.png"), base)
cv2.imwrite(str(PRE / "shield.png"), sh_img[250:600, 480:800])
