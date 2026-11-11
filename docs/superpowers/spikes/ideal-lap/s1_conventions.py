"""Stage 1: what do the G channels mean, signs, drag, axle shares, grip fraction."""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, cat, sm, G, HZ, W4

for name in sys.argv[1:]:
    f = load_file(name); laps = real_laps(f)
    print("=" * 100); print(f["file"][:70], "|", f["model"], "|", f["cls"], "| laps", len(laps), "of", len(f["laps"]), "| best %.3f" % min(l["time"] for l in laps))
    v = cat(laps, "v", 10); gx = cat(laps, "gx", 10); gy = cat(laps, "gy", 10); gz = cat(laps, "gz", 10)
    dv = np.concatenate([np.gradient(sm(l["v"], 10), 1 / HZ) for l in laps]) / G
    ok = v > 15
    k = np.polyfit(gx[ok], dv[ok], 1)
    print("[long] dv/dt vs gx: slope %.3f offset %.3f corr %.4f" % (k[0], k[1], np.corrcoef(gx[ok], dv[ok])[0, 1]))
    if laps[0]["posy"] is not None:
        grade = np.concatenate([np.gradient(sm(l["posy"], 25), 1 / HZ) / np.maximum(sm(l["v"], 25), 5) for l in laps])
        res = gx - dv
        print("       grade range %.3f..%.3f | corr(gx - dv/dt, grade) %.3f, slope %.2f (accelerometer would be ~ +1 or -1; kinematic ~ 0)" % (
            np.percentile(grade, 1), np.percentile(grade, 99), np.corrcoef(res[ok], grade[ok])[0, 1], np.polyfit(grade[ok], res[ok], 1)[0]))
        # vertical: kinematic vertical accel of the path = d2y/dt2
        d2y = np.concatenate([np.gradient(np.gradient(sm(l["posy"], 25), 1 / HZ), 1 / HZ) for l in laps]) / G
        gz_s = cat(laps, "gz", 25)
        kk = np.polyfit(d2y[ok], gz_s[ok], 1)
        print("[vert] gz vs d2(PosY)/dt2: slope %.3f offset %.3f corr %.3f | gz mean %.3f" % (kk[0], kk[1], np.corrcoef(d2y[ok], gz_s[ok])[0, 1], gz.mean()))
    yaw = cat(laps, "yaw", 10)
    k = np.polyfit(v[ok] * yaw[ok] / G, gy[ok], 1)
    print("[lat]  gy vs v*yaw/g: slope %.3f corr %.4f" % (k[0], np.corrcoef(v[ok] * yaw[ok] / G, gy[ok])[0, 1]))
    L = laps[0]
    print("[have] load %s susp %s fy %s fx %s gripfrac %s slip %s rideh %s posy %s" % tuple(L[k] is not None for k in ("load", "susp", "fy", "fx", "gf", "slip", "rh", "posy")))
    if L["load"] is not None and L["fx"] is not None:
        Fz = np.vstack([cat(laps, "load", 10, i) for i in range(4)]); Fx = np.vstack([cat(laps, "fx", 10, i) for i in range(4)]); Fy = np.vstack([cat(laps, "fy", 10, i) for i in range(4)])
        slow = (v < 40) & (np.abs(gx) < 0.1) & (np.abs(gy) < 0.1)
        k = np.polyfit(gy[ok], Fy.sum(0)[ok], 1); kx = np.polyfit(gx[ok], Fx.sum(0)[ok], 1)
        m_lat = abs(k[0]) / G
        print("[force] sum Fy vs gy: %.0f N/g (mass %.0f kg) corr %.4f | sum Fx vs gx: %.0f N/g corr %.4f" % (k[0], m_lat, np.corrcoef(gy[ok], Fy.sum(0)[ok])[0, 1], kx[0], np.corrcoef(gx[ok], Fx.sum(0)[ok])[0, 1]))
        sgn = np.sign(kx[0]); m = m_lat
        # drag: m*g*(gx + grade) = sgn*sumFx - D(v)
        grade_ = grade if laps[0]["posy"] is not None else 0 * v
        D = sgn * Fx.sum(0) - m * G * (gx + grade_)
        A = np.c_[np.ones(ok.sum()), v[ok] ** 2]
        c, *_ = np.linalg.lstsq(A, D[ok], rcond=None)
        r = D[ok] - A @ c
        print("        drag = %.0f N + %.3f*v^2 (=%.0f N at 250 km/h, %.2f g) | residual sd %.0f N (%.3f g)" % (c[0], c[1], c[0] + c[1] * (250 / 3.6) ** 2, (c[0] + c[1] * (250 / 3.6) ** 2) / m / G, r.std(), r.std() / m / G))
        cor = np.abs(gy) > 1.0
        fs = (Fy[:2].sum(0) / Fy.sum(0))[cor]
        fz_static_front = np.nan
        print("        front share of lateral force in corners: median %.3f (p10 %.3f p90 %.3f)" % (np.median(fs), *np.percentile(fs, [10, 90])))
        drv = (cat(laps, "thr") > 95) & (v > 20)
        print("        traction: front share of drive force at full throttle, by speed: " + ", ".join("%d: %.2f" % (a, np.median((Fx[:2].sum(0) / Fx.sum(0))[drv & (v * 3.6 >= a) & (v * 3.6 < a + 40)])) for a in range(80, 300, 40) if (drv & (v * 3.6 >= a) & (v * 3.6 < a + 40)).sum() > 50))
        bk = cat(laps, "brk") > 60
        print("        braking: front share of brake force: median %.3f | front share of load then %.3f" % (np.median((Fx[:2].sum(0) / Fx.sum(0))[bk]), np.median((Fz[:2].sum(0) / Fz.sum(0))[bk])))
        if L["gf"] is not None:
            gf = np.vstack([cat(laps, "gf", 1, i) for i in range(4)])
            Fzr = np.vstack([cat(laps, "load", 1, i) for i in range(4)]); Fxr = np.vstack([cat(laps, "fx", 1, i) for i in range(4)]); Fyr = np.vstack([cat(laps, "fy", 1, i) for i in range(4)])
            mu = np.hypot(Fxr, Fyr) / np.maximum(Fzr, 300)
            print("[gripfrac] range %.2f..%.2f, quantiles p10/50/90/99: %s" % (gf.min(), gf.max(), np.round(np.percentile(gf, [10, 50, 90, 99]), 3)))
            for i, w in enumerate(W4):
                ok2 = Fzr[i] > 1500
                qs = np.percentile(gf[i][ok2], [0, 20, 40, 60, 80, 90, 97, 100])
                print("        %s mean mu by grip-fraction band: " % w + ", ".join("%.2f-%.2f: %.2f" % (a, b, np.mean(mu[i][ok2 & (gf[i] >= a) & (gf[i] <= b)])) for a, b in zip(qs[:-1], qs[1:]) if b > a))
    print("[flags] ABS active %.1f s, TC active %.1f s, brake>60%% %.1f s, full throttle %.1f s (all laps)" % (
        (cat(laps, "abs") > 0).sum() / HZ, (cat(laps, "tc") > 0).sum() / HZ, (cat(laps, "brk") > 60).sum() / HZ, (cat(laps, "thr") > 95).sum() / HZ))
