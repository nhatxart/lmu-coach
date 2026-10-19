# Diagnostics — symptom → cause → fix, mapped to lmu-coach data

Thresholds marked *(EN)* are engineering heuristics: calibrate per car against the tire model and the
driver's best laps, and prefer relative comparisons (this lap vs. best, this corner vs. same-type corners)
over absolute numbers. When tire-model confidence is low, say so and lean on yaw/steering logic.

## Channels → concepts

| Concept | lmu-coach channel(s) | Notes |
|---|---|---|
| Rotation | yaw rate `mLocalRot.y` (deg/s) | the MRP is its peak |
| Front / rear slip | per-wheel slip angle (lat/long ground vel) | vs tire model `peak_slip_deg`, `efficient_range` |
| Platform slip | body slip angle β | ≈ rear slip; front slip ≈ β + road-wheel steer |
| Grip use | grip utilization, combined G | proxy, not measured force |
| Load transfer | brake %, throttle %, long/lat G | brake → front load, throttle → rear load |
| Engine braking | gear, RPM, shift timestamps | peak right after last downshift |
| Long. limit | ABS/TC flags, wheel speed vs ground speed | no ABS → front wheel speed collapse = lock |
| Corner structure | speed min, yaw max, steer max, pedal crossover | ≈ coincide at a clean MRP |

## Derived features to compute before asking the LLM *(EN)*
The local model should receive these per corner, not raw 100 Hz traces.

1. **MRP alignment**: time spread between min speed, max yaw, max steering, brake→throttle crossover.
   > ~0.3 s = smeared MRP.
2. **MRL (yaw plateau)**: time yaw stays within ~5 % of its max with throttle 5–60 % → "waiting".
3. **Rotation share before throttle**: ∫yaw dt from turn-in to throttle pickup ÷ total heading change of
   the corner. Target ≥ 0.5 (~0.55). < 0.45 → under-rotated entry.
4. **Understeer onset**: steering still increasing while yaw rate flat/falling (d yaw/dt ≤ 0, d steer/dt > 0)
   for > ~0.1 s, front slip > peak. This is the telemetry version of "adding lock but the world stopped
   turning faster".
5. **Grip-use gap at turn-in** ("tire sound gap"): normalized combined G (√(ax²+ay²) / that car's
   achievable G at that speed) dips > ~15 % between brake release and steering build. Gap = under the
   limit; break later and couple the release to the steering.
6. **Brake↔steer coupling**: lag from brake release start to steering > ~3° start. Positive lag = mode 1
   (release then steer). Also flag the release *duration* vs steering ramp duration (fast drop ≪ ramp).
7. **Brake trace shape**: time to peak (≈1 blink light/aero, ≈2 heavy); re-application bumps after release
   began (a "hill"); peak-to-terminal ratio on straights (aero cars bleed); terminal trail brake T % at
   throttle pickup.
8. **Downshift timing**: last downshift before / after turn-in; yaw or rear-slip spike within ~0.3 s of a
   shift; lap-to-lap variance in downshift timing.
9. **Countersteer type**: steering crossing to counter sign. Fast + large + β large → active; small with
   driver force still inward (or wheel drifting with no rate spike) → passive. Count active corrections.
10. **Correction speed**: time from rear-slip spike to countersteer peak; slow (> ~0.4 s) + growing β = spin risk.
11. **Unwind vs throttle**: steering decreasing before throttle > ~5 % → unwound early.
12. **Throttle shape**: initial application rate and the % reached before modulation; lift-reapply cycles;
    TC/wheelspin = level 3; no rear-slip rise + push = level 1.
13. **Consistency**: lap-to-lap σ of entry speed, brake point, peak pressure, downshift timing, MRP position.

## Entry (brake point → MRP)

| Symptom | Likely cause | Fix (one change) | Feel cue |
|---|---|---|---|
| Brake falls to ~0 before steering; grip gap; then big lock, front slip past peak | Release-then-steer (mode 1) — fronts unloaded | Brake a bit later and start turning as the release starts; release length ≈ steering ramp | Car points itself with little lock; continuous tire noise |
| Brake and steering rising together; front ABS; yaw falling | Re-applying brake while steering | Straighten to add brake; triangle-shaped trace | — |
| Heavy brake + large lock; rear slip > front; countersteer | Too much brake at turn-in (level 2→3) | Release faster once rotation starts (advanced string) | Rear "breathes" → let go sooner |
| ABS on with > ~10° lock; yaw flat | Overloaded fronts | Lower pressure at turn-in; ABS only in a straight line | Numb wheel, straight on |
| Steering flick at turn-in (high rate in first 0.2 s), inside-front lock, mid-corner push | Steering used first; ignores engine braking | Turn in slowly; let brake + downshift start rotation | Light hands, the car turns under you |
| Yaw / rear-slip spike right after a downshift | Downshift too early/abrupt, or mid-corner | Last downshift before turn-in, bigger blip, or later | Rear steps out exactly on the shift |
| Choppy stair-step release with yaw oscillation | Hunting over/under the limit | Smaller anticipatory steps; string drill | Fewer corrections |
| Entry speed varies lap to lap at same corner | Peak pressure / reference / downshift inconsistency | Fix brake marker (initial vs peak), same peak %, same gear+RPM at turn-in | Same engine note every lap |
| Turn-in point late, then aggressive lock to save apex | Late turn-in → asking for rotation too early | Turn in earlier and more progressively (V-line) | — |
| Over-slowed: early release, re-accelerate before MRP | Braked too early / built too tight an arc | Brake later; earlier, gentler turn-in | — |

## Mid-corner (around the MRP)

| Symptom | Likely cause | Fix | Feel cue |
|---|---|---|---|
| MRP spread > 0.3 s; yaw plateau with part throttle (MRL) | Waiting/hesitation; entry rotation short | Commit to max rotation at min speed; pivot into throttle | One clear pivot moment |
| Rotation share at throttle < 0.45 | Under-rotated entry | More rotation on late entry (hold a little brake, steer more toward MRP) | — |
| Mid-corner brake > ~10–15 % on high-aero car; slow-growing rear slip | Too much trail brake | Drop brake sooner once rotation is there | Rear nervous → let go |
| Brake 0 % long before MRP, steering growing, front slip > peak | Underload (coasting) | Hold ~1–10 % to the MRP | Wheel stays loaded |
| Throttle 10–40 % held before commit | Maintenance throttle crutch | Earlier/faster release, less mid brake, later downshift, earlier turn-in | Car rotates without "propping" |
| Steering frozen at peak, no room to add or unwind | Balance wrong earlier | Fix entry | — |
| Low steering but shallow line, apex missed | Entry over-rotated early | Release earlier/quicker; later downshift; spread rotation | — |
| Car hits apex too early, straightens, re-turns (bounce) | Under speed | Carry more speed next lap (brake less) | Too easy to rotate |

## Exit (MRP → straight)

| Symptom | Likely cause | Fix | Feel cue |
|---|---|---|---|
| Steering unwinds before throttle | Fear / hesitation | Throttle first, unwind a beat later | Wheel lightens as power comes |
| Throttle up, steering stuck at peak, front slip > peak, then lift | Unproductive throttle / slow unwind | Wait until pointed; one clean squeeze | One application |
| Push on power, no rear slip rise (level 1) | Throttle too timid | Faster initial application to the traction limit | Car keeps rotating gently |
| TC/wheelspin, rear slip spike, yaw rises on power | Level 3 or rotating on power | Same rate, lower initial %; get rotation before MRP | — |
| Alternating under/over on exit, runs wide | Entry rotation short (understeer→snap) | Fix entry rotation share | — |
| Active countersteer on exit, line goes wide | Correcting with steering first | Ease throttle 10–20 % first, then passive | — |
| Lift after a known upshift/crest snap every lap | Not anticipating | Pre-ease throttle before the known kick | Act before the rear moves |

## Whole-lap / habit patterns
- **Cluster corners by type** (hard-braking slow, medium, fast/lift, direction-change) and look for the same
  fault across a cluster. Habits repeat by corner type; fix the habit, not one corner.
- **Opposite styles in one corner**: heavy, slow trail brake early, then a sudden drop + big lock late =
  "two drivers in one corner". Coach one continuous motion.
- Many active countersteers + rear temps climbing → entry over-rotation from engine braking/brake.
- Fronts hot, frequent past-peak front slip → steering too much; rotate with the pedals.
- High ABS/TC counts with flat lap times → driving on electronics.
- Lap-to-lap spread in one corner → check downshift rhythm and brake peak first.

## Setup vs technique
Suggest setup only if the imbalance persists with correct inputs across several laps (e.g. oversteer with
≤ 5 % mid-corner brake, downshift before turn-in, clean throttle). Name the technique ruled out. Remind
that a better *feeling* setup must be confirmed by the delta.

## Car class notes *(EN — verify per car from data)*
- High-downforce prototypes (Hypercar/LMP2): U-line tendency, later turn-in, more linear steering, thin
  mid-corner brake trace, strong peak→terminal bleed, small slick peak slip. Check whether ABS exists; if
  not, lockup shows as front wheel speed collapsing.
- GT3/LMGT3: heavier, less aero, ABS + TC → more V-line, more trail brake, exponential steering, easy to
  lose rotation by leaning on ABS mid-turn, strong engine-braking effects.
- Hybrids/regen alter rear braking on entry; flag if deploy/regen correlates with entry balance.
