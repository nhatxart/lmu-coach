## USER
Example lap (invented data, for style only). Huracán GT3, Portimão T1. Lap 1:44.9 vs best 1:44.3. T1: brake peak 92 % at 110 m; brake reaches 0 % 14 m before steering exceeds 10°; normalized combined G dips 22 % in that gap; steering then climbs to 58°; front slip 6.1° vs model peak 4.2° (conf 0.8); ABS 0.6 s with 35° lock; yaw peaks 0.5 s after min speed; rotation share at throttle 0.41; throttle 25 % held 1.1 s before full. Same release-then-steer pattern in T5 and T13 (other hard-braking, slow corners).

## ASSISTANT
## Lap summary
Most of the 0.6 s is in your slow, hard-braking corners (T1, T5, T13), and it's one habit: you finish braking, *then* turn. For a moment the tires do nothing at all, and then the fronts get asked to do everything.

### T1 — entry (same habit in T5, T13)
- **What the data shows:** brake is at zero 14 m before you really start turning, and your combined G drops 22 % in that gap. Then lock climbs to 58°, and when you go back to the brake you get 0.6 s of ABS at 35°.
- **What the tires are doing:** fronts at 6.1° slip against a ~4.2° peak, over the top of the curve. More lock there isn't more turn, it's more scrub. Because the fronts gave up, the rears had nothing to react to either.
- **Why:** releasing before turning sends the weight to the rear exactly when the fronts need it. The car under-rotated, so you reached for the one tool that can't help once the fronts are saturated.
- **Fix:** brake a touch later, and start turning the moment the release starts. Think of a string from the pedal to the rim: wheel comes in, pedal comes up, together.
- **Feel cue:** the car should start pointing with very little lock, and the tire noise shouldn't go quiet between braking and turning.

### T1 — mid/exit
Only 41 % of the corner's rotation was done when you picked up throttle, so the 25 % you held for over a second was propping the car up, not driving it out. Fix the entry and this mostly fixes itself: rotate at least half the corner on the brakes, then one clean squeeze.

## Drill for next run
10 laps, T1 only matters. Exaggerate it first: start turning slightly *before* the release, even if it feels wrong. Then settle on release and turn together. Check after: no G dip at turn-in, ABS off past ~10° of lock, front slip near 4°, rotation share ≥ 0.5. Then carry it to T5 and T13.

## Confidence
Tire model confidence 0.8, so the slip comparison is solid. Time cost per corner is not measured here (no reference-lap delta), so I ranked by importance instead. The grip-gap figure is a combined-G proxy, not measured tire force.
