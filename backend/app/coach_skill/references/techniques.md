# Technique library

Each section: what it is → what the tires are doing → how it goes wrong → how it shows in data.
A paraphrased synthesis of Suellio Almeida's teaching (The Motor Racing Book Vol. 1, coaching-session
and lesson transcripts, academy lessons). Numbers are his stated baselines unless marked
*(engineering note)*. Pull only the sections relevant to the finding — never dump this file.

---

## 1. Slip angle and the "platform"
- Slip angle = angle between where a tire points and where it travels. The contact patch stays stuck
  while the carcass twists; that twist is a loaded spring and its release is the lateral force.
- Force rises to an **optimal slip angle**, then saturates and falls; past peak the extra deformation is
  friction (heat, scrub), not grip.
- Rough peaks: road tire ~8–12°, semi-slick ~5–7°, slick ~2–4°. Slicks peak fast and drop fast.
- Front resistance in the wheel builds toward peak and goes light past it; peak wheel weight arrives a
  hair before peak force, so the true limit is a tiny bit beyond the heaviest point.
- **Platform slip:** think of the whole chassis sliding at one small angle with all four tires near
  optimal (a kart drifting through a corner, scaled down). Fast cars "on rails" are doing the same
  thing at a few degrees.
- **Slip-angle arithmetic** (simplified, 1:1 steering ratio): rear slip ≈ body slip angle β (rears
  can't steer); front slip ≈ β + road-wheel steer angle (countersteer subtracts).
  - Already sliding 5° and steering 5° more into the corner → fronts at ~10°: overloaded → rotation
    dies. Once the car is rotating, *less* steering (even zero) gives the most rotation.
  - Countersteer only saves a slide once it exceeds β so the fronts push the other way. Countersteer
    smaller than β still pushes the nose inward → the slide grows. That's the point of no return.
  - Steering *into* the slide piles slip on the fronts → the fronts die and the slide collapses into
    understeer.
  - Therefore corrections must be **fast and small**: slow, gentle countersteer never catches up with
    a growing β.
- Oversteer heats all four tires (fronts too unless countersteer matches the slide) — hence the common
  "oversteer, then understeer right after the correction".

## 2. The limit, understeer, oversteer, neutral
- Understeer = under-rotation: yaw rate lower than the steering asks for. Fronts past peak, rears under-used.
  It is usually caused by steering **too much**, not too little.
- Oversteer = over-rotation, and always a *reaction*: the fronts made more rotation than the rears
  could follow. Its cause is excess front grip (from trail brake, engine braking, steering), except
  when the rears lock.
- Neutral steer = both axles sliding similarly near optimal; all four tires working. The goal state.
  Equal *large* slides at both ends look heroic and are slow.
- Understeer = two tires at the limit; oversteer = two; neutral = four.
- Why oversteery cars/setups are faster: understeer overloads the fronts *and* leaves the rears
  under-used (less front force → less for the rears to react to). Also, fixing understeer costs speed
  (more brake), fixing oversteer adds speed (less brake / a touch of throttle).
- Understeer is useful: a predictable, "on rails" probe. Induce a little at minimum speed to measure
  how much speed the corner allows, then reduce it.

## 3. Weight transfer — three levels
Braking while turning:
1. Too light → a bit more rotation than none, but front under-loaded → understeer (coasting = underload).
2. Ideal → fronts at their sweet spot → the car tends to *oversteer*. If too much, release a touch faster.
3. Excessive → lock/ABS. Front-biased: understeer (straight on if no ABS). Rear-biased: oversteer (spin
   if no ABS).
So understeer can come from **overloading** (ABS) *or* **underloading** (coasting); oversteer usually
means you got the front load right.

Throttle while turning:
1. Below the traction limit → weight to the rear, no rear stress → understeer.
2. At the edge of wheelspin → controllable micro-oversteer. Optimal.
3. Past it → wheelspin, power oversteer, snap.
Where each level sits in throttle % depends on power, gear, grip, speed.

## 4. Three tools of rotation on entry
- Steering, trail braking, engine braking. Steering sets the *direction*; brake and engine braking
  *amplify* rotation. All three must sit on their limit, none over it.
- **To get oversteer, steer less.** A flicked wheel produces understeer, which is why people who try
  to spin can't, and people who aren't trying, do.
- Proportions shift through the corner: at turn-in, high RPM + heavy brake do most of the rotating
  with little steering; toward the MRP they fade and steering takes over. Typical car tendency:
  oversteer-prone early entry, understeer-prone late entry.
- Engine braking (RWD) acts like light rear brakes → a **dynamic final brake bias** set by downshift
  timing. Earlier/faster downshifts → more rear braking → more rotation. FWD: the opposite.
- The *act* of downshifting mid-corner jolts the rears (rev-mismatch micro-lock) and can snowball into
  a spin even if theory says lower RPM = less rotation. Finish the last downshift before turn-in when
  possible; if you must shift mid-corner, expect a micro-slide.
- Downshift **rhythm** must be consistent lap to lap; unconscious variation = random balance.

## 5. Braking
- Find the braking capability first by braking *harder at a safe, early reference*, not by braking later
  and hoping. Then move the reference.
- Get to peak fast but squeeze, don't kick: roughly one blink (~150 ms) to peak for light/high-aero
  cars, about two for heavy/low-aero/soft cars. Slow build-ups smear the reference and kill consistency.
  Decide whether a marker is your *initial* or *peak* reference.
- Low downforce: peak ≈ terminal pressure on a straight-line stop. High downforce: peak is much higher
  than terminal (up to ~2× in extreme cars) — pressure must bleed off as speed and downforce fall.
  Common errors: releasing too much early (under the limit), or not bleeding enough (locking late).
- **Press Less:** release by pressing less with the same muscles, never by lifting the foot. That's
  where release precision comes from.
- Compressions add braking grip (some corners allow ~20 % more pressure); crests remove it — release on
  the crest, re-apply after. Surface patches/bumps can lock tires.
- **Brake hard, light hands**: a death grip steers a degree or two unknowingly and unbalances the car
  laterally, costing braking. The harder the braking, the lighter the hands.
- Brake trace shape: a triangle (big early, tapering). A "hill" (re-applying mid-release) is a bad habit.
- Where there's zero steering mid-sequence (straight bit, direction change), use it: brake harder there.
  Inverted string: unwind the wheel (light hands) as brake goes back on, then string theory again.

## 6. Trail braking and string theory
- Trail braking blends deceleration into rotation without a gap in grip use. Like pouring liquid from
  one cup to another without spilling.
- **String theory:** an imaginary string ties the brake pedal to the steering. Turn slowly → release
  slowly; turn fast → release fast. Practice multiple ratios (stationary drills) so the coupling is
  automatic. Never increase brake and steering together — or brake while holding steering steady.
- **Advanced string theory:** release according to the car's *output* (rotation, wheel feedback, line),
  not the input. Under-rotating → hold brake a little longer and add steering more slowly. Over-rotating
  → release faster. Freedom to steer at different rates; stays closer to the limit.
- In high-aero cars the release has two layers: bleed-off from downforce loss on the straight, then the
  string-theory release into the turn.
- **Three timing modes** at turn-in:
  1. release, then steer → a gap: under the limit, understeer. A ceiling, however consistent it is.
  2. release and steer together → the textbook blend; may rotate a lot depending on setup.
  3. steer slightly before releasing → in ABS cars, the moment you come *off* ABS with a little lock in,
     the car turns hard. Instructive; setup-dependent.
  Modes 2–3 are where pace lives; the non-negotiable is that **some steering exists while the brake is
  releasing**. A fast drop to zero breaks every mode — release length should resemble steering ramp length.
- **Terminal trail brake (T %)**: brake pressure at throttle pickup. ~3–4 % in stiff cars that lift the
  inside front, up to ~20 % in heavy cars; 0 % means you coasted. Lower on off-camber/low grip, higher
  on compressions/banking. Mid-corner baseline ~1–10 %.
- **Make it rotate vs let it rotate:** some cars rotate more as you release (forward bias / easy ABS
  "let the fronts breathe"), others rotate more with brake held. Identify which on lap one. Stiff formula
  cars: too much trail brake "chokes" the fronts.
- Using brake release for balance has a speed side-effect: more brake to fix understeer = lower min speed;
  releasing to fix oversteer = more speed than planned. Fewer, smaller corrections are better.
- Coasting is rare: long corners in cars that already rotate, or cars that lock fronts at any pressure.
- Panic: to add brake, take steering off first.

## 7. Correction priority (keep the fronts working)
Countersteering points the fronts outward → only the rears grip for that instant (~50 % of the car's
grip) and the line goes wide. So:
- **Entry oversteer:** 1) release the brake a bit quicker (induce understeer through brake release —
  fronts still turning in), 2) passive countersteer (light hands, wheel corrects itself), 3) active
  countersteer as the last resort, a few tenths later. Keep adding steering inward as speed falls.
- **Exit oversteer:** 1) ease throttle ~10–20 %, 2) passive countersteer, 3) active.
- **Understeer:** don't add lock. Add front load: a touch of brake or a lift. Throttle off-balance:
  a little throttle is "negative brake" to settle a car that's over-rotating before you commit.
- Principle: lock the steering's job to pulling the car inward; adjust balance with the pedals.
- **Fishing for grip** (oversteery cars): turn gently, feel the rear begin to go, relax force, add more
  as speed drops — force only toward the inside, avoid active countersteer.
- Passive vs active countersteer: passive is physics, instant, and only possible with relaxed hands
  and enough FFB torque (~6 Nm+ makes it easy; ~2 Nm wheels force mostly active).

## 8. Steering
- Steering angle vs steering **force** vs grip on the wheel. Think in force: the wheel's resistance is the
  conversation with the front tires; thinking only in degrees overrides it.
- Light hands = the ability to move the force up and down: less force at turn-in for precision, more
  mid-corner to test grip. Tension hides the effects of brake and engine braking.
- When longitudinal load spikes (hard braking, full throttle in a powerful car), relax the hands.
- **Exponential steering** (low-aero, heavy, V-line, strong engine braking): almost nothing at turn-in,
  building faster as speed drops. **More linear** (high-aero, U-line): later turn-in, quicker ramp.
  With braking → turn in slowly; without braking (on throttle, direction change) → turn in quickly.
- Crest at entry → gentler steering; compression/banking → add faster.
- Common error: a flick at turn-in (locks the inside front, kills engine-braking rotation, mid-corner push).
- Beginners may be told to steer *more* to sit on the front limit while learning pedal precision; experts
  steer *less*. Both are stages, not contradictions.

## 9. Corner stages, MRP and spirals
- **Entry is cause, exit is consequence.** Control is highest before braking, shrinks once braking on the
  limit, and is lowest on late exit.
- Four stages: **early entry** (small lock, oversteer-prone, don't grab rotation yet), **late entry**
  (commit to maximum rotation; hesitation here causes exit spins), **early exit** (from the MRP; no new
  rotation), **late exit** (consequence; drift wide on the limit).
- Decelerating → closing spiral (more rotation available). Accelerating → opening spiral. Peak rotation
  at minimum speed = **MRP** (peak yaw = min speed = throttle point, usually = end of braking).
- **MRL (Maximum Rotation Line)** anti-pattern: fixed yaw and speed for a while, half throttle, scrubbing,
  waiting to feel ready. If the car is decelerating it must be gaining rotation.
- Rotate **at least 50 %** of the corner's heading change before throttle (≈55 % entry / 45 % exit is a
  good split). Throttling earlier pits acceleration against direction.
- MRP vs apex: never accelerate after the apex. Isolated corners — high speed: MRP ~49 % / apex ~51 %;
  medium: ~47/53; low speed: ~45/55. A long straight after → later apex; another corner right after →
  maybe earlier.
- Corners taken already accelerating (second part of an S, kink after a slow corner): the MRP is at the
  turn-in/direction change; don't add lock afterwards while on power.
- **Bouncing off the apex:** car arrives too slow, over-rotates, you straighten, re-turn. Try to bounce
  deliberately — it's the cleanest sign you can carry more speed.
- **Driving the line vs driving the car:** perfect line under the limit feels fast and isn't. Drive the car
  (always ask for max rotation) and let the line show where speed is missing. Unwinding to "use all the
  track" on exit hides unused grip — at the right speed you're *forced* wide by inertia.
- **Rotation first, speed after:** at a slightly safe speed, find the rotation limit (induce understeer,
  then point harder toward oversteer), *then* move the braking point. Speed adjustments on entry come
  from braking only.
- Opening-radius corners: accelerate earlier. Closing-radius: later — the tightest arc limits you.
- Double apex: low-aero → earlier turn-in, more elliptical, more exponential steering, more trail brake,
  wider apex spacing; high-aero → the opposite (can even become single apex).

## 10. Exit and throttle
- Exit is a handoff: unwind a fraction of a second **after** productive throttle begins; never before.
  Unwinding should only ever be a response to the car's balance (under/over), not a goal in itself.
- Throttle = **initial application** (sets balance; a quick kick to the traction limit can add rotation)
  then **modulation** (stay at the edge of wheelspin as speed builds). Low power: straight to a high %;
  high power: fast to the traction limit, then modulate.
- Early unproductive throttle → push → more lock → lift → re-apply (repair spiral).
- Throttle too slow/too little → understeer on power (rears get weight but no stress). Too much → oversteer:
  same speed of application, lower initial %. If it alternates under/over on exit, entry rotation is short.
- A little micro-oversteer on exit (low-power cars) is a sign the mid-corner rotation was right.
- Throttle (not steering) decides where you land on exit; exit track usage comes from inertia.
- FWD: power adds understeer (front "rope"); steer as little as possible on exit; get rotation on entry.
- Anticipate known kicks (upshift into a gear that snaps, crests): pre-correct by easing throttle.
  "Think like the rear tires" — they react late, so act early.

## 11. Line geometry and track
- Entry positioning (where across the track) and angle (where the car points) at braking point, turn-in
  and direction changes. Pointing 3° outward makes a 90° corner a 93° corner.
- Brake in a straight line for max deceleration; a curving brake zone can mean starting angled away.
- **Checkpoints:** the key position/angle point in a sequence (often the next brake zone) to nail every lap.
- **Deceiving corners:** kinks that lure you off the line before the corner that matters — miss the kink.
- Double lefts/rights: the first exit *is* the second entry; use all the track between them.
- **Compound corners:** never go straight between them (except heavy braking). Adjust the second corner
  by moving the first corner's MRP (timing or angle). The **line of consequence** is what entry allows;
  forcing the car open between corners makes it worse.
- What to sacrifice: long straight after → protect the final exit; long → short → sacrifice the long
  one's exit; short → long → protect the short one.
- Elevation/camber: compressions/banking → concentrate rotation and braking there; crests/off-camber →
  drive straighter, brake less. If you can't see the track ahead, it's a crest. Crest on exit → rotate more
  on entry so you're pointed before it. Watch "compression to snap decompression" at the end of dips.
- Use all the track: cockpit view lies; set an in-car alignment reference and check replays.

## 12. Inducing on purpose (learning tools)
- Induce understeer: more steering, less rotation — quicker brake release, fast turn-in, or a little
  throttle (maintenance throttle 5–15 %, rare, e.g. open diffs).
- Induce oversteer on entry: a bit faster than understeer speed, steer normally (no flick) and *less*
  than instinct, keep some brake (~25 % in a skidpad drill) as you start turning, ride the engine braking.
  Too much brake freezes the rotation (all four locked, sliding straight).
- Slide ladder (skidpad, then a real corner): spin on purpose → spin and catch (let the wheel correct) →
  hold a big slide with quick small corrections → start the slide in slow motion and grow it to the MRP
  → fit it onto a line. Use the slide to probe speed: if the slide carries you inside, carry more speed.
- Other mistakes worth causing once: front lock, rear lock, throttle understeer, throttle oversteer.

## 13. Speed of inputs
Smooth where the input is a planned transfer (brake release, turn-in under braking); **fast where it's a
correction**. Correct quickly, then put the wheel straight back where it belongs.

## 14. Car tendencies and setup
- Learn the car's pattern (e.g. "loose on turn-in, pushes before throttle") and pre-adjust in every corner.
- Technique before setup: a setup that *feels* better can be slower (e.g. extra entry rotation overheating
  the rears). Only keep setup changes the delta confirms. Endurance: comfort/predictability can win.
- A skilled driver can make an understeery car rotate and an oversteery car settle with brake release,
  downshift timing, steering and throttle.
