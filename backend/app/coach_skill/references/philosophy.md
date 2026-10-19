# Core philosophy (distilled, in our own words)

These are the ideas every coaching answer must stay consistent with. They are a synthesis of
Suellio Almeida's public teaching (Almeida Racing Academy lessons, blog, interviews) rewritten for
an engineering coach. Where we add our own engineering framing it is marked *(engineering note)*.

## 1. Driving is managing four contact patches
The driver's whole job is to find the grip limit of each tire quickly and stay on it for as much of
the lap as possible. Every technique — trail braking, engine braking, steering shape, throttle
shape — exists for one reason: to keep all four tires near their **optimal slip angle** for the
whole corner.

## 2. "The limit" is per tire, and has two sides
The limit of a tire is the most force it can make to change the car's speed or direction.
- Front tires at the limit first → understeer.
- Rear tires at the limit first → oversteer.
- Longitudinal limit → lockup / ABS / wheelspin / TC.
Reframe for drivers: **understeer = under-rotation, oversteer = over-rotation** relative to what
you asked for. It's a yaw-rate expectation problem, not a vibe.

## 3. If you can cause it, you can prevent it
Predictability comes from deliberately producing the thing you fear. Induce understeer on purpose
(safe, predictable half of the limit) to learn the front; induce oversteer on purpose (on entry,
with a little brake and *less* steering) to learn the rear. A driver who can make the car slide on
demand can stop it sliding on demand. This is the backbone of "controlled over-driving".

## 4. Rotation has three sources on entry
Steering, trail braking (brake release), engine braking (downshifts / RPM). Steering is the
*last* layer, not the first. Early in the corner most rotation comes from brake and engine braking
with very little steering; deeper in, as RPM and brake fade, steering takes over. Trying to rotate
with steering alone overloads the fronts — that's why people who try to spin a car can't, and
people who aren't trying, do.

## 5. Rotation and speed are inversely related
Maximum rotation happens at minimum speed. That point (the MRP) splits every corner into a
closing spiral (more steering, more rotation, less speed, still braking) and an opening spiral
(less steering, less rotation, more speed, on throttle). Asking for more rotation while the car is
gaining speed puts it over the limit — snap oversteer or a push.

## 6. Smooth is fast — but smooth means *rate*, not *slowness*
Brake application should be aggressive and early to peak; release is progressive and tied to
steering. Inputs fail when their rate exceeds what the tire can follow, not when they're "big".
In low grip (rain, cold tires) watch input rate more than input magnitude.

## 7. Light hands
Hold the wheel loosely enough that force feedback can move it. The wheel tells you front slip
angle (resistance builds toward peak, then goes light past it). Relaxed hands let the car find its
own neutral steer, let passive countersteer happen with zero delay, and let the steering unwind
itself when you go back on the brakes.

## 8. Technique before setup
A car that "feels better" after a setup change may only be hiding a technique problem (e.g. extra
aero masking mechanical oversteer created by trail braking). Separate driver-induced rotation from
car balance before touching the setup.

## 9. Learning like a musician
Deliberate, single-variable practice. Break a skill into building blocks, practice each until it's
automatic, then forget about it and let it run underneath. Ten focused laps on one variable beat
fifty laps of "just driving". Self-critique hard, but on one thing at a time.

## 10. Consistency first, then speed
Find the limit from the safe side (minimum-speed understeer), then step toward oversteer in small
increments. Speed you can't repeat isn't pace yet.

## 11. Entry is cause, exit is consequence
Control is greatest before braking and smallest on late exit. Most exit problems are entry problems.

## 12. Four pillars
Feeling understeer, feeling oversteer, inducing understeer, inducing oversteer. Neutral steer is what
you get when all four are fast and small enough to blend.

## 13. Efficiency and root causes
A driver can only consciously hold one or two things. Find the single cue upstream that makes several
other inputs fall into place, rather than a list of corrections.

## 14. Coach, don't instruct
The aim is an autonomous driver who understands why, not one who follows marker instructions.

*(engineering note)* For lmu-coach this translates directly: the probabilistic tire model's peak
slip angle is our measured "optimal slip angle"; grip utilization is how close each axle sits to it;
yaw rate vs. steering is how we measure under/over-rotation; the MRP is found from speed, yaw rate,
steering and pedal crossover.
