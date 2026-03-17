# Fuzzy Logic Exam Study Sheet

Based on Lab 4, Lab 5, Lab 6, and Tsukamoto Self-Study

---

## SECTION 1: Membership Functions (Lab 4)

### 1.1 — Creating MFs with scikit-fuzzy

```python
# Triangular: trimf(universe, [left_foot, peak, right_foot])
mf = fuzz.trimf(x, [25, 50, 75])

# Trapezoidal: trapmf(universe, [left_foot, left_shoulder, right_shoulder, right_foot])
mf = fuzz.trapmf(x, [0, 0, 30, 45])

# Gaussian: gaussmf(universe, mean, sigma)
mf = fuzz.gaussmf(x, 5, 1)

# Sigmoid: sigmf(universe, centre, slope)
mf = fuzz.sigmf(x, 5, 2)
```

### 1.2 — Triangular MF Formula (VERY LIKELY exam question: fill in missing branch)

```python
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0               # outside the triangle
    elif x <= b:
        return (x - a) / (b - a) # ascending slope (left side)
    else:
        return (c - x) / (c - b) # descending slope (right side)  ← LIKELY MISSING LINE
```

### 1.3 — Trapezoidal MF Formula (fill in missing branch)

```python
def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0               # outside
    elif x <= b:
        return (x - a) / (b - a) # ascending slope
    elif x <= c:
        return 1.0               # flat top (plateau)  ← LIKELY MISSING LINE
    else:
        return (d - x) / (d - c) # descending slope
```

### 1.4 — Gaussian MF Formula

```python
return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
```

### 1.5 — Sigmoid MF Formula

```python
return 1.0 / (1.0 + np.exp(-a * (x - c)))
```

### 1.6 — Fuzzification (compute membership degree of a crisp value)

```python
mu = fuzz.interp_membership(x_wind, wind_calm, 8.0)
```

---

## SECTION 2: Hedges (Lab 4)

### 2.1 — All Hedge Formulas (MEMORISE — exam will ask you to compute one)

| Hedge          | Formula                | Exponent | Type          |
|----------------|------------------------|----------|---------------|
| **Very**       | `mu ** 2`              | 2        | Concentration |
| **Extremely**  | `mu ** 3`              | 3        | Concentration |
| **Very Very**  | `mu ** 4`              | 4        | Concentration |
| **Slightly**   | `mu ** 1.7`            | 1.7      | Concentration |
| **A Little**   | `mu ** 1.3`            | 1.3      | Concentration |
| **More or Less** | `mu ** 0.5` (√μ)    | 0.5      | Dilation      |
| **Somewhat**   | `mu ** (1/3)` (∛μ)     | 1/3      | Dilation      |
| **NOT**        | `1 - mu`               | —        | Complement    |
| **Indeed**     | `2*mu² if mu≤0.5; 1-2*(1-mu)² if mu>0.5` | — | Mixed |

### 2.2 — Key hedge computations to memorise

```
μ = 0.86:  very = 0.7396,  extremely = 0.6361,  very_very = 0.5470,  more_or_less = 0.9274
μ = 0.5:   very = 0.25,    extremely = 0.125,   slightly = 0.3078
μ = 0.3:   very = 0.09
μ = 0.7:   more_or_less = 0.8367
```

### 2.3 — Implement hedges from scratch

```python
def hedge_very(mu):        return mu ** 2
def hedge_extremely(mu):   return mu ** 3
def hedge_very_very(mu):   return mu ** 4
def hedge_slightly(mu):    return mu ** 1.7
def hedge_a_little(mu):    return mu ** 1.3
def hedge_more_or_less(mu):return mu ** 0.5
def hedge_somewhat(mu):    return mu ** (1/3)
def hedge_not(mu):         return 1 - mu
def hedge_indeed(mu):
    return np.where(mu <= 0.5, 2 * mu**2, 1 - 2*(1 - mu)**2)
```

**Key concept:** Concentration (exponent > 1) → narrows/reduces membership. Dilation (exponent < 1) → widens/increases membership.

---

## SECTION 3: Fuzzy Set Operations (Lab 4)

### 3.1 — The Four Operations

```python
fuzzy_AND    = min(mu_a, mu_b)              # AND = minimum (intersection)
fuzzy_OR     = max(mu_a, mu_b)              # OR  = maximum (union)
fuzzy_NOT    = 1 - mu_a                     # NOT = complement
prob_OR      = mu_a + mu_b - mu_a * mu_b    # Probabilistic OR (algebraic sum)
```

### 3.2 — Array versions (for plotting)

```python
and_result = np.minimum(short, average)     # element-wise AND
or_result  = np.maximum(short, tall)        # element-wise OR
not_result = 1 - average                    # element-wise NOT
probor     = short + tall - short * tall    # element-wise prob OR
```

**Key fact:** When both values are non-zero, prob OR ≥ max OR. When one is zero, they are equal.

---

## SECTION 4: Mamdani Inference — From Scratch (Lab 5)

### The 5 Steps of Mamdani Inference

**Step 1 — Fuzzification:** Evaluate each input against all its MFs
```python
mu_inadequate = fuzz.interp_membership(x_fund, fund_inadequate, 35)
```

**Step 2 — Rule Evaluation:** Apply fuzzy operators to get firing strengths
```python
# Rule: IF funding IS adequate OR staffing IS small THEN risk IS low
rule1_strength = max(mu_adequate, mu_small)    # OR = max
# Rule: IF funding IS marginal AND staffing IS large THEN risk IS normal
rule2_strength = min(mu_marginal, mu_large)    # AND = min
```

**Step 3 — Implication (Clipping):** Clip each consequent MF at its rule's firing strength
```python
rule1_output = np.minimum(rule1_strength, risk_low)     # min-implication (clip)
rule2_output = np.minimum(rule2_strength, risk_normal)
```

**Step 4 — Aggregation:** Combine all clipped MFs using max
```python
aggregated = np.maximum(rule1_output, np.maximum(rule2_output, rule3_output))
```

**Step 5 — Defuzzification:** Convert the aggregated set to a crisp value

### 4.1 — Defuzzification Methods (Lab 5)

```python
# COG — Centre of Gravity (most common)
cog = np.sum(x * mf) / np.sum(mf)

# MOM — Mean of Maxima
max_val = np.max(mf)
max_indices = np.where(np.isclose(mf, max_val))[0]
mom = np.mean(x[max_indices])

# SOM — Smallest of Maxima
som = x[max_indices[0]]

# LOM — Largest of Maxima
lom = x[max_indices[-1]]
```

---

## SECTION 5: The Dapping Worked Example with Hedges (Lab 4 Exercise 5)

**Given:** μ_stormy(8) = 0.5, μ_fresh(8) = 0.38, μ_low(10) = 0.3, μ_average(10) = 0.7

### Rule 1: IF wind IS extremely stormy OR temp IS very low THEN dapping IS not very poor

```python
extremely_stormy = 0.5 ** 3        # = 0.125
very_low         = 0.3 ** 2        # = 0.09
rule1_strength   = max(0.125, 0.09)  # OR = 0.125

# Consequent hedge: "not very" applied to firing strength
very_rule1       = 0.125 ** 2      # = 0.0156
not_very_rule1   = 1 - 0.0156     # = 0.9844   ← clip 'poor' at this value
```

### Rule 2: IF wind IS fresh AND temp IS more_or_less average THEN dapping IS mediocre

```python
mol_average      = 0.7 ** 0.5      # = 0.8367  (√0.7)
rule2_strength   = min(0.38, 0.8367)  # AND = 0.38   ← clip 'mediocre' at 0.38
```

### Rule 3: IF wind IS slightly stormy AND temp IS NOT low THEN dapping IS a_little excellent

```python
slightly_stormy  = 0.5 ** 1.7      # = 0.3078
not_low          = 1 - 0.3         # = 0.7
rule3_antecedent = min(0.3078, 0.7)  # AND = 0.3078

# Consequent hedge: "a_little"
a_little_rule3   = 0.3078 ** 1.3   # ≈ 0.215   ← clip 'excellent' at this value
```

---

## SECTION 6: pyfuzzylite — Mamdani Engine (Lab 5 + Lab 6)

### 6.1 — Define InputVariable

```python
fl.InputVariable(
    name="speed_error", minimum=-30, maximum=30,
    terms=[
        fl.Trapezoid("large_negative", -30, -30, -15, -5),
        fl.Triangle("small_negative", -10, -5, 0),
        fl.Triangle("zero", -5, 0, 5),
        fl.Triangle("small_positive", 0, 5, 10),
        fl.Trapezoid("large_positive", 5, 15, 30, 30),
    ],
)
```

### 6.2 — Define OutputVariable (Mamdani)

```python
fl.OutputVariable(
    name="throttle", minimum=-50, maximum=50,
    aggregation=fl.Maximum(),         # aggregation method
    defuzzifier=fl.Centroid(200),     # defuzzification method
    terms=[
        fl.Triangle("maintain", -5, 0, 5),
        fl.Trapezoid("accelerate_hard", 10, 30, 50, 50),
    ],
)
```

### 6.3 — Define RuleBlock (Mamdani)

```python
fl.RuleBlock(
    name="rules",
    conjunction=fl.Minimum(),     # AND = min
    disjunction=fl.Maximum(),     # OR  = max
    implication=fl.Minimum(),     # Mamdani: clip (min-implication)
    activation=fl.General(),
    rules=[
        fl.Rule.create("if speed_error is large_positive then throttle is accelerate_hard"),
        fl.Rule.create("if speed_error is zero and acceleration is steady then throttle is maintain"),
        fl.Rule.create("if funding is adequate or staffing is small then risk is low"),
    ],
)
```

### 6.4 — Run Engine and Get Output

```python
engine.input_variable("speed_error").value = 20.0
engine.input_variable("acceleration").value = 0.0
engine.process()
result = engine.output_variable("throttle").value
```

### 6.5 — Register Custom Hedges in pyfuzzylite

```python
from fuzzylite import settings
settings.factory_manager.hedge.constructors["slightly"] = \
    lambda: fl.HedgeLambda("slightly", lambda x: x**1.7)
settings.factory_manager.hedge.constructors["more_or_less"] = \
    lambda: fl.HedgeLambda("more_or_less", lambda x: x**0.5)
settings.factory_manager.hedge.constructors["extremely"] = \
    lambda: fl.HedgeLambda("extremely", lambda x: x**3)
```

### 6.6 — Switching Defuzzifiers

```python
engine.output_variable("risk").defuzzifier = fl.Centroid(500)
engine.output_variable("risk").defuzzifier = fl.MeanOfMaximum(500)
engine.output_variable("risk").defuzzifier = fl.SmallestOfMaximum(500)
engine.output_variable("risk").defuzzifier = fl.LargestOfMaximum(500)
engine.output_variable("risk").defuzzifier = fl.Bisector(500)
```

---

## SECTION 7: Sugeno Inference (Lab 5)

### 7.1 — Key Differences from Mamdani

| Aspect              | Mamdani                         | Sugeno                          |
|---------------------|----------------------------------|----------------------------------|
| Output terms        | Fuzzy sets (Triangle, Trapezoid) | `fl.Constant("name", value)`     |
| Defuzzifier         | `fl.Centroid(200)`              | `fl.WeightedAverage("TakagiSugeno")` |
| Aggregation         | `fl.Maximum()`                  | `None` (or fl.Maximum())         |
| Implication         | `fl.Minimum()`                  | `None` (or fl.AlgebraicProduct())|
| Defuzzification     | Area-based (COG, MOM, etc.)     | Weighted average (algebraic)     |

### 7.2 — Sugeno OutputVariable

```python
fl.OutputVariable(
    name="dapping", minimum=0.0, maximum=100.0,
    aggregation=None,
    defuzzifier=fl.WeightedAverage("TakagiSugeno"),
    terms=[
        fl.Constant("poor", 25),       # singleton at 25
        fl.Constant("mediocre", 50),    # singleton at 50
        fl.Constant("excellent", 75),   # singleton at 75
    ],
)
```

### 7.3 — Sugeno Weighted Average Formula

$$z^* = \frac{\sum w_i \cdot z_i}{\sum w_i} = \frac{w_1 \cdot z_1 + w_2 \cdot z_2 + w_3 \cdot z_3}{w_1 + w_2 + w_3}$$

```python
result = (w1 * z1 + w2 * z2 + w3 * z3) / (w1 + w2 + w3)
```

### 7.4 — Full Sugeno Manual Calculation Example (wind=8, temp=10)

```python
# Rule 1: extremely stormy OR very low → poor (z=25)
w1 = max(0.5**3, 0.3**2)              # max(0.125, 0.09) = 0.125

# Rule 2: fresh AND more_or_less avg → mediocre (z=50)
w2 = min(0.38, 0.7**0.5)              # min(0.38, 0.8367) = 0.38

# Rule 3: slightly stormy AND NOT low → a_little excellent
w3 = (min(0.5**1.7, 1-0.3)) ** 1.3   # min(0.3078, 0.7)^1.3 ≈ 0.215

tip = (0.125*25 + 0.38*50 + 0.215*75) / (0.125 + 0.38 + 0.215)
```

---

## SECTION 8: Tsukamoto Inference (Self-Study Notebook)

### 8.1 — Key Concept: Monotonic Output MFs + Inversion

Tsukamoto requires **monotonic** output MFs (always increasing or always decreasing).
Given firing strength $w_i$, find $z_i$ by **inverting** the MF: $\mu(z_i) = w_i$

### 8.2 — Ramp Functions (Tsukamoto output MFs)

```python
# Ascending ramp: 0 at x<=a, 1 at x>=b
def ramp_ascending(x, a, b):
    return np.clip((x - a) / (b - a), 0, 1)

# Descending ramp: 1 at x<=a, 0 at x>=b
def ramp_descending(x, a, b):
    return np.clip((b - x) / (b - a), 0, 1)
```

### 8.3 — Inverse Functions (EXAM LIKELY: "write the inverse")

```python
# Ascending ramp from a to b:  mu = (z - a)/(b - a)  →  z = a + (b - a)*w
def inv_ascending(w, a, b):
    return a + (b - a) * w

# Descending ramp from a to b: mu = (b - z)/(b - a)  →  z = b - (b - a)*w
def inv_descending(w, a, b):
    return b - (b - a) * w
```

**Tipper example inverses:**
```python
def inv_cheap(w):     return 12 - 7 * w      # descending ramp (5→12), z = 12 - 7w
def inv_average(w):   return 10 + 8 * w      # ascending ramp (10→18), z = 10 + 8w
def inv_generous(w):  return 18 + 7 * w      # ascending ramp (18→25), z = 18 + 7w
```

### 8.4 — Tsukamoto OutputVariable in pyfuzzylite

```python
fl.OutputVariable(
    name="tip", minimum=5, maximum=25,
    aggregation=None,
    defuzzifier=fl.WeightedAverage("Tsukamoto"),
    terms=[
        fl.Ramp("cheap", 12, 5),       # descending: start=12, end=5 (high at 5)
        fl.Ramp("average", 10, 18),     # ascending: start=10, end=18
        fl.Ramp("generous", 18, 25),    # ascending: start=18, end=25
    ],
)
```

### 8.5 — Full Tsukamoto Step-by-Step (service=3, food=8)

```python
# Step 1: Fuzzify
mu_poor = gaussmf(3, 1.5, 0)       # ≈ 0.1353
mu_good = gaussmf(3, 1.5, 5)       # ≈ 0.3247
mu_excellent = gaussmf(3, 1.5, 10) # ≈ 0.0000
mu_rancid = trapmf(8, 0,0,1,3)     # = 0.0
mu_delicious = trapmf(8, 7,9,10,10)# = 0.5

# Step 2: Rule firing strengths
w1 = max(mu_poor, mu_rancid)       # max(0.1353, 0) = 0.1353  → cheap
w2 = mu_good                       # 0.3247                    → average
w3 = max(mu_excellent, mu_delicious) # max(0, 0.5) = 0.5       → generous

# Step 3: Invert output MFs (TSUKAMOTO-SPECIFIC)
z1 = 12 - 7 * w1    # inv_cheap(0.1353)   = 11.053
z2 = 10 + 8 * w2    # inv_average(0.3247) = 12.598
z3 = 18 + 7 * w3    # inv_generous(0.5)   = 21.5

# Step 4: Weighted average (same formula as Sugeno)
tip = (w1*z1 + w2*z2 + w3*z3) / (w1 + w2 + w3)
```

### 8.6 — Why Monotonic? (Conceptual question)

If the MF is a **triangle** (non-monotonic), the equation μ(z) = w has **two solutions** — the inversion is ambiguous. Ramps/sigmoids are monotonic → exactly **one** solution → unambiguous.

---

## SECTION 9: Comparing All Three Models (Lab 5 Part 4)

### 9.1 — Side-by-Side Comparison Table

| Aspect            | Mamdani                          | Sugeno                           | Tsukamoto                        |
|-------------------|----------------------------------|----------------------------------|----------------------------------|
| **Output MFs**    | Fuzzy sets (Triangle, Trap)      | Constants (singletons)           | Monotonic (Ramp, Sigmoid)        |
| **Rule output**   | Clipped fuzzy set                | Crisp value (constant)           | Crisp value (by MF inversion)    |
| **Aggregation**   | Union (max) of fuzzy sets        | Weighted average                 | Weighted average                 |
| **Defuzzification**| COG, MOM, SOM, LOM, Bisector   | Not needed (built-in WA)         | Not needed (built-in WA)         |
| **Computation**   | Heavy (area-based)               | Light (algebraic)                | Moderate (inversion step)        |
| **Interpretability**| HIGH ★★★                       | Medium ★★                        | Medium ★★                        |
| **pfl defuzzifier** | `fl.Centroid(200)`             | `fl.WeightedAverage("TakagiSugeno")` | `fl.WeightedAverage("Tsukamoto")` |
| **pfl output term** | `fl.Triangle(...)` etc.        | `fl.Constant("name", val)`       | `fl.Ramp("name", start, end)`    |
| **pfl implication** | `fl.Minimum()`                 | `None` or `fl.AlgebraicProduct()` | `fl.AlgebraicProduct()`         |

---

## SECTION 10: Fuzzy Control Loop (Lab 6)

### 10.1 — Closed-Loop Simulation Steps

```python
for each time step:
    1. speed_error = target_speed - current_speed       # compute error
    2. acceleration = plant.get_acceleration()           # read sensor
    3. se_clamped = np.clip(speed_error, -30, 30)       # clamp to input range
    4. engine.input_variable("speed_error").value = se_clamped
       engine.input_variable("acceleration").value = ac_clamped
    5. engine.process()                                  # fuzzy inference
    6. throttle = engine.output_variable("throttle").value
    7. new_speed = plant.step(throttle, disturbance)     # apply to plant
```

### 10.2 — Plant Physics (Newton's second law)

```python
speed_ms = self.speed / 3.6                        # km/h → m/s
throttle_force = throttle * self.force_gain         # controller output → Newtons
drag_force = self.drag_coeff * speed_ms ** 2        # aerodynamic drag
net_force = throttle_force - drag_force - disturbance
acceleration = net_force / self.mass                # F = ma → a = F/m
speed_change_kmh = acceleration * self.dt * 3.6     # m/s² → km/h change
self.speed = max(0.0, self.speed + speed_change_kmh)
```

### 10.3 — Steady-State Error (Conceptual)

**Why does the car never reach exactly 100 km/h?**
- When error enters the "zero" region, controller outputs "maintain" (≈0 throttle)
- But drag requires non-zero throttle to hold speed
- No **integral action** → can't accumulate past errors
- This is a limitation of proportional-only fuzzy controllers

**Fixes:**
1. Shift "maintain" output MF positive: `Triangle("maintain", -2, 3, 8)` instead of `(-5, 0, 5)`
2. Add integral term (accumulate error like PID)
3. Expand "zero" rules: add "if error is zero AND accel is braking → accelerate_light"

---

## SECTION 11: Quick-Reference — Most Likely "Write 1-2 Lines" Questions

| Question Type                              | Answer                                                  |
|--------------------------------------------|---------------------------------------------------------|
| Complete the triangle MF (else branch)     | `return (c - x) / (c - b)`                             |
| Complete the trapezoid MF (plateau)        | `return 1.0`                                           |
| Compute "very tall" for μ=0.86             | `0.86 ** 2 = 0.7396`                                   |
| Compute "extremely stormy" for μ=0.5       | `0.5 ** 3 = 0.125`                                     |
| Compute "more or less average" for μ=0.7   | `0.7 ** 0.5 = 0.8367`                                  |
| Compute "NOT low" for μ=0.3               | `1 - 0.3 = 0.7`                                        |
| AND two values                             | `min(mu_a, mu_b)`                                       |
| OR two values                              | `max(mu_a, mu_b)`                                       |
| Clip a consequent MF                       | `np.minimum(firing_strength, output_mf)`                |
| Aggregate rule outputs                     | `np.maximum(r1_out, r2_out)`                            |
| COG defuzzification                        | `np.sum(x * mf) / np.sum(mf)`                          |
| Sugeno/Tsukamoto weighted average          | `(w1*z1 + w2*z2 + w3*z3) / (w1 + w2 + w3)`            |
| Fuzzify a crisp value                      | `fuzz.interp_membership(universe, mf_array, crisp_val)` |
| Write a fuzzy rule (pyfuzzylite)           | `fl.Rule.create("if X is A and Y is B then Z is C")`   |
| Run the engine                             | `engine.input_variable("x").value = v; engine.process()`|
| Sugeno constant term                       | `fl.Constant("poor", 25)`                              |
| Tsukamoto ramp term                        | `fl.Ramp("cheap", 12, 5)`                              |
| Invert ascending ramp (a→b)               | `z = a + (b - a) * w`                                   |
| Invert descending ramp (a→b)              | `z = b - (b - a) * w`  or equivalently `z = a - (a-b)*w`|
| Probabilistic OR                           | `a + b - a*b`                                           |
| Register a custom hedge                    | `settings.factory_manager.hedge.constructors["name"] = lambda: fl.HedgeLambda("name", lambda x: x**p)` |
| MOM defuzzification                        | Find all x where μ is max, take their mean              |
| SOM defuzzification                        | Smallest x where μ is max                               |
| LOM defuzzification                        | Largest x where μ is max                                |

---

## SECTION 12: pyfuzzylite Term Types Summary

```python
# Input terms (used by ALL three models)
fl.Triangle("name", left, peak, right)
fl.Trapezoid("name", a, b, c, d)
fl.Gaussian("name", mean, sigma)
fl.Ramp("name", start, end)        # also used as Tsukamoto output

# Mamdani output terms (fuzzy sets)
fl.Triangle("maintain", -5, 0, 5)
fl.Trapezoid("brake_hard", -50, -50, -30, -10)

# Sugeno output terms (singletons)
fl.Constant("poor", 25)

# Tsukamoto output terms (monotonic)
fl.Ramp("cheap", 12, 5)            # descending: mu=1 near 5, mu=0 near 12
fl.Ramp("generous", 18, 25)        # ascending: mu=0 at 18, mu=1 at 25
```

---

## SECTION 13: Formula Quick-Reference Card

$$\text{Triangle: } \mu(x;a,b,c) = \max\left(\min\left(\frac{x-a}{b-a},\;\frac{c-x}{c-b}\right),\;0\right)$$

$$\text{Hedge: } \mu^{hedged} = [\mu]^p$$

$$\text{AND: } \min(\mu_A, \mu_B) \qquad \text{OR: } \max(\mu_A, \mu_B) \qquad \text{NOT: } 1 - \mu_A$$

$$\text{Prob OR: } \mu_A + \mu_B - \mu_A \cdot \mu_B$$

$$\text{COG: } z^* = \frac{\sum z \cdot \mu(z)}{\sum \mu(z)}$$

$$\text{Sugeno/Tsukamoto WA: } z^* = \frac{\sum w_i \cdot z_i}{\sum w_i}$$

$$\text{Tsukamoto inversion (ascending ramp a→b): } z = a + (b-a) \cdot w$$

$$\text{Tsukamoto inversion (descending ramp a→b): } z = b - (b-a) \cdot w$$
