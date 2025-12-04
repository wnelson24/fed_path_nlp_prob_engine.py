# fed_path_nlp_prob_engine.py


The engine converts futures prices into **implied meeting rates**, then solves  
for the probability of a **25bp hike vs hold vs cut** using a two-state model.

Resulting structure:

- `prob_hike_25bp`  
- `prob_cut_25bp`  
- implied meeting rate  
- expected policy rate after next FOMC

---

### 2. OIS Curve & Terminal Rate Estimation

Using short-dated OIS points:

- 1M  
- 3M  
- 6M  
- 1Y  
- 2Y  
- 5Y  

The engine calculates:

- **implied terminal rate** (1Y/2Y midpoint proxy)  
- curvature of the forward path  
- consistency with Fed Funds futures  

This gives a baseline **policy path** prior to incorporating NLP shocks.

---

### 3. NLP Headline Shock Detection

"Fed" OR "Powell" OR "FOMC"

Then:

1. Pulls full story text  
2. Vectorizes with TF-IDF  
3. Runs a logistic classifier (placeholder, upgradeable)  
4. Produces a **hawkish/dovish score** in the range:


-0.50 (dovish) → +0.50 (hawkish)



This score represents the **directional surprise** implied by Fed-linked news.

You can upgrade this easily with:

- FinBERT  
- RoBERTa  
- a labeled hawkish/dovish dataset  
- rule-based central-bank sentiment scoring  

---

### 4. Shock-to-Repricing Model

The final stage maps the NLP score into:

- new hike probability  
- new cut-path probability  
- new terminal rate  
- repricing amplitude



Example mapping:

Repriced Hike Prob = old_prob_hike + α * hawkishness
Repriced Terminal = old_terminal + β * hawkishness



This gives a **flow-adjusted view** of policy expectations.

---

## Example Output

--- FED PATH PROBABILITY & NLP SHOCK ENGINE ---
Implied Meeting Rate: 5.290%
Prob Hike 25bp: 12.0%
Prob Cut 25bp: 88.0%
Terminal Rate (base): 4.930%
Hawkishness Score: +0.172
--- NLP-Adjusted Repricing ---
Repriced Hike Prob: 23.4%
Repriced Cut Prob: 76.6%
Repriced Terminal: 5.102%
--- END ---



---

## How to Interpret the Output

### **1. Implied Meeting Rate**
Derived from nearest Fed Funds future.  
Represents market expectation before shock.

### **2. Probabilities (Hike / Cut)**
A compact view of **path skew**:

- High `prob_cut_25bp` → easing bias  
- High `prob_hike_25bp` → tightening bias  
- Balanced → uncertainty / event risk  

### **3. Terminal Rate**
Core macro anchor for:

- FX volatility  
- rates convexity  
- risk assets  
- forward curve pricing  

### **4. Hawkishness Score**
NLP-inferred tone from Fed headlines:

- > 0 → hawkish  
- < 0 → dovish  
- magnitude = strength of signal  

### **5. NLP-Adjusted Repricing**
How the market **should** reprice if headlines influence:

- policy expectations  
- terminal rate  
- cut-path probability  

This is especially useful around:

- CPI  
- NFP  
- FOMC  
- Powell speeches  
- Fed Minutes  
- geopolitical shocks  

---

## How Traders Use This

### **FX Options**
- interpret why RR or ATM vols move around macro events  
- understand event premium build-up  
- link cut-path pricing to vol demand  

### **Rates / STIR**
- detect path vs terminal vs slope repricing  
- identify flow-driven intraday moves  
- measure data vs comms shock impact  

### **Macro Trading**
- attribute curve moves by cause  
- model cycle asymmetry  
- monitor forward expectations in real time  

---

## Running the Model

### 1. Insert your Eikon App Key
```python
ek.set_app_key("YOUR_APP_KEY_HERE")

