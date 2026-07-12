# Task 2: SaaS Conversion Analysis
**Product:** PolarityIQ / Family Office Intelligence
**Candidate:** [Your Name]

## 1. The Missing Variable: ACV & Trial Friction
Before prescribing tactical UI fixes, the variable that determines whether a 3% conversion rate is a crisis or a baseline metric is missing: **Does the trial require a credit card upfront?**

Recent B2B free-trial conversion data shows the distribution is bimodal. 
* If this is a **no-card opt-in trial**, 3% sits near the natural resting point for unqualified signups — low, but not an anomaly. 
* If a **card is required**, 3% is a genuine emergency. Card-required trials typically convert much higher, with 25–35% considered good. 

This distinction matters more than any UX tactic and is the first metric I would confirm before touching the product.

## 2. Core Hypotheses (Ranked)

**Hypothesis 1: Trust collapses on the first spot-check.**
Anyone using a family office intelligence tool already knows several real family offices personally. The first thing a serious trial user does is search for a name they know to check if the data holds up. One stale or wrong contact field and they quietly conclude the whole dataset cannot be trusted and leave. 
* *Diagnosis:* This shows up in analytics as generic drop-off, which gets misdiagnosed as an onboarding problem when it is actually a verification-visibility problem.
* *Fix:* Show the source and verification method behind every field (even free ones) so trust is earned in seconds instead of assumed.

**Hypothesis 2: The trial user isn't the buyer.**
A high-ACV subscription for fund managers rarely gets bought on a self-serve card swipe by whoever happened to sign up (usually a junior analyst). Without a human sales layer converting an engaged free user into an internal champion, there is a structural ceiling on conversion that no onboarding flow can fix. 

**Hypothesis 3: The free tier is miscalibrated.**
Either the free tier already reveals verified contact data for free (removing the reason to pay), or it reveals so little that nobody sees proof of quality before bouncing. Both produce a 3% conversion rate but require opposite fixes.

## 3. Validation & Immediate Actions
Choosing between these hypotheses without data is a guess dressed up as a decision. Before spending engineering cycles, I would validate:
1. Event-level data on where free users drop relative to the paywall.
2. Direct interviews with non-converting trial users.

**If forced to act today without that data:**
1. **Address Hypothesis 1:** Expose verification method and source on every field, including free ones. Trust is this product's actual currency.
2. **Address Hypothesis 2:** Flag any trial user running more than 2-3 queries as a high-intent signal for immediate manual outreach, rather than dropping them into an automated email sequence.