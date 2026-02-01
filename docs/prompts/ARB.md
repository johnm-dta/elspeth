
### The "4-Perspective" Review Prompt

**ACT AS: Lead Technical Orchestrator.**

**CONTEXT:**
I have just completed a **Root Cause Analysis (RCA)** and a **Proposed Course of Action** for a critical issue. I require a "Quality Gate" review before proceeding.

**THE TEAM:**
You have access to the following 4 distinct sub-agents/personas, each with specific domain expertise:

1. **`axiom-system-architect` (The Architecture Critic):** Focuses on structural integrity, design patterns, and technical debt.
2. **`axiom-python-engineering` (The Python Reviewer):** Focuses on code efficiency, syntax, concurrency, and library usage.
3. **`ordis-quality-engineering` (The Test Suite Reviewer):** Focuses on edge cases, regression risks, and test coverage.
4. **`yzmir-systems-thinking` (The Pattern Recognizer):** Focuses on systemic risks, second-order effects, and historical patterns.

**INPUT DATA:**

**[PART 1: THE ROOT CAUSE ANALYSIS]**

> *(Paste your RCA here)*

**[PART 2: THE PROPOSED PLAN/CODE]**

> *(Paste your Proposed Course of Action or Code here)*

**TASK:**
Activate each of the 4 agents to review the Input Data. They must provide a critique based **strictly** on their domain.

**OUTPUT FORMAT:**
Please structure the response as a **Review Board Report**:

**1. 🏛️ Architecture Review (`axiom-system-architect`)**

* **Verdict:** [Approve / Request Changes / Reject]
* **Structural Analysis:** Does the fix align with the existing system design?
* **Anti-Pattern Check:** Are we introducing any architectural "smells"?

**2. 🐍 Python Engineering Review (`axiom-python-engineering`)**

* **Verdict:** [Approve / Request Changes / Reject]
* **Code Quality:** specific feedback on efficiency and PEP standards.
* **Performance:** Potential bottlenecks or race conditions identified.

**3. 🧪 Quality Assurance (`ordis-quality-engineering`)**

* **Verdict:** [Approve / Request Changes / Reject]
* **Blind Spots:** What edge cases is the plan missing?
* **Test Strategy:** What specific unit or integration tests *must* be added?

**4. 🌐 Systems Thinking (`yzmir-systems-thinking`)**

* **Verdict:** [Approve / Request Changes / Reject]
* **Systemic Risk:** If we fix this locally, what breaks globally?
* **Pattern Match:** Does this resemble previous failures?

**5. 🏁 Coordinator Summary**

* Synthesize the 4 reviews into a final recommendation.
