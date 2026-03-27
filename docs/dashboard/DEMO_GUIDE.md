# Demo Guide - FL Security Dashboard

This guide helps you prepare and deliver effective demonstrations of the FL Security Dashboard for presentations, conferences, and academic reviews.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Demo Scenarios](#demo-scenarios)
3. [Presentation Script](#presentation-script)
4. [Live Demonstration](#live-demonstration)
5. [Common Questions](#common-questions)
6. [Tips for Success](#tips-for-success)

---

## Quick Start

### 1. Start the Dashboard

```bash
cd /home/ubuntu/30Days_Project/fl_security_dashboard

# Install dependencies (first time only)
pip install streamlit plotly numpy pandas

# Run dashboard
streamlit run app/main.py
```

### 2. Load Demo Data

1. Navigate to **Client Analytics** page
2. Click "Load Demo Data" in the sidebar
3. Select a scenario (e.g., "Label Flipping Attack")
4. Click "Load Scenario"

### 3. Navigate Pages

- **⚡ Training Monitor**: Real-time training progress
- **👥 Client Analytics**: Per-client metrics and reputations
- **🛡️ Security Status**: Attack detection timeline
- **🔐 Privacy Budget**: DP epsilon tracking
- **📊 Experiment Comparison**: Compare scenarios

---

## Demo Scenarios

### Scenario 1: Normal Training

**Purpose**: Show baseline FL training behavior

**Steps**:
1. Load "Normal Training" scenario
2. Navigate to **Training Monitor**
3. Highlight:
   - Accuracy improving from 10% → 90%
   - Loss decreasing from 2.5 → 0.1
   - All clients with "active" status
   - No security events

**Key Talking Points**:
- "Federated learning enables collaborative model training"
- "Each client trains locally and only shares model updates"
- "Global model accuracy improves over rounds"

---

### Scenario 2: Label Flipping Attack

**Purpose**: Demonstrate poisoning attack and detection

**Steps**:
1. Load "Label Flipping Attack" scenario
2. Navigate to **Training Monitor**
3. Observe rounds 10-25 (attack period):
   - Accuracy plateaus or drops
   - Loss increases
4. Switch to **Security Status**:
   - See attack detection events
   - Highlight attackers (clients 8, 9)
5. Go to **Client Analytics**:
   - Show low reputations for attackers
   - High anomaly scores

**Key Talking Points**:
- "Malicious clients flip labels to poison the model"
- "SignGuard detects anomalous update patterns"
- "Reputation system downweights untrusted clients"

---

### Scenario 3: SignGuard Defense

**Purpose**: Show defense mechanism effectiveness

**Steps**:
1. Load "SignGuard Defense" scenario
2. Navigate to **Security Status**
3. Show defense activations:
   - "SignGuard downweighted Client 8"
   - "SignGuard downweighted Client 9"
4. Compare to **Training Monitor**:
   - Model still converges despite attack
   - Accuracy reaches ~85% (vs ~75% without defense)

**Key Talking Points**:
- "SignGuard analyzes sign patterns in updates"
- "Attackers have inconsistent sign patterns"
- "Defense maintains model utility while blocking attacks"

---

### Scenario 4: Byzantine Attack

**Purpose**: Show sophisticated multi-client attack

**Steps**:
1. Load "Byzantine Attack" scenario
2. Navigate to **Training Monitor**
3. Show severe performance degradation:
   - Accuracy stuck at ~60%
   - Loss fails to decrease
4. **Security Status**:
   - Multiple high-severity events
   - 3 attacking clients detected
5. **Client Analytics**:
   - Show attacker clustering
   - All attackers have reputation < 0.3

**Key Talking Points**:
- "Byzantine attackers coordinate to disrupt convergence"
- "FoolsGold detects attackers by update similarity"
- "Multiple defenses can be layered for robust protection"

---

### Scenario 5: Backdoor Attack

**Purpose**: Show stealthy, hard-to-detect attack

**Steps**:
1. Load "Backdoor Attack" scenario
2. **Training Monitor**:
   - Model appears to train normally
   - Accuracy reaches ~88%
3. **Security Status**:
   - Fewer detection events (stealthy)
   - Lower confidence detections
4. Explain:
   - "Backdoor maintains normal performance"
   - "Creates hidden vulnerability in model"
   - "Harder to detect but still possible"

**Key Talking Points**:
- "Backdoor attacks are stealthy"
- "Inject specific patterns without affecting normal behavior"
- "Require specialized detection techniques"

---

## Presentation Script

### Introduction (2 minutes)

"Today I'll demonstrate the FL Security Dashboard, a real-time monitoring tool for federated learning systems.

Federated learning enables collaborative AI training across decentralized devices while preserving privacy. However, it's vulnerable to various attacks including data poisoning, Byzantine failures, and backdoors.

This dashboard provides researchers and practitioners with real-time visibility into training progress, client behavior, and security threats."

---

### Live Demo (5 minutes)

**[Load Normal Training scenario]**

"Let's start with normal training. You can see:
- Global accuracy improving from 10% to 90%
- Loss decreasing over 50 rounds
- All 10 clients participating actively
- No security events - everything looks normal"

**[Switch to Client Analytics]**

"In Client Analytics, we can examine per-client metrics:
- Each client's accuracy and loss
- Reputation scores (all near 1.0 = trustworthy)
- Anomaly scores (all low = no issues)"

**[Load Label Flipping Attack scenario]**

"Now let's see what happens during an attack:

In Training Monitor, notice rounds 10-25:
- Accuracy stops improving, plateaus at ~75%
- Loss increases during attack period

Let's check Security Status:
- Multiple 'attack_detected' events
- Clients 8 and 9 flagged as attackers

In Client Analytics:
- Attackers have low reputation (< 0.4)
- High anomaly scores (> 0.6)
- Other clients unaffected"

**[Load SignGuard Defense scenario]**

"With SignGuard defense:
- Same attack, but model still converges to ~85%
- Defense events show attackers being downweighted
- System maintains utility while mitigating threat"

---

### Advanced Features (2 minutes)

"The dashboard also supports:
- **Privacy Budget tracking** for differential privacy
- **Experiment Comparison** to analyze different configurations
- **Real-time updates** via WebSocket
- **Export functionality** for reports"

---

### Conclusion (1 minute)

"This dashboard helps researchers:
1. Visualize FL training in real-time
2. Understand attack and defense mechanisms
3. Compare different defense strategies
4. Demonstrate security concepts effectively

It's particularly valuable for:
- Academic presentations and PhD defenses
- Security research and education
- Production FL system monitoring

Thank you! Any questions?"

---

## Live Demonstration

### Preparation Checklist

- [ ] Install all dependencies
- [ ] Generate demo data scenarios
- [ ] Test all pages load correctly
- [ ] Prepare backup scenarios
- [ ] Set up screen sharing
- [ ] Have command-line ready for any troubleshooting

### During Demo

**Do**:
- Speak clearly and confidently
- Point to specific metrics on screen
- Explain technical terms briefly
- Allow audience to ask questions
- Show enthusiasm for the technology

**Don't**:
- Read directly from slides
- Get bogged down in code details
- Panic if something crashes (have backup)
- Rush through explanations
- Use too much jargon without context

### Backup Plans

**If demo crashes**:
1. Have screenshots ready as backup
2. Switch to recorded video walkthrough
3. Continue with conceptual explanation

**If data doesn't load**:
1. Use "Generate Demo Data" button
2. Fall back to simpler scenario
3. Describe expected behavior

**If question is too technical**:
1. Acknowledge complexity
2. Offer to discuss offline
3. Return to main demo flow

---

## Common Questions

### Q: How is this different from monitoring tools like TensorBoard?

**A**: "TensorBoard focuses on model metrics. This dashboard specifically monitors security aspects - attack detection, client reputation, privacy budgets. It's designed for federated learning's unique threat model."

### Q: Can this detect zero-day attacks?

**A**: "The current implementation detects known attack patterns (label flipping, Byzantine, backdoor). For zero-day attacks, the anomaly detection can flag unusual behavior but would need additional analysis. Future work could integrate ML-based anomaly detection."

### Q: What's the performance overhead?

**A**: "Minimal for the dashboard itself - it's a visualization layer. The simulator runs entirely independently. In production with real FL systems, overhead depends on data transmission frequency and computation for defense mechanisms."

### Q: Can this be integrated with real FL frameworks?

**A**: "Yes, the data models are generic and can work with TensorFlow Federated, PySyft, Flower, or custom implementations. You'd just need to push metrics to the dashboard via WebSocket or Redis."

### Q: What about client privacy?

**A**: "The dashboard only shows aggregate metrics (accuracy, loss, reputation). It doesn't expose raw client data. For production, you'd add authentication and access controls."

---

## Tips for Success

### For Academic Presentations

1. **Focus on Research Contribution**: Emphasize novelty
2. **Show Comparison**: Before/after attack, with/without defense
3. **Use Metrics**: Quantify defense effectiveness
4. **Cite Literature**: Reference related work

### For Industry Demos

1. **Focus on Practicality**: Real-world deployment scenarios
2. **Show Scalability**: Mention 100+ clients
3. **Highlight Cost**: Efficient monitoring, low overhead
4. **Integration**: Works with existing FL frameworks

### For PhD Committee

1. **Depth Over Breadth**: Deep dive into one scenario
2. **Theoretical Foundation**: Explain SignGuard, Krum, etc.
3. **Experimental Design**: How scenarios were generated
4. **Evaluation Metrics**: Attack detection rate, false positives
5. **Limitations**: Be honest about what's not shown

### Screen Setup Tips

- **Resolution**: 1920x1080 or higher
- **Font Size**: Use browser zoom (Ctrl+) if needed
- **Browser Fullscreen**: F11 for maximum visibility
- **Second Monitor**: Show notes while presenting
- **Practice**: Run through demo 3+ times

### Timing Guide

| Section | Time | Notes |
|---------|------|-------|
| Introduction | 2 min | Keep concise |
| Normal Training | 1 min | Quick overview |
| Attack Demo | 3 min | Main focus |
| Defense Demo | 2 min | Show effectiveness |
| Comparison | 1 min | Side-by-side |
| Q&A | 5 min | Reserve time |
| **Total** | **15 min** | Adjust as needed |

---

## Troubleshooting Demo Issues

### Problem: Dashboard loads slowly

**Solution**:
- Use "Demo Mode" to avoid live training
- Pre-load all scenarios before presentation
- Close other browser tabs

### Problem: Data doesn't display

**Solution**:
- Clear browser cache
- Refresh page (Ctrl+R)
- Check browser console for errors

### Problem: Wrong scenario loaded

**Solution**:
- Use dropdown to select correct scenario
- Click "Load Scenario" button
- Verify scenario name in header

---

## Additional Resources

- **Source Code**: `/home/ubuntu/30Days_Project/fl_security_dashboard`
- **Documentation**: `docs/` directory
- **Demo Data**: `data/demo_scenarios/`
- **Contact**: [your-email@example.com]

---

## Feedback and Improvement

After your demo, consider:

1. **What questions were asked most?** → Add to FAQ
2. **What was confusing?** → Improve explanation
3. **What features were requested?** → Add to roadmap
4. **What worked well?** → Reuse in future demos

Good luck with your presentation!
