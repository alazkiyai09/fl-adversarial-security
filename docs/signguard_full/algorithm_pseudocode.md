# SignGuard Algorithm Pseudocode

This document contains LaTeX pseudocode for the main SignGuard algorithms for the paper appendix.

---

## Algorithm 1: Client-Side Signing

```
Algorithm 1: Client-Side Update Signing
Input: Client id i, global model θ_t, training data D_i, private key sk_i
Output: Signed update σ_i

1: θ_i^0 ← LocalTrain(θ_t, D_i)                       // Local training
2: Δ_i ← θ_i^0 - θ_t                                       // Compute update
3: U_i ← (client_id, round, Δ_i, metrics)                  // Create update
4: σ_i ← ECDSA.Sign(Serialize(U_i), sk_i)                   // Sign update
5: k_i ← ECDSA.PublicKey(sk_i)                               // Get public key
6: return σ_i, k_i
```

---

## Algorithm 2: Server-Side Verification

```
Algorithm 2: Signature Verification and Aggregation
Input: Signed updates {σ_j}, global model θ_t, server state S_t
Output: New global model θ_{t+1}

1: V ← {}                                              // Verified updates
2: R ← {}                                              // Rejected clients
3: for each σ_j in {σ_j} do:
4:     if ECDSA.Verify(σ_j.update, σ_j.signature, σ_j.public_key) then
5:         V ← V ∪ {σ_j}
6:     else
7:         R ← R ∪ {σ_j.client_id}
8:     end if
9: end for
10:
11: if |V| < n_min then
12:     raise InsufficientClientsError
13:
14: // Anomaly detection
15: for each σ_j in V do:
16:     score_j ← EnsembleDetect(σ_j.update, θ_t)
17:     if score_j > τ then
18:         V ← V \ {σ_j}
19:         R ← R ∪ {σ_j.client_id}
20:     end if
21: end for
22:
23: // Reputation updates
24: for each σ_j in V do
25:     rep_j ← DecayReputation(
26:         rep_j[t-1],
27:         score_j.combined,
28:         is_verified=true,
29:     )
30: end for
31:
32: // Aggregation
33: weights ← {reputation_j : σ_j ∈ V}
34: θ_{t+1} ← Σ (weights_j × σ_j.update.parameters) / Σ weights
35:
36: return θ_{t+1}, V, R
```

---

## Algorithm 3: Ensemble Detection

```
Algorithm 3: Multi-Factor Anomaly Detection
Input: Model update U, global model θ, detection history H
Output: Anomaly score A ∈ [0,1]

1: // L2 norm detection
2: d_norm ← ‖U.parameters - θ.parameters‖_2            // L2 norm
3: s_norm ← NormalizeByHistory(d_norm, H)
4:
5: // Cosine similarity detection
6: v ← (U.parameters - θ.parameters) / ‖U.parameters - θ.parameters‖_2
7: d_cos ← 1 - (v · v̄)                                   // Cosine distance
8: s_cos ← NormalizeByHistory(d_cos, H)
9:
10: // Loss deviation detection
11: loss ← U.metrics["loss"]
12: s_loss ← NormalizeByHistory(loss, H)
13:
14: // Combined score
15: A ← w_norm × s_norm + w_cos × s_cos + w_loss × s_loss
16:
17: return A, (s_norm, s_cos, s_loss)
```

---

## Algorithm 4: Decay-Based Reputation

```
Algorithm 4: Time-Decay Reputation Update
Input: Current reputation r_t, anomaly score a_t, decay rate δ
Output: New reputation r_{t+1}

1: // Apply time decay
2: r_temp ← r_t × δ^(t - t_last)
3:
4: // Apply bonus or penalty
5: if a_t < τ_honest then                              // Low anomaly
6:     r_{t+1} ← min(r_max, r_temp + β)                 // Bonus
7: else if a_t > τ_malicious then                        // High anomaly
8:     r_{t+1} ← max(r_min, r_temp - a_t × ρ)            // Penalty
9: else:
10:     r_{t+1} ← r_temp
11: end if
12:
13: return r_{t+1}
```

---

## Algorithm 5: Reputation-Weighted Aggregation

```
Algorithm 5: Reputation-Weighted Aggregation
Input: Updates {U_j}, reputations {rep_j}, global model θ
Output: Aggregated model θ'

1: ← {}
2: total_weight ← 0
3:
4: for each U_j in {U_j} do
5:     client_id ← U_j.client_id
6:     if rep_j[client_id] < ρ_min then
7:         continue                                   // Skip low-reputation clients
8:     end if
9:     total_weight ← total_weight + rep_j[client_id]
10: end for
11:
12: if total_weight = 0 then
13:     raise NoValidClientsError
14:
15: // Aggregate parameters
16: for param_name in θ do
17:     weighted_sum ← 0
18:     for each U_j in {U_j} do
19:         weight ← rep_j[U_j.client_id] / total_weight
20:         weighted_sum ← weighted_sum + weight × U_j.parameters[param_name]
21:     end for
22:     θ[param_name] ← weighted_sum
23: end for
24:
25: return θ
```

---

## Algorithm 6: Krum Defense

```
Algorithm 6: Krum Robust Aggregation
Input: Updates {U_j}, number of Byzantines f
Output: Selected update U*

1: n ← |{U_j}|
2: k ← n - f - 2
3:
4: // Compute pairwise distances
5: for i = 1 to n do
6:     for j = 1 to n, j ≠ i do
7:         d[i,j] ← ‖U_i.parameters - U_j.parameters‖
8:     end for
9: end for
10:
11: // Compute scores
12: for i = 1 to n do
13:     // Get k closest neighbors
14:     sorted_distances ← Sort(d[i, :])
15:     score[i] ← Σ_{l=1}^{k} sorted_distances[l]
16: end for
17:
18: // Return update with minimum score
19: i* ← argmin(score)
20: return U_i*
```

---

## Notation Summary

- θ_t: Global model at round t
- Δ_i: Parameter update from client i
- σ_i: Signed update from client i
- τ: Anomaly threshold
- rep_j: Reputation of client j
- w: Aggregation weights
- n: Number of clients
- f: Number of Byzantine clients
- δ: Decay rate
- β: Honesty bonus
- ρ: Penalty factor

---

See the main paper for theoretical analysis and proofs.
