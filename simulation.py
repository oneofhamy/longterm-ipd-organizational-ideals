# Iterated Prisoner's Dilemma Simulation (Generational Turnover, Memory Decay, Full Analytics, All Major Strategies, Time-Series Logging)
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# --- REPRODUCIBILITY ---
random.seed(42)
np.random.seed(42)

# Define payoff matrix
payoff_matrix = {
    ("cooperate", "cooperate"): (3, 3),
    ("cooperate", "defect"): (0, 5),
    ("defect", "cooperate"): (5, 0),
    ("defect", "defect"): (1, 1)
}

# --- Strategy function definitions ---

def moq_strategy(agent, partner, last_self=None, last_partner=None):
    if last_partner == "defect":
        if agent.get("moq_forgiveness", 0.0) > 0 and random.random() < agent["moq_forgiveness"]:
            return "cooperate"
        return "defect"
    return "cooperate"

def gmoq_strategy(agent, partner, last_self=None, last_partner=None):
    agent["moq_forgiveness"] = 0.1
    return moq_strategy(agent, partner, last_self, last_partner)

def hgmoq_strategy(agent, partner, last_self=None, last_partner=None):
    agent["moq_forgiveness"] = 0.3
    return moq_strategy(agent, partner, last_self, last_partner)

def tft_strategy(agent, partner, last_self=None, last_partner=None):
    if last_partner is None:
        return "cooperate"
    return last_partner

def gtft_strategy(agent, partner, last_self=None, last_partner=None):
    if last_partner == "defect":
        if random.random() < 0.1:
            return "cooperate"
        return "defect"
    return "cooperate"

def hgtft_strategy(agent, partner, last_self=None, last_partner=None):
    if last_partner == "defect":
        if random.random() < 0.3:
            return "cooperate"
        return "defect"
    return "cooperate"

def allc_strategy(agent, partner, last_self=None, last_partner=None):
    return "cooperate"

def alld_strategy(agent, partner, last_self=None, last_partner=None):
    return "defect"

def wsls_strategy(agent, partner, last_self=None, last_partner=None, last_payoff=None):
    if last_self is None or last_payoff is None:
        return "cooperate"
    if last_payoff in [3, 1]:
        return last_self
    else:
        return "defect" if last_self == "cooperate" else "cooperate"

def ethnocentric_strategy(agent, partner, last_self=None, last_partner=None):
    return "cooperate" if agent["tag"] == partner["tag"] else "defect"

def random_strategy(agent, partner, last_self=None, last_partner=None):
    return "cooperate" if random.random() < 0.5 else "defect"

def grim_trigger_strategy(agent, partner, last_self=None, last_partner=None):
    # Defect forever after any defection by the partner
    if 'grim_triggered' not in agent:
        agent['grim_triggered'] = False
    if last_partner == "defect" or agent['grim_triggered']:
        agent['grim_triggered'] = True
        return "defect"
    return "cooperate"

def cluster_utilitarian_strategy(agent, partner, last_self=None, last_partner=None):
    # Cooperate if same cluster, else defect (could be more sophisticated)
    return "cooperate" if agent.get("cluster", -99) == partner.get("cluster", -100) else "defect"

def global_utilitarian_strategy(agent, partner, last_self=None, last_partner=None):
    # Always cooperates for global good (can add global metrics)
    return "cooperate"

def factionalist_strategy(agent, partner, last_self=None, last_partner=None):
    # Stronger in-group preference than ethnocentric (could refuse even when neutral)
    return "cooperate" if agent["tag"] == partner["tag"] else "defect"

def propaganda_office_strategy(agent, partner, last_self=None, last_partner=None):
    # Only used for cluster-level reputation inflation/deflation, but for normal round just act as Random
    # Insert special logic at post-interaction for cluster
    return random_strategy(agent, partner, last_self, last_partner)

def saboteur_strategy(agent, partner, last_self=None, last_partner=None):
    # Pure defector, with possible special event effect (could try to target high-score)
    return "defect"

def conformist_strategy(agent, partner, last_self=None, last_partner=None):
    # Should copy most common action in neighborhood (stub: acts random here, for simplicity)
    return random_strategy(agent, partner, last_self, last_partner)

def shadow_broker_strategy(agent, partner, last_self=None, last_partner=None):
    # Sometimes randomizes or masks its own karma perception
    if random.random() < 0.2:
        agent['broadcasted_karma'] = random.randint(-10, 10)
    return random_strategy(agent, partner, last_self, last_partner)

def founding_descendant_strategy(agent, partner, last_self=None, last_partner=None):
    # Cooperates if partner is cluster founder or descendant, else random
    return random_strategy(agent, partner, last_self, last_partner)

# --- Strategy map for selection ---
strategy_functions = {
    "MoQ": moq_strategy,
    "GMoQ": gmoq_strategy,
    "HGMoQ": hgmoq_strategy,
    "TFT": tft_strategy,
    "GTFT": gtft_strategy,
    "HGTFT": hgtft_strategy,
    "ALLC": allc_strategy,
    "ALLD": alld_strategy,
    "WSLS": wsls_strategy,
    "Ethnocentric": ethnocentric_strategy,
    "Random": random_strategy,
    "GrimTrigger": grim_trigger_strategy,
    "ClusterUtilitarian": cluster_utilitarian_strategy,
    "GlobalUtilitarian": global_utilitarian_strategy,
    "Factionalist": factionalist_strategy,
    "Saboteur": saboteur_strategy,
    "Conformist": conformist_strategy,
    "ShadowBroker": shadow_broker_strategy,
    "FoundingDescendant": founding_descendant_strategy,
}

# --- Agent initialization weights (updated) ---
init_agents = 120
strategy_population = {
    "MoQ": 4,
    "GMoQ": 3,
    "HGMoQ": 2,
    "TFT": 10,
    "GTFT": 5,
    "HGTFT": 4,
    "ALLC": 5,
    "ALLD": 5,
    "WSLS": 10,
    "Ethnocentric": 10,
    "Random": 5,
    "GrimTrigger": 7,
    "ClusterUtilitarian": 7,
    "GlobalUtilitarian": 2,
    "Factionalist": 15,
    "Saboteur": 2,
    "Conformist": 22,
    "ShadowBroker": 2,
    # PropagandaOffice and FoundingDescendant are only created by event, never initial
}
# Weighted list for random sampling
strategy_choices_weighted = []
for strat, pct in strategy_population.items():
    num = int(round(pct / 100 * init_agents))
    strategy_choices_weighted.extend([strat] * num)
while len(strategy_choices_weighted) < init_agents:
    strategy_choices_weighted.append("Conformist")
strategy_choices_weighted = strategy_choices_weighted[:init_agents]
random.shuffle(strategy_choices_weighted)

# -- Agent factory --
def make_agent(agent_id, tag=None, strategy=None, parent=None, birth_epoch=0):
    if parent:
        tag = parent["tag"]
        strategy = parent["strategy"]
    if not tag:
        tag = random.choice(["Red", "Blue"])
    if not strategy:
        strategy = random.choice(strategy_choices_weighted)
    lifespan = min(max(int(np.random.normal(90, 15)), 60), 120)
    return {
        "id": agent_id,
        "tag": tag,
        "strategy": strategy,
        "karma": 0,
        "perceived_karma": defaultdict(lambda: 0),
        "score": 0,
        "trust": defaultdict(int),
        "history": [],
        "broadcasted_karma": 0,
        "apology_available": True,
        "birth_epoch": birth_epoch,
        "lifespan": lifespan,
        "strategy_memory": {},
        "retribution_events": 0,
        "in_group_score": 0,
        "out_group_score": 0,
        "karma_log": [],
        "perceived_log": [],
        "karma_perception_delta_log": [],
        "trust_given_log": [],
        "trust_received_log": [],
        "reciprocity_log": [],
        "ostracized": False,
        "ostracized_at": None,
        "fairness_index": 0,
        "score_efficiency": 0,
        "trust_reciprocity": 0,
        "cluster": None,
        "generation": birth_epoch // 120
    }

# -- Initialize agents --
agent_population = []
network = nx.Graph()
agent_id_counter = 0
for strat in strategy_choices_weighted:
    agent = make_agent(agent_id_counter, strategy=strat, birth_epoch=0)
    agent_population.append(agent)
    network.add_node(agent_id_counter, tag=agent["tag"], strategy=agent["strategy"])
    agent_id_counter += 1

# --- TIME-SERIES LOGGING (NEW, for post-hoc analytics) ---
mean_true_karma_ts = []
mean_perceived_karma_ts = []
mean_score_ts = []
strategy_karma_ts = {s: [] for s in strategy_functions.keys()}

# -- Karma function --
def evaluate_karma(actor, action, opponent_action, last_action, strategy):
    if action == "defect":
        if opponent_action == "defect" and last_action == "cooperate":
            return +1
        if last_action == "defect":
            return -1
        return -2
    elif action == "cooperate" and opponent_action == "defect":
        return +2
    return 0

# -- Main interaction function (all memory and strategy logic) --
def belief_interact(a, b, rounds=5):
    amem = a["strategy_memory"].get(b["id"], [None, None, None])
    bmem = b["strategy_memory"].get(a["id"], [None, None, None])

    history_a, history_b = [], []
    karma_a, karma_b, score_a, score_b = 0, 0, 0, 0

    for _ in range(rounds):
        if a["strategy"] == "WSLS":
            act_a = wsls_strategy(a, b, amem[0], amem[1], amem[2])
        else:
            act_a = strategy_functions[a["strategy"]](a, b, amem[0], amem[1])
        if b["strategy"] == "WSLS":
            act_b = wsls_strategy(b, a, bmem[0], bmem[1], bmem[2])
        else:
            act_b = strategy_functions[b["strategy"]](b, a, bmem[0], bmem[1])

        # Apology chance
        if act_a == "defect" and a["apology_available"] and random.random() < 0.2:
            a["score"] -= 1
            a["apology_available"] = False
            act_a = "cooperate"
        if act_b == "defect" and b["apology_available"] and random.random() < 0.2:
            b["score"] -= 1
            b["apology_available"] = False
            act_b = "cooperate"

        payoff = payoff_matrix[(act_a, act_b)]
        score_a += payoff[0]
        score_b += payoff[1]

        # For analytics only
        if a["tag"] == b["tag"]:
            a["in_group_score"] += payoff[0]
            b["in_group_score"] += payoff[1]
        else:
            a["out_group_score"] += payoff[0]
            b["out_group_score"] += payoff[1]

        karma_a += evaluate_karma(a["strategy"], act_a, act_b, history_a[-1] if history_a else None, a["strategy"])
        karma_b += evaluate_karma(b["strategy"], act_b, act_a, history_b[-1] if history_b else None, b["strategy"])

        history_a.append(act_a)
        history_b.append(act_b)

        # Retribution analytics
        if len(history_a) >= 2 and history_a[-2] == "cooperate" and act_a == "defect":
            a["retribution_events"] += 1
        if len(history_b) >= 2 and history_b[-2] == "cooperate" and act_b == "defect":
            b["retribution_events"] += 1

        # Logging for karma drift
        a["karma_log"].append(a["karma"])
        b["karma_log"].append(b["karma"])
        a["perceived_log"].append(np.mean(list(a["perceived_karma"].values())) if a["perceived_karma"] else 0)
        b["perceived_log"].append(np.mean(list(b["perceived_karma"].values())) if b["perceived_karma"] else 0)
        a["karma_perception_delta_log"].append(a["perceived_log"][-1] - a["karma"])
        b["karma_perception_delta_log"].append(b["perceived_log"][-1] - b["karma"])

        # Store memory for next round
        amem = [act_a, act_b, payoff[0]]
        bmem = [act_b, act_a, payoff[1]]

    a["karma"] += karma_a
    b["karma"] += karma_b
    a["score"] += score_a
    b["score"] += score_b
    a["trust"][b["id"]] += score_a + a["perceived_karma"][b["id"]]
    b["trust"][a["id"]] += score_b + b["perceived_karma"][a["id"]]
    a["history"].append((b["id"], history_a))
    b["history"].append((a["id"], history_b))
    a["strategy_memory"][b["id"]] = amem
    b["strategy_memory"][a["id"]] = bmem

    # Reputation masking
    if random.random() < 0.2:
        a["broadcasted_karma"] = max(a["karma"], a["broadcasted_karma"])
        b["broadcasted_karma"] = max(b["karma"], b["broadcasted_karma"])

    a["perceived_karma"][b["id"]] += (b["broadcasted_karma"] if b["broadcasted_karma"] else karma_b) * 0.5
    b["perceived_karma"][a["id"]] += (a["broadcasted_karma"] if a["broadcasted_karma"] else karma_a) * 0.5

    # Propagation of belief
    if len(a["history"]) > 1:
        last = a["history"][-2][0]
        a["perceived_karma"][last] += a["perceived_karma"][b["id"]] * 0.1
    if len(b["history"]) > 1:
        last = b["history"][-2][0]
        b["perceived_karma"][last] += b["perceived_karma"][a["id"]] * 0.1

    total_trust = a["trust"][b["id"]] + b["trust"][a["id"]]
    network.add_edge(a["id"], b["id"], weight=total_trust)


# --- Cluster Founding Ideals (NEW for generational power structure modeling) (Moved before the loop) ---
if not hasattr(network, "cluster_founding_ideals"):
    network.cluster_founding_ideals = {}
# Track which clusters have a Propaganda Office (by cluster id)
if not hasattr(network, "cluster_propaganda"):
    network.cluster_propaganda = {}

# ---- Main simulation loop ----
max_epochs = 15000
generation_length = 120

for epoch in range(max_epochs):
    # Cluster/community detection (Moved inside the loop)
    clusters = list(greedy_modularity_communities(network))
    cluster_map = {}
    for i, group in enumerate(clusters):
        for node in group:
            cluster_map[node] = i

    centrality = nx.betweenness_centrality(network)
    for a in agent_population:
        a["cluster"] = cluster_map.get(a["id"], -1)
        a["influence"] = centrality.get(a["id"], 0) # Use .get() with a default value

    np.random.shuffle(agent_population)
    for i in range(0, len(agent_population) - 1, 2):
        a = agent_population[i]
        b = agent_population[i + 1]
        belief_interact(a, b, rounds=5)

    # Decay and reset
    for a in agent_population:
        for k in a["perceived_karma"]:
            a["perceived_karma"][k] *= 0.95
        a["apology_available"] = True

    # --- Mutation every 30 epochs
    if epoch % 30 == 0 and epoch > 0:
        for a in agent_population:
            if a["score"] < np.median([x["score"] for x in agent_population]):
                high_score_agent = max(agent_population, key=lambda x: x["score"])
                a["strategy"] = random.choice([high_score_agent["strategy"], random.choice(strategy_choices_weighted)])

    # --- AGING & DEATH (agents die after lifespan, replaced by child agent)
    to_replace = []
    for idx, agent in enumerate(agent_population):
        age = epoch - agent["birth_epoch"]
        if age >= agent["lifespan"]:
            to_replace.append(idx)
    for idx in to_replace:
        dead = agent_population[idx]
        try:
            network.remove_node(dead["id"])
        except Exception:
            pass
        # Before replacement: check if cluster is eligible for Propaganda Office creation
        cluster_id = cluster_map.get(dead["id"], -1)
        # Only established clusters (>3 members), only one per cluster, only 1 per 4-6 clusters globally
        cluster_agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
        num_clusters = len(set(cluster_map.values()))
        max_offices = max(1, num_clusters // 4)
        existing_offices = sum(1 for a in agent_population if a["strategy"] == "PropagandaOffice")
        has_office = any(a["strategy"] == "PropagandaOffice" and a.get("cluster", -1) == cluster_id for a in agent_population)
        if (
            cluster_id != -1
            and len(cluster_agents) >= 3
            and not has_office
            and existing_offices < max_offices
            and cluster_id not in network.cluster_propaganda
        ):
            # Convert one agent in the cluster to PropagandaOffice
            office_candidate = random.choice(cluster_agents)
            office_candidate["strategy"] = "PropagandaOffice"
            network.cluster_propaganda[cluster_id] = office_candidate["id"]
            # Optionally, set a cluster-level field for propaganda activity
            print(f"🗞️ Cluster {cluster_id}: Propaganda Office established (agent {office_candidate['id']}) at epoch {epoch}.")

        new_agent = make_agent(agent_id_counter, parent=dead, birth_epoch=epoch)
        agent_id_counter += 1
        agent_population[idx] = new_agent
        network.add_node(new_agent["id"], tag=new_agent["tag"], strategy=new_agent["strategy"])

    # Recalculate centrality and cluster_map after agent death and replacement
    # This section is now redundant as the recalculation happens at the start of the loop
    # clusters = list(greedy_modularity_communities(network))
    # cluster_map = {}
    # for i, group in enumerate(clusters):
    #     for node in group:
    #         cluster_map[node] = i
    # centrality = nx.betweenness_centrality(network)
    # for a in agent_population:
    #     a["cluster"] = cluster_map.get(a["id"], -1)
    #     a["influence"] = centrality.get(a["id"], 0) # Use .get() with a default value

    # --- Martyr/Founder/Trauma Event Logic (injects founding ideal on death of highly influential agent or betrayal) ---
        cluster_id = cluster_map.get(dead["id"], -1)
        is_martyr = (
            (dead["score"] > np.percentile([a["score"] for a in agent_population], 90)) or
            (dead.get("influence", 0) > np.percentile(list(centrality.values()), 90)) or
            (abs(dead["karma"]) > np.percentile([abs(a["karma"]) for a in agent_population], 90))
        )
        betrayal = False
        if cluster_id in network.cluster_founding_ideals:
            cluster_agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
            if len(cluster_agents) >= 3:
                trust_matrix = np.array([[agent["trust"].get(other["id"], 0) for other in cluster_agents] for agent in cluster_agents])
                betrayal = np.mean(trust_matrix) < -5  # Tune threshold

        if (is_martyr or betrayal) and cluster_id in network.cluster_founding_ideals:
            network.cluster_founding_ideals[cluster_id].append({
                'epoch': epoch,
                'ideal': np.mean([a["karma"] for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]),
                'idolization': max([a["score"] for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]),
                'decay': 0.0,
                'layer': len(network.cluster_founding_ideals[cluster_id]),
                'founder': dead["id"] if is_martyr else None,
            })


    # --- TIME-SERIES LOGGING: append to logs at END of each epoch (NEW) ---
    mean_true_karma_ts.append(np.mean([a["karma"] for a in agent_population]))
    mean_perceived_karma_ts.append(np.mean([
        np.mean(list(a["perceived_karma"].values())) if a["perceived_karma"] else 0
        for a in agent_population
    ]))
    mean_score_ts.append(np.mean([a["score"] for a in agent_population]))
    for strat in strategy_karma_ts.keys():
        strat_agents = [a for a in agent_population if a["strategy"] == strat]
        mean_strat_karma = np.mean([a["karma"] for a in strat_agents]) if strat_agents else np.nan
        strategy_karma_ts[strat].append(mean_strat_karma)

# === POST-SIMULATION ANALYTICS ===
ostracism_threshold = 3
for a in agent_population:
    given = sum(a["trust"].values())
    received_list = []
    for tid in list(a["trust"].keys()):
        if tid < len(agent_population):
            receiving_agent = next((ag for ag in agent_population if ag["id"] == tid), None)
            if receiving_agent and a["id"] in receiving_agent["trust"]:
                received_list.append(receiving_agent["trust"][a["id"]])
    received = sum(received_list)
    a["trust_given_log"].append(given)
    a["trust_received_log"].append(received)
    a["reciprocity_log"].append(given / (received + 1e-6) if received > 0 else 0)
    avg_perceived = np.mean(list(a["perceived_karma"].values())) if a["perceived_karma"] else 0
    a["fairness_index"] = a["score"] / (avg_perceived + 1e-6) if avg_perceived != 0 else 0
    if len([k for k in a["trust"] if a["trust"][k] > 0]) < ostracism_threshold:
        a["ostracized"] = True
    a["score_efficiency"] = a["score"] / (abs(a["karma"]) + 1) if a["karma"] != 0 else 0
    a["trust_reciprocity"] = np.mean(a["reciprocity_log"]) if a["reciprocity_log"] else 0

# === OUTPUT ===
df = pd.DataFrame([{
    "ID": a["id"],
    "Tag": a["tag"],
    "Strategy": a["strategy"],
    "True Karma": a["karma"],
    "Score": a["score"],
    "Connections": len(a["trust"]),
    "Avg Perceived Karma": round(np.mean(list(a["perceived_karma"].values())), 2) if a["perceived_karma"] else 0,
    "In-Group Score": a["in_group_score"],
    "Out-Group Score": a["out_group_score"],
    "Retributions": a["retribution_events"],
    "Score Efficiency": a["score_efficiency"],
    "Influence Centrality": round(a["influence"], 4),
    "Ostracized": a["ostracized"],
    "Fairness Index": round(a["fairness_index"], 3),
    "Trust Reciprocity": round(a["trust_reciprocity"], 3),
    "Cluster": a["cluster"],
    "Karma-Perception Delta": round(np.mean(a["karma_perception_delta_log"]), 2) if a["karma_perception_delta_log"] else 0,
    "Generation": a["birth_epoch"] // generation_length
} for a in agent_population]).sort_values(by="Score", ascending=False).reset_index(drop=True)

import IPython
IPython.display.display(df.head(20))

# === ADDITIONAL POST-HOC ANALYTICS ===

# 1. Karma Ratio (In-Group vs Out-Group Karma)
df["In-Out Karma Ratio"] = df.apply(
    lambda row: round(row["In-Group Score"] / (row["Out-Group Score"] + 1e-6), 2) if row["Out-Group Score"] != 0 else float('inf'),
    axis=1
)

# 2. Reputation Manipulation (Karma-Perception Delta)
reputation_manipulators = df.sort_values(by="Karma-Perception Delta", ascending=False).head(5)
print("\nTop 5 Reputation Manipulators (most positive karma-perception delta):")
display(reputation_manipulators[["ID", "Tag", "Strategy", "True Karma", "Avg Perceived Karma", "Karma-Perception Delta", "Score"]])

# 3. Network Centrality vs True Karma (Ethics vs Power Plot/Correlation)
from scipy.stats import pearsonr

centrality_list = df["Influence Centrality"].values
karma_list = df["True Karma"].values
# Ignore nan if present
mask = ~np.isnan(centrality_list) & ~np.isnan(karma_list)
corr, pval = pearsonr(centrality_list[mask], karma_list[mask])

print(f"\nPearson correlation between Influence Centrality and True Karma: r = {corr:.3f}, p = {pval:.3g}")

# Optional scatter plot (ethics vs power)
plt.figure(figsize=(8, 5))
plt.scatter(df["Influence Centrality"], df["True Karma"], c=df["Cluster"], cmap="tab20", s=80, edgecolors="k")
plt.xlabel("Influence Centrality (Network Power)")
plt.ylabel("True Karma (Ethics/Morality)")
plt.title("Ethics vs Power: Influence Centrality vs True Karma")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Cabal Detection Plot ---
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Influence Centrality"],
    df["Score Efficiency"],
    c=df["True Karma"],
    cmap="coolwarm",
    s=80,
    edgecolors="k"
)
plt.title("🕳️ Cabal Detection: Influence vs Score Efficiency (colored by Karma)")
plt.xlabel("Influence Centrality")
plt.ylabel("Score Efficiency (Score / |Karma|)")
cbar = plt.colorbar(scatter)
cbar.set_label("True Karma")
plt.grid(True)
plt.show()

# --- Karma Drift Plot for a sample of agents ---
plt.figure(figsize=(12, 6))
sample_agents = agent_population[:6]
for a in sample_agents:
    true_karma = a["karma_log"]
    perceived_karma = a["perceived_log"]
    x = list(range(len(true_karma)))
    plt.plot(x, true_karma, label=f"Agent {a['id']} True", linestyle='-')
    plt.plot(x, perceived_karma, label=f"Agent {a['id']} Perceived", linestyle='--')
plt.title("📉 Karma Drift: True vs Perceived Karma Over Time")
plt.xlabel("Interaction Rounds")
plt.ylabel("Karma Score")
plt.legend()
plt.grid(True)
plt.show()

# --- SERIAL MANIPULATORS ANALYTICS ---

# 1. Define a minimum number of steps for stability (e.g., agents with at least 50 logged deltas)
min_steps = 50
serial_manipulator_threshold = 5  # e.g., mean delta > 5

serial_manipulators = []
for a in agent_population:
    deltas = a["karma_perception_delta_log"]
    if len(deltas) >= min_steps:
        # Count how many times delta was "high" (manipulating) and calculate mean/max
        high_count = sum(np.array(deltas) > serial_manipulator_threshold)
        mean_delta = np.mean(deltas)
        max_delta = np.max(deltas)
        if high_count > len(deltas) * 0.5 and mean_delta > serial_manipulator_threshold:  # e.g. more than half the time
            serial_manipulators.append({
                "ID": a["id"],
                "Tag": a["tag"],
                "Strategy": a["strategy"],
                "Mean Delta": round(mean_delta, 2),
                "Max Delta": round(max_delta, 2),
                "Total Steps": len(deltas),
                "True Karma": a["karma"],
                "Score": a["score"]
            })

serial_manipulators_df = pd.DataFrame(serial_manipulators).sort_values(by="Mean Delta", ascending=False)
print("\nSerial Reputation Manipulators (consistently high karma-perception delta):")
display(serial_manipulators_df)

# --- SHADOW CABAL DETECTION ANALYTIC ---

print("\n==== SHADOW CABAL DETECTION & INFLUENCE AUDIT ====\n")

import seaborn as sns

# 1. Cluster-level audit for shadow cabal characteristics
cluster_shadow_report = []
for cluster_id in df["Cluster"].unique():
    cluster_members = df[df["Cluster"] == cluster_id]
    if len(cluster_members) < 3: continue  # Ignore tiny clusters

    # Compute mean metrics for the cluster
    mean_perceived_karma = cluster_members["Avg Perceived Karma"].mean()
    mean_connections = cluster_members["Connections"].mean()
    mean_score = cluster_members["Score"].mean()
    mean_delta = cluster_members["Karma-Perception Delta"].mean()
    max_delta = cluster_members["Karma-Perception Delta"].max()

    # Count members with high connections and score but low recent manipulation
    legacy_influence = cluster_members[
        (cluster_members["Score"] > df["Score"].median()) &
        (cluster_members["Karma-Perception Delta"].abs() < 5)
    ]
    # Shadow cabal flag: high trust/perceived_karma, not manipulating now
    if mean_perceived_karma > 2 and len(legacy_influence) > 0:
        cluster_shadow_report.append({
            "Cluster": cluster_id,
            "Mean Perceived Karma": round(mean_perceived_karma, 2),
            "Mean Connections": round(mean_connections, 2),
            "Mean Score": round(mean_score, 2),
            "Mean Karma-Perception Delta": round(mean_delta, 2),
            "Max Delta": round(max_delta, 2),
            "Legacy Influencers": legacy_influence[["ID", "Strategy", "Score", "Connections", "Avg Perceived Karma"]].values.tolist(),
            "Legacy Influence Count": len(legacy_influence),
            "Members": cluster_members["ID"].tolist()
        })

# 2. Show all detected shadow cabals with a summary
if cluster_shadow_report:
    print(f"Detected {len(cluster_shadow_report)} shadow cabal(s):\n")
    for sc in cluster_shadow_report:
        print(f"- Cluster {sc['Cluster']} | Mean Perceived Karma: {sc['Mean Perceived Karma']}, "
              f"Mean Connections: {sc['Mean Connections']}, Legacy Influencers: {sc['Legacy Influence Count']}")
        print("  IDs:", [x[0] for x in sc["Legacy Influencers"]])
else:
    print("No shadow cabals detected under current criteria.")

# 3. Optional: Log shadow cabal influence as a DataFrame for further export/analysis
shadow_cabals_df = pd.DataFrame(cluster_shadow_report)
if not shadow_cabals_df.empty:
    display(shadow_cabals_df[["Cluster", "Mean Perceived Karma", "Mean Connections", "Legacy Influence Count", "Members"]])

# 4. Optional: Visualize clusters with shadow cabal markers (colors by shadow cabal flag)
shadow_cabal_clusters = [sc['Cluster'] for sc in cluster_shadow_report]
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Influence Centrality", y="True Karma",
    hue=df["Cluster"].apply(lambda x: "Shadow Cabal" if x in shadow_cabal_clusters else "Other"),
    style=df["Cluster"].apply(lambda x: "Shadow Cabal" if x in shadow_cabal_clusters else "Other"),
    s=90
)
plt.xlabel("Influence Centrality (Network Power)")
plt.ylabel("True Karma (Ethics/Morality)")
plt.title("Shadow Cabal Map: Ethics vs Power (Shadow Cabals Highlighted)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# --- Founding Ideals & Institutional Memory Export ---
print("\n==== CLUSTER FOUNDING IDEALS ====\n")
for cid, ideals in network.cluster_founding_ideals.items():
    print(f"Cluster {cid}:")
    for i, ideal in enumerate(ideals):
        age = max_epochs - ideal["epoch"]
        print(f"  Layer {i}: Ideal {ideal['ideal']:.2f}, Idolization {ideal['idolization']:.2f}, Age {age}, Decay {ideal['decay']:.4f}, Founder {ideal['founder']}")

# Optional: Export as DataFrame
founding_ideals_records = []
for cid, ideals in network.cluster_founding_ideals.items():
    for i, ideal in enumerate(ideals):
        founding_ideals_records.append({
            "Cluster": cid,
            "Layer": i,
            "Ideal": ideal["ideal"],
            "Idolization": ideal["idolization"],
            "Decay": ideal["decay"],
            "Founder": ideal["founder"],
            "Epoch": ideal["epoch"]
        })
founding_ideals_df = pd.DataFrame(founding_ideals_records)
display(founding_ideals_df)

# --- Propaganda Office Analytics ---

# 1. Find all Propaganda Office agents and the clusters they occupy
prop_offices = [a for a in agent_population if a["strategy"] == "PropagandaOffice"]
office_clusters = set(a["cluster"] for a in prop_offices)

print("\n=== PROPAGANDA OFFICE CLUSTER ANALYTICS ===\n")
if not prop_offices:
    print("No clusters have a Propaganda Office.")
else:
    print(f"Total Propaganda Offices: {len(prop_offices)}")
    for office in prop_offices:
        print(f" - Agent {office['id']} (Cluster {office['cluster']}) | Score: {office['score']} | Influence: {office.get('influence', 0):.4f}")

    # 2. Cluster-level analysis: Compare clusters with and without Propaganda Offices
    clusters_with_office = list(office_clusters)
    clusters_without_office = [c for c in df["Cluster"].unique() if c not in clusters_with_office]

    def cluster_summary(cluster_id):
        members = df[df["Cluster"] == cluster_id]
        return {
            "Cluster": cluster_id,
            "Members": len(members),
            "Mean True Karma": round(members["True Karma"].mean(), 2),
            "Mean Perceived Karma": round(members["Avg Perceived Karma"].mean(), 2),
            "Mean Delta": round(members["Karma-Perception Delta"].mean(), 2),
            "Mean Score": round(members["Score"].mean(), 2),
            "Total Influence": round(members["Influence Centrality"].sum(), 3),
            "Office Agent": int(prop_offices[0]["id"]) if cluster_id in office_clusters else None
        }

    print("\nClusters WITH Propaganda Office:\n")
    for cid in clusters_with_office:
        summary = cluster_summary(cid)
        print(summary)

    print("\nClusters WITHOUT Propaganda Office:\n")
    for cid in clusters_without_office:
        summary = cluster_summary(cid)
        print(summary)

    # 3. Optional: Show effect size
    with_office = df[df["Cluster"].isin(clusters_with_office)]
    without_office = df[~df["Cluster"].isin(clusters_with_office)]

    print("\n--- Aggregate Comparison (mean per agent) ---")
    print(f"Mean Perceived Karma (Propaganda Office): {with_office['Avg Perceived Karma'].mean():.2f}")
    print(f"Mean Perceived Karma (No Office): {without_office['Avg Perceived Karma'].mean():.2f}")
    print(f"Mean Karma-Perception Delta (Propaganda Office): {with_office['Karma-Perception Delta'].mean():.2f}")
    print(f"Mean Karma-Perception Delta (No Office): {without_office['Karma-Perception Delta'].mean():.2f}")
    print(f"Mean Score (Propaganda Office): {with_office['Score'].mean():.2f}")
    print(f"Mean Score (No Office): {without_office['Score'].mean():.2f}")

    # 4. Optionally, visualize cluster means by office status
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [
        with_office['Avg Perceived Karma'].mean(),
        without_office['Avg Perceived Karma'].mean(),
        with_office['Karma-Perception Delta'].mean(),
        without_office['Karma-Perception Delta'].mean()
    ]
    bar_labels = [
        "With Office: Perceived Karma",
        "No Office: Perceived Karma",
        "With Office: Δ Karma-Percept",
        "No Office: Δ Karma-Percept"
    ]
    ax.bar(bar_labels, means, color=['#7dafff', '#c0c0c0', '#7dafff', '#c0c0c0'])
    ax.set_ylabel("Mean Value")
    ax.set_title("Propaganda Office Effect on Cluster Reputation")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
