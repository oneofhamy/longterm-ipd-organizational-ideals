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
    ("cooperate", "defect"): (0, 6),
    ("defect", "cooperate"): (6, 0),
    ("defect", "defect"): (0, 0)
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
    # Behaves like a cluster utilitarian (in-group cooperation, out-group defection)
    return cluster_utilitarian_strategy(agent, partner, last_self, last_partner)

def saboteur_strategy(agent, partner, last_self=None, last_partner=None):
    # Pure defector, with possible special event effect (could try to target high-score)
    return "defect"

def conformist_strategy(agent, partner, last_self=None, last_partner=None):
    # Should copy most common action in neighborhood (stub: acts random here, for simplicity)
    return factionalist_strategy(agent, partner, last_self, last_partner)

def shadow_broker_strategy(agent, partner, last_self=None, last_partner=None):
    # Sometimes randomizes or masks its own karma perception
    if random.random() < 0.2:
        agent['broadcasted_karma'] = random.randint(-10, 10)
    return factionalist_strategy(agent, partner, last_self, last_partner)

# Helper function to check if an agent is a founder or descendant
# Define the window for being considered a descendant globally for simplicity in this function
# This could be made dynamic based on simulation parameters if needed.
DESCENDANT_WINDOW_EPOCHS = 180 # Example: 1.5 times the initial average lifespan (120*1.5)

def is_founder_or_descendant(agent_id, cluster_id, birth_epoch, network):
    if cluster_id == -1 or cluster_id not in network.cluster_founding_ideals:
        return False

    for ideal_layer in network.cluster_founding_ideals[cluster_id]:
        # Check if the agent is the founder of this layer
        if ideal_layer.get('founder') == agent_id:
            return True
        # Check if the agent was born within the descendant window after this layer's founding epoch
        if birth_epoch >= ideal_layer['epoch'] and birth_epoch <= ideal_layer['epoch'] + DESCENDANT_WINDOW_EPOCHS:
            return True
    return False

def founding_descendant_strategy(agent, partner, last_self=None, last_partner=None):
    # Behave identically to cluster_utilitarian_strategy
    return cluster_utilitarian_strategy(agent, partner, last_self, last_partner)

def accountant_strategy(agent, partner, last_self=None, last_partner=None):
    return factionalist_strategy(agent, partner, last_self, last_partner)

strategy_functions["Accountant"] = accountant_strategy
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
    "PropagandaOffice": propaganda_office_strategy,
    "Accountant": accountant_strategy,
}

# --- Agent initialization weights (updated) ---
init_agents = 200
strategy_population = {
    "MoQ": 4,                     # 4%
    "GMoQ": 2,                    # 2%
    "HGMoQ": 1,                   # 1%
    "TFT": 7,                     # 7%
    "GTFT": 5,                    # 5%
    "HGTFT": 2,                   # 2%
    "ALLC": 2,                    # 2%
    "ALLD": 5,                    # 5%
    "WSLS": 8,                    # 8%
    "Ethnocentric": 12,           # 12%
    "Random": 4,                  # 4%
    "GrimTrigger": 3,             # 3%
    "ClusterUtilitarian": 4,      # 4%
    "GlobalUtilitarian": 1,       # 1%
    "Factionalist": 12,           # 12%
    "Saboteur": 1,                # 1%
    "Conformist": 24,             # 24%
    "ShadowBroker": 1,            # 1%
    "accountant": 2,              # 2%
    # PropagandaOffice and FoundingDescendant are only created by event, never initial weight
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
        # The strategy is determined by the spawning logic, not inherited directly
        # strategy = parent["strategy"]
    if not tag:
        tag = random.choice(["Red", "Blue"])
    if not strategy:
        # Initial agents use the weighted distribution
        strategy = random.choice(strategy_choices_weighted)
    lifespan = min(max(int(np.random.normal(120, 15)), 90), 150)
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
    if strategy == "Accountant":
    agent["laundered_score"] = {}  # Maps other agent IDs to amount laundered for them

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
        elif a["strategy"] == "FoundingDescendant":
             act_a = founding_descendant_strategy(a, b, amem[0], amem[1]) # Use the specific strategy function
        else:
            act_a = strategy_functions[a["strategy"]](a, b, amem[0], amem[1])

        if b["strategy"] == "WSLS":
            act_b = wsls_strategy(b, a, bmem[0], bmem[1], bmem[2])
        elif b["strategy"] == "FoundingDescendant":
            act_b = founding_descendant_strategy(b, a, bmem[0], bmem[1]) # Use the specific strategy function
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
            
        # Accountant laundering logic: when another agent cooperates with an Accountant, Accountant launders score for that agent
        if b["strategy"] == "Accountant" and act_a == "cooperate":
            if a["id"] not in b["laundered_score"]:
                b["laundered_score"][a["id"]] = 0
            b["laundered_score"][a["id"]] += 2  # Launder 2 score per cooperation, or choose your rule
            # Update Accountant's shown score to reflect sum of laundered values
            b["score"] = sum(b["laundered_score"].values())

        if b["strategy"] == "Accountant" and act_a == "cooperate":
            if a["id"] not in b["laundered_score"]:
                b["laundered_score"][a["id"]] = 0
            b["laundered_score"][a["id"]] += 2
            b["score"] = sum(b["laundered_score"].values())

        if a["strategy"] == "Accountant" and act_b == "cooperate":
            if b["id"] not in a["laundered_score"]:
                a["laundered_score"][b["id"]] = 0
            a["laundered_score"][b["id"]] += 2
            a["score"] = sum(a["laundered_score"].values())
        
        if b.get("cluster", -1) != -1:
            cluster_agents = [ag for ag in agent_population if ag.get("cluster", -1) == b["cluster"]]
            true_cluster_score = sum(ag["score"] for ag in cluster_agents) / len(cluster_agents)
            # Store this in a["perceived_cluster_score"], with fast decay unless chosen again
            if not hasattr(a, "perceived_cluster_score"):
                a.perceived_cluster_score = {}
            a.perceived_cluster_score[b["cluster"]] = {"score": true_cluster_score, "decay": 1.0}

        # Decay these at the end of each epoch (faster than normal memory)
        for a in agent_population:
            if hasattr(a, "perceived_cluster_score"):
                for cid in list(a.perceived_cluster_score.keys()):
                    a.perceived_cluster_score[cid]["decay"] *= 0.85  # 3x faster
                    if a.perceived_cluster_score[cid]["decay"] < 0.05:
                        del a.perceived_cluster_score[cid]

        # Define how "visibility" increases: e.g. add +20% per successful cooperation up to 100% clarity
        if a["strategy"] == "ShadowBroker" and act_b == "cooperate" and b.get("cluster", -1) is not None:
            cluster_id = b["cluster"]
            # ShadowBroker builds up knowledge of that cluster's karma difference
            if not hasattr(a, "cluster_karma_visibility"):
                a.cluster_karma_visibility = {}  # {cluster_id: percent_known (0-1)}
            vis = a.cluster_karma_visibility.get(cluster_id, 0.0)
            vis = min(vis + 0.2, 1.0)  # +20% per cooperation, capped at 100%
            a.cluster_karma_visibility[cluster_id] = vis

            # Save the latest karma difference for this cluster for the broker
            cluster_agents = [ag for ag in agent_population if ag.get("cluster", -1) == cluster_id]
            if cluster_agents:
                real_karma = np.mean([ag["karma"] for ag in cluster_agents])
                perceived_karma = np.mean([
                    np.mean(list(ag["perceived_karma"].values())) if ag["perceived_karma"] else 0
                    for ag in cluster_agents
                ])
                karma_delta = real_karma - perceived_karma
                if not hasattr(a, "cluster_karma_delta"):
                    a.cluster_karma_delta = {}
                a.cluster_karma_delta[cluster_id] = karma_delta

        if b["strategy"] == "ShadowBroker" and act_a == "cooperate" and a.get("cluster", -1) is not None:
            broker = b
            cluster_id = a["cluster"]

            # Build up visibility for the broker
            if not hasattr(broker, "cluster_karma_visibility"):
                broker.cluster_karma_visibility = {}
            vis = broker.cluster_karma_visibility.get(cluster_id, 0.0)
            vis = min(vis + 0.2, 1.0)
            broker.cluster_karma_visibility[cluster_id] = vis

            # Save latest karma difference for the broker
            cluster_agents = [ag for ag in agent_population if ag.get("cluster", -1) == cluster_id]
            if cluster_agents:
                real_karma = np.mean([ag["karma"] for ag in cluster_agents])
                perceived_karma = np.mean([
                    np.mean(list(ag["perceived_karma"].values())) if ag["perceived_karma"] else 0
                    for ag in cluster_agents
                ])
                karma_delta = real_karma - perceived_karma
                if not hasattr(broker, "cluster_karma_delta"):
                    broker.cluster_karma_delta = {}
                broker.cluster_karma_delta[cluster_id] = karma_delta

            # Choose a random cluster (that isn't the cooperating agent's cluster) from those broker has visibility into
            visible_clusters = [cid for cid, vis in broker.cluster_karma_visibility.items() if cid != cluster_id and vis > 0]
            if visible_clusters:
                peek_cluster = random.choice(visible_clusters)
                peek_delta = broker.cluster_karma_delta.get(peek_cluster, 0)
                # Agent stores peeked value in a special dict
                if not hasattr(a, "shadow_peeks"):
                    a.shadow_peeks = {}
                a.shadow_peeks[peek_cluster] = {
                    "value": peek_delta,
                    "decay": 1.0  # Start at full, decay 3x as fast as memory
                }

        # If agent a is a Propaganda Office, and cooperates with in-cluster agent b, apply effect
        if a["strategy"] == "PropagandaOffice" and act_a == "cooperate" and a.get("cluster", -1) == b.get("cluster", -1) and a.get("cluster", -1) != -1:
            # Mask negative karma, boost reputation for b (as seen by all)
            masked_karma = b["karma"]
            if masked_karma < 0:
                masked_karma = masked_karma * 0.2  # Mask 80% of negativity
            inflation_bias = 0.5  # or whatever strength you like
            for other_agent in agent_population:
                if other_agent["id"] != b["id"]:
                    previous = other_agent["perceived_karma"].get(b["id"], 0)
                    new_perceived = previous * 0.7 + masked_karma * 0.25 + inflation_bias
                    other_agent["perceived_karma"][b["id"]] = new_perceived

        # And the same for b if b is the Propaganda Office
        if b["strategy"] == "PropagandaOffice" and act_b == "cooperate" and b.get("cluster", -1) == a.get("cluster", -1) and b.get("cluster", -1) != -1:
            masked_karma = a["karma"]
            if masked_karma < 0:
                masked_karma = masked_karma * 0.2
            inflation_bias = 0.5
            for other_agent in agent_population:
                if other_agent["id"] != a["id"]:
                    previous = other_agent["perceived_karma"].get(a["id"], 0)
                    new_perceived = previous * 0.7 + masked_karma * 0.25 + inflation_bias
                    other_agent["perceived_karma"][a["id"]] = new_perceived

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

    # SHADOWBROKER: Own karma always seen as 0 except by other shadowbrokers
    for agent in [a, b]:
        if agent["strategy"] == "ShadowBroker":
            for other_agent in agent_population:
                if other_agent["id"] == agent["id"]:
                    continue  # skip self
                if other_agent["strategy"] == "ShadowBroker":
                    # See true karma difference (or just true karma)
                    other_agent["perceived_karma"][agent["id"]] = agent["karma"]  # or whatever metric you want
                else:
                    other_agent["perceived_karma"][agent["id"]] = 0

# --- Cluster Founding Ideals (NEW for generational power structure modeling) (Moved before the loop) ---
if not hasattr(network, "cluster_founding_ideals"):
    network.cluster_founding_ideals = {}
# Track which clusters have a Propaganda Office (by cluster id)
if not hasattr(network, "cluster_propaganda"):
    network.cluster_propaganda = {}

# ---- Main simulation loop ----
max_epochs = 18000
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
        if a["strategy"] == "ShadowBroker":
            a["cluster"] = None  # or -1, just be consistent everywhere!
        else:
            a["cluster"] = cluster_map.get(a["id"], -1)

    a["influence"] = centrality.get(a["id"], 0)
    # Calculate mean cluster score at the beginning of each epoch
    cluster_scores = {}
    for cluster_id in set(cluster_map.values()):
        if cluster_id != -1:
            cluster_members = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
            if cluster_members:
                cluster_scores[cluster_id] = np.mean([a["score"] for a in cluster_members])

    # --- Initialize Founding Ideals for All Clusters at Epoch 0 (NEW) ---
    # This ensures all clusters present at epoch 0 have a founding ideal recorded.
    # Subsequent ideal layers are added by the Martyr/Founder/Trauma event logic.
    if epoch == 0:
         for cluster_id in set(cluster_map.values()): # Iterate through unique cluster IDs
            if cluster_id != -1: # Only process valid clusters
                cluster_members_initial = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
                if len(cluster_members_initial) > 0: # Ensure the cluster is not empty
                    # Find the agent with the highest initial score as a potential founder
                    founder_candidate = max(cluster_members_initial, key=lambda x: x["score"])
                    # Initialize the list for this cluster if it doesn't exist (shouldn't for epoch 0, but good practice)
                    if cluster_id not in network.cluster_founding_ideals:
                         network.cluster_founding_ideals[cluster_id] = []
                    network.cluster_founding_ideals[cluster_id].append({
                        'epoch': epoch,
                        'ideal': np.mean([a["karma"] for a in cluster_members_initial]),  # Use mean karma as ideal
                        'idolization': founder_candidate["score"],  # Use founder's initial score as idolization
                        'decay': 0.0, # Decay needs to be implemented over time
                        'layer': len(network.cluster_founding_ideals[cluster_id]),
                        'founder': founder_candidate["id"],
                    })

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

    # --- Propaganda Office Cluster Influence (NEW) ---
    # --- Improved Propaganda Office Cluster Influence ---
    for office_agent in agent_population:
        if office_agent["strategy"] == "PropagandaOffice":
            office_cluster_id = office_agent.get("cluster", -1)
            if office_cluster_id == -1:
                continue
            # All agents in this cluster
            cluster_members = [a for a in agent_population if a.get("cluster", -1) == office_cluster_id]
            for target_agent in cluster_members:
                # All other agents (including outsiders)
                for other_agent in agent_population:
                    if other_agent["id"] == target_agent["id"]:
                        continue  # Don't self-perceive
                    # Mask negative karma
                    masked_karma = target_agent["karma"]
                    if masked_karma < 0:
                        masked_karma = masked_karma * 0.2  # Reduce visible negativity by 80%
                    # Inflate perceived karma a bit (bias)
                    inflation_bias = 0.5  # Strong bias, tune as desired
                    # New perceived value = weighted blend of old perception, masked karma, and bias
                    previous = other_agent["perceived_karma"].get(target_agent["id"], 0)
                    new_perceived = previous * 0.7 + masked_karma * 0.25 + inflation_bias
                    other_agent["perceived_karma"][target_agent["id"]] = new_perceived

    # --- Mutation every 30 epochs
    if epoch % 30 == 0 and epoch > 0:
        # Only apply mutation to agents that are NOT FoundingDescendant or PropagandaOffice
        for a in agent_population:
            if a["strategy"] not in ["FoundingDescendant", "PropagandaOffice"]:
                if a["score"] < np.median([x["score"] for x in agent_population]):
                    high_score_agent = max(agent_population, key=lambda x: x["score"])
                    # Mutation can result in any weighted strategy EXCEPT FoundingDescendant or PropagandaOffice
                    available_mutation_strategies = [s for s in strategy_choices_weighted if s not in ["FoundingDescendant", "PropagandaOffice"]]
                    if available_mutation_strategies:
                        a["strategy"] = random.choice([high_score_agent["strategy"], random.choice(available_mutation_strategies)])

    # --- AGING & DEATH (agents die after lifespan, replaced by child agent)
    to_replace = []
    for idx, agent in enumerate(agent_population):
        age = epoch - agent["birth_epoch"]
        if age >= agent["lifespan"]:
            to_replace.append(idx)

    # Important: Sort to_replace in descending order to avoid index issues when removing/replacing
    to_replace.sort(reverse=True)

    for idx in to_replace:
        dead = agent_population[idx]
        try:
            network.remove_node(dead["id"])
        except Exception:
            pass # Node might have already been removed if it was a Propaganda Office that respawned

        cluster_id = cluster_map.get(dead["id"], -1)
        is_propaganda_office = dead["strategy"] == "PropagandaOffice"

        # Propaganda Office Respawn Logic
        if is_propaganda_office and cluster_id != -1:
            # Check if the cluster still exists and has enough members
            current_cluster_agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
            if len(current_cluster_agents) >= 5: # Check for >= 5 members
                # Respawn Propaganda Office in the same cluster
                new_agent = make_agent(agent_id_counter, strategy="PropagandaOffice", parent=dead, birth_epoch=epoch)
                agent_id_counter += 1
                # Replace the dead agent with the new one
                agent_population[idx] = new_agent
                network.add_node(new_agent["id"], tag=new_agent["tag"], strategy=new_agent["strategy"])
                # Update the cluster_propaganda mapping to the new agent's ID
                network.cluster_propaganda[cluster_id] = new_agent["id"]
                print(f"✨ Propaganda Office agent {dead['id']} respawned in Cluster {cluster_id} as agent {new_agent['id']} at epoch {epoch}.")
                continue # Skip the default agent replacement logic

        # 1/3 chance that child rebels and does NOT inherit parent's strategy/trait
        if random.random() < 1/3:
            # Assign random allowed strategy (excluding special strategies)
            allowed_strats = [s for s in strategy_functions.keys() if s not in ["FoundingDescendant", "PropagandaOffice"]]
            child_strategy = random.choice(allowed_strats)
             new_agent = make_agent(agent_id_counter, strategy=child_strategy, birth_epoch=epoch)
        else:
            # Default: inherit parent trait/strategy
            new_agent = make_agent(agent_id_counter, parent=dead, birth_epoch=epoch)
        agent_id_counter += 1
        agent_population[idx] = new_agent
        network.add_node(new_agent["id"], tag=new_agent["tag"], strategy=new_agent["strategy"])

        # FoundingDescendant Spawning Logic (NEW/MODIFIED)
        # Only spawn FoundingDescendant if not a Propaganda Office respawn and in a valid cluster
        if cluster_id != -1:
            # Get all cluster scores and rank them
            cluster_scores = {}
            for cid in set(cluster_map.values()):
                if cid != -1:
                    agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cid]
                    if agents:
                        cluster_scores[cid] = np.mean([a["score"] for a in agents])
            sorted_clusters = sorted(cluster_scores, key=lambda c: cluster_scores[c], reverse=True)
            n = len(sorted_clusters)
            if n == 0:
                descendant_prob = 0.0
            else:
                top_cut = n // 3
                mid_cut = 2 * n // 3

                if cluster_id in sorted_clusters[:top_cut]:
                    # TOP 1/3: maintain >= 10% descendants at all times
                    cluster_agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
                    descendant_agents = [a for a in cluster_agents if a.get("strategy") == "FoundingDescendant"]
                    descendant_ratio = len(descendant_agents) / len(cluster_agents) if cluster_agents else 0

                    if descendant_ratio < 0.10:
                        new_agent["strategy"] = "FoundingDescendant"
                        new_agent["is_descendant"] = True
                        print(f"🧬 Top cluster {cluster_id}: Enforcing descendant quota. Agent {new_agent['id']} is FoundingDescendant at epoch {epoch}.")
                    # else: No forced promotion; child inherits normally or as rebel (handled elsewhere)
                elif cluster_id in sorted_clusters[top_cut:mid_cut]:
                    # MIDDLE 1/3: 5% chance to spawn descendant
                    if random.random() < 0.05:
                        new_agent["strategy"] = "FoundingDescendant"
                        new_agent["is_descendant"] = True
                        print(f"🧬 Middle cluster {cluster_id}: 5% chance descendant promotion. Agent {new_agent['id']} is FoundingDescendant at epoch {epoch}.")
                # Else: bottom 1/3 never gets new descendants (do nothing)

        # --- Check for NEW Propaganda Office Creation (if the dead agent wasn't an office) ---
        # This logic is for creating *new* offices in clusters that don't have one.
        # It should ideally consider the replaced agent's cluster.
        # Let's keep it associated with the *dead* agenif cluster_id != -1:
    # Get all cluster scores and rank them
    cluster_scores = {}
    for cid in set(cluster_map.values()):
        if cid != -1:
            agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cid]
            if agents:
                cluster_scores[cid] = np.mean([a["score"] for a in agents])
    sorted_clusters = sorted(cluster_scores, key=lambda c: cluster_scores[c], reverse=True)
    n = len(sorted_clusters)
    if n == 0:
        descendant_prob = 0.0
    else:
        top_cut = n // 3
        mid_cut = 2 * n // 3

        if cluster_id in sorted_clusters[:top_cut]:
            # TOP 1/3: maintain >= 10% descendants at all times
            cluster_agents = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id]
            descendant_agents = [a for a in cluster_agents if a.get("strategy") == "FoundingDescendant"]
            descendant_ratio = len(descendant_agents) / len(cluster_agents) if cluster_agents else 0

            if descendant_ratio < 0.10:
                new_agent["strategy"] = "FoundingDescendant"
                new_agent["is_descendant"] = True
                print(f"🧬 Top cluster {cluster_id}: Enforcing descendant quota. Agent {new_agent['id']} is FoundingDescendant at epoch {epoch}.")
            # else: No forced promotion; child inherits normally or as rebel (handled elsewhere)
        elif cluster_id in sorted_clusters[top_cut:mid_cut]:
            # MIDDLE 1/3: 5% chance to spawn descendant
            if random.random() < 0.05:
                new_agent["strategy"] = "FoundingDescendant"
                new_agent["is_descendant"] = True
                print(f"🧬 Middle cluster {cluster_id}: 5% chance descendant promotion. Agent {new_agent['id']} is FoundingDescendant at epoch {epoch}.")
        # Else: bottom 1/3 never gets new descendants (do nothing)t's cluster for now.
        if not is_propaganda_office:
            cluster_id_for_new_office = cluster_map.get(dead["id"], -1)
            # Need to check the *current* state of the cluster *after* replacement for size and existing office
            cluster_agents_for_new_office = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id_for_new_office]
            num_clusters = len(set(cluster_map.values()))
            max_offices = max(1, num_clusters // 4)
            existing_offices = sum(1 for a in agent_population if a["strategy"] == "PropagandaOffice")
            # Check if the cluster *already* has a propaganda office (either newly spawned or existing)
            has_office = any(a["strategy"] == "PropagandaOffice" and a.get("cluster", -1) == cluster_id_for_new_office for a in agent_population)

            if (
                cluster_id_for_new_office != -1
                and len(cluster_agents_for_new_office) >= 5 # Minimum size 5 for new office creation
                and not has_office # Ensure the cluster doesn't have an office already
                and existing_offices < max_offices
                # Removed: and cluster_id_for_new_office not in network.cluster_propaganda # Redundant check with has_office
            ):
                # Convert one agent in the cluster to PropagandaOffice
                # Select from the *current* agents in the cluster after replacement
                office_candidate = random.choice(cluster_agents_for_new_office)
                office_candidate["strategy"] = "PropagandaOffice"
                network.cluster_propaganda[cluster_id_for_new_office] = office_candidate["id"]
                # Optionally, set a cluster-level field for propaganda activity
                print(f"🗞️ Cluster {cluster_id_for_new_office}: Propaganda Office established (agent {office_candidate['id']}) at epoch {epoch}.")

        # --- Martyr/Founder/Trauma Event Logic (injects founding ideal on death of highly influential agent or betrayal) ---
        # This logic runs on the death of an agent.
        # We need to ensure the cluster still exists and has members *after* the replacement happens
        # before calculating ideals based on current cluster state.
        cluster_id_martyr = cluster_map.get(dead["id"], -1) # Still use dead agent's cluster ID

        # Check if the cluster still exists and has enough members *after* replacements
        cluster_agents_martyr = [a for a in agent_population if cluster_map.get(a["id"], -1) == cluster_id_martyr]

        if cluster_id_martyr != -1 and len(cluster_agents_martyr) >= 3: # Minimum size 3 to check for betrayal/calculate metrics
            is_martyr = (
                (dead["score"] > np.percentile([a["score"] for a in agent_population], 90)) or
                (dead.get("influence", 0) > np.percentile(list(centrality.values()), 90)) or
                (abs(dead["karma"]) > np.percentile([abs(a["karma"]) for a in agent_population], 90))
            )
            betrayal = False
            # Only check for betrayal if there are enough agents in the cluster to form a trust matrix
            if len(cluster_agents_martyr) >= 3:
                # Build trust matrix only for agents *within* the cluster
                trust_matrix = np.array([[agent["trust"].get(other["id"], 0) for other in cluster_agents_martyr] for agent in cluster_agents_martyr])
                # Avoid division by zero if trust_matrix is empty or all zeros
                if trust_matrix.size > 0:
                     betrayal = np.mean(trust_matrix) < -5  # Tune threshold

            # Only add a new ideal layer if a martyr/betrayal event occurs AND the cluster is being tracked
            if (is_martyr or betrayal) and cluster_id_martyr in network.cluster_founding_ideals:
                 # Ensure founding_ideals_records exists before appending (this is used later for the DataFrame)
                 # This is still not ideal placement for initializing founding_ideals_records, but preserving original flow for now.
                 if 'founding_ideals_records' not in locals():
                      founding_ideals_records = []
                 network.cluster_founding_ideals[cluster_id_martyr].append({
                     'epoch': epoch,
                     'ideal': np.mean([a["karma"] for a in cluster_agents_martyr]), # Calculate ideal based on *current* cluster members
                     'idolization': max([a["score"] for a in cluster_agents_martyr]), # Calculate idolization based on *current* cluster members
                     'decay': 0.0, # Decay needs to be implemented over time
                     'layer': len(network.cluster_founding_ideals[cluster_id_martyr]),
                     'founder': dead["id"] if is_martyr else None, # Founder is the dead agent if it was a martyr
                 })

    # Shadow broker peeks decay 3x as fast as normal memory decay
    for a in agent_population:
        if hasattr(a, "shadow_peeks"):
            for cid in list(a.shadow_peeks.keys()):
                a.shadow_peeks[cid]["decay"] *= 0.85
                if a.shadow_peeks[cid]["decay"] < 0.05:  # Drop if almost faded
                    del a.shadow_peeks[cid]

    if 'total_laundered_history' not in globals():
        total_laundered_history = []
    total_laundered = sum(sum(acc.get("laundered_score", {}).values()) for acc in agent_population if acc.get("strategy") == "Accountant")
    total_laundered_history.append(total_laundered)
    
    # --- TIME-SERIES LOGGING: append to logs at END of each epoch (NEW) ---
    mean_true_karma_ts.append(np.mean([a["karma"] for a in agent_population]))
    mean_perceived_karma_ts.append(np.mean([
        np.mean(list(a["perceived_karma"].values())) if a["perceived_karma"] else 0
        for a in agent_population
    ]))
    mean_score_ts.append(np.mean([a["score"] for a in agent_population]))
    for strat in strategy_karma_ts.keys():
        strat_agents = [a for a in agent_population if a["strategy"] == strat]
        # Ensure there are agents for the strategy before calculating mean
        mean_strat_karma = np.mean([a["karma"] for a in strat_agents]) if strat_agents else np.nan
        strategy_karma_ts[strat].append(mean_strat_karma)

    for agent in agent_population:
        for other_agent in agent_population:
            if other_agent["id"] == agent["id"]:
                continue
            # Memory decay, then update based on Accountant status
            prev = agent["perceived_score"].get(other_agent["id"], 0) * 0.95
            if other_agent["strategy"] == "Accountant":
                if agent["strategy"] == "Accountant":
                    agent["perceived_score"][other_agent["id"]] = other_agent["score"]
                else:
                    agent["perceived_score"][other_agent["id"]] = 0
            else:
                agent["perceived_score"][other_agent["id"]] = other_agent["score"] * 0.95 + prev * 0.05

# === POST-SIMULATION ANALYTICS ===
ostracism_threshold = 3
for a in agent_population:
    given = sum(a["trust"].values())
    received_list = []
    for tid in list(a["trust"].keys()):
        # Check if tid is a valid index in agent_population
        # This check is flawed as trust keys are agent IDs, not indices.
        # A better approach is to look up the agent by ID.
        # This requires iterating through agent_population for each tid, which can be slow.
        # A dictionary mapping IDs to agents might be better if performance is critical.
        # For now, keeping the original logic but acknowledging its potential issues.
        receiving_agent = next((ag for ag in agent_population if ag["id"] == tid), None)
        if receiving_agent and a["id"] in receiving_agent["trust"]:
            received_list.append(receiving_agent["trust"][a["id"]])

    received = sum(received_list)
    # The original code appended to log lists here, but these logs are not used for the final df
    # a["trust_given_log"].append(given)
    # a["trust_received_log"].append(received)
    # Avoid division by zero
    a["trust_reciprocity"] = given / (received + 1e-6) if received != 0 else (1 if given > 0 else 0) # Added check for received == 0 and given > 0

    avg_perceived = np.mean(list(a["perceived_karma"].values())) if a["perceived_karma"] else 0
    # Avoid division by zero
    a["fairness_index"] = a["score"] / (abs(avg_perceived) + 1e-6) if avg_perceived != 0 else (a["score"] if a["score"] >= 0 else 0) # Added abs and handle avg_perceived == 0
    if len([k for k in a["trust"] if a["trust"][k] > 0]) < ostracism_threshold:
        a["ostracized"] = True
    # Avoid division by zero
    a["score_efficiency"] = a["score"] / (abs(a["karma"]) + 1) if a["karma"] != 0 else a["score"] # Added abs and handle a["karma"] == 0
    # The original code calculated trust_reciprocity again here using the full log, but the log was only appended once above.
    # Let's use the value calculated above.
    # a["trust_reciprocity"] = np.mean(a["reciprocity_log"]) if a["reciprocity_log"] else 0

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
if "In-Group Score" in df.columns and "Out-Group Score" in df.columns:
    df["In-Out Karma Ratio"] = df.apply(
        lambda row: round(row["In-Group Score"] / (row["Out-Group Score"] + 1e-6), 2) if row["Out-Group Score"] != 0 else float('inf'),
        axis=1
    )
    print("\nCalculated In-Out Karma Ratio:")
    display(df[["ID", "Tag", "Strategy", "In-Group Score", "Out-Group Score", "In-Out Karma Ratio"]].head())
else:
    print("\nSkipping In-Out Karma Ratio calculation: 'In-Group Score' or 'Out-Group Score' column not found.")

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
# Ensure mask is not empty
if np.any(mask):
    corr, pval = pearsonr(centrality_list[mask], karma_list[mask])
    print(f"\nPearson correlation between Influence Centrality and True Karma: r = {corr:.3f}, p = {pval:.3g}")
else:
    print("\nNot enough data to calculate Pearson correlation between Influence Centrality and True Karma.")

# Optional scatter plot (ethics vs power)
plt.figure(figsize=(8, 5))
# Handle potential NaN values in Cluster before plotting
plot_df = df.dropna(subset=["Cluster"])
if not plot_df.empty:
    scatter = plt.scatter(plot_df["Influence Centrality"], plot_df["True Karma"], c=plot_df["Cluster"], cmap="tab20", s=80, edgecolors="k")
    plt.xlabel("Influence Centrality (Network Power)")
    plt.ylabel("True Karma (Ethics/Morality)")
    plt.title("Ethics vs Power: Influence Centrality vs True Karma")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster ID") # Add colorbar
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to plot Ethics vs Power.")

# --- Cabal Detection Plot ---
plt.figure(figsize=(10, 6))
# Handle potential NaN values in Cluster, Influence Centrality, Score Efficiency, True Karma before plotting
plot_df = df.dropna(subset=["Cluster", "Influence Centrality", "Score Efficiency", "True Karma"])
if not plot_df.empty:
    scatter = plt.scatter(
        plot_df["Influence Centrality"],
        plot_df["Score Efficiency"],
        c=plot_df["True Karma"],
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
else:
     print("Not enough data to plot Cabal Detection.")

# --- Karma Drift Plot for a sample of agents ---
plt.figure(figsize=(12, 6))
# Select sample agents that actually have karma_log data
sample_agents = [a for a in agent_population if a["karma_log"]][:6]
if sample_agents:
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
else:
    print("No agents with karma log data to plot Karma Drift.")

# --- SERIAL MANIPULATORS ANALYTICS ---
# 1. Define a minimum number of steps for stability (e.g., agents with at least 50 logged deltas)
min_steps = 50
serial_manipulator_threshold = 5  # e.g., mean delta > 5

serial_manipulators = []
for a in agent_population:
    deltas = a["karma_perception_delta_log"]
    # Ensure deltas list is not empty before calculations
    if len(deltas) >= min_steps:
        # Count how many times delta was "high" (manipulating) and calculate mean/max
        high_count = sum(np.array(deltas) > serial_manipulator_threshold)
        mean_delta = np.mean(deltas)
        max_delta = np.max(deltas)
        # Ensure mean_delta is not NaN before comparison
        if not np.isnan(mean_delta) and high_count > len(deltas) * 0.5 and mean_delta > serial_manipulator_threshold:  # e.g. more than half the time
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
# Ensure 'Cluster' column exists and is not empty
if "Cluster" in df.columns and not df["Cluster"].dropna().empty:
    for cluster_id in df["Cluster"].unique():
        cluster_members = df[df["Cluster"] == cluster_id]
        if len(cluster_members) < 3: continue  # Ignore tiny clusters

        # Compute mean metrics for the cluster, ensuring non-empty before mean calculation
        mean_perceived_karma = cluster_members["Avg Perceived Karma"].mean() if not cluster_members["Avg Perceived Karma"].empty else np.nan
        mean_connections = cluster_members["Connections"].mean() if not cluster_members["Connections"].empty else np.nan
        mean_score = cluster_members["Score"].mean() if not cluster_members["Score"].empty else np.nan
        mean_delta = cluster_members["Karma-Perception Delta"].mean() if not cluster_members["Karma-Perception Delta"].empty else np.nan
        max_delta = cluster_members["Karma-Perception Delta"].max() if not cluster_members["Karma-Perception Delta"].empty else np.nan

        # Count members with high connections and score but low recent manipulation
        # Ensure df["Score"] and df["Karma-Perception Delta"].abs() are not empty before comparison
        median_score = df["Score"].median() if not df["Score"].empty else 0
        legacy_influence = cluster_members[
            (cluster_members["Score"] > median_score) &
            (cluster_members["Karma-Perception Delta"].abs() < 5)
        ]
        # Shadow cabal flag: high trust/perceived_karma, not manipulating now
        # Ensure mean_perceived_karma is not NaN
        if not np.isnan(mean_perceived_karma) and mean_perceived_karma > 2 and len(legacy_influence) > 0:
            cluster_shadow_report.append({
                "Cluster": cluster_id,
                "Mean Perceived Karma": round(mean_perceived_karma, 2),
                "Mean Connections": round(mean_connections, 2) if not np.isnan(mean_connections) else np.nan,
                "Mean Score": round(mean_score, 2) if not np.isnan(mean_score) else np.nan,
                "Mean Karma-Perception Delta": round(mean_delta, 2) if not np.isnan(mean_delta) else np.nan,
                "Max Delta": round(max_delta, 2) if not np.isnan(max_delta) else np.nan,
                "Legacy Influencers": legacy_influence[["ID", "Strategy", "Score", "Connections", "Avg Perceived Karma"]].values.tolist(),
                "Legacy Influence Count": len(legacy_influence),
                "Members": cluster_members["ID"].tolist()
            })
else:
    print("Skipping Shadow Cabal Detection: 'Cluster' column not found or empty.")

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
# Handle potential NaN values in Cluster before plotting
plot_df = df.dropna(subset=["Cluster"])
if not plot_df.empty:
    sns.scatterplot(
        data=plot_df, x="Influence Centrality", y="True Karma",
        hue=plot_df["Cluster"].apply(lambda x: "Shadow Cabal" if x in shadow_cabal_clusters else "Other"),
        style=plot_df["Cluster"].apply(lambda x: "Shadow Cabal" if x in shadow_cabal_clusters else "Other"),
        s=90
    )
    plt.xlabel("Influence Centrality (Network Power)")
    plt.ylabel("True Karma (Ethics/Morality)")
    plt.title("Shadow Cabal Map: Ethics vs Power (Shadow Cabals Highlighted)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
else:
    print("Not enough data to plot Shadow Cabal Map.")

# --- Founding Ideals & Institutional Memory Export ---
print("\n==== CLUSTER FOUNDING IDEALS ====\n")
# Ensure network.cluster_founding_ideals exists and is not empty
if hasattr(network, "cluster_founding_ideals") and network.cluster_founding_ideals:
    for cid, ideals in network.cluster_founding_ideals.items():
        print(f"Cluster {cid}:")
        for i, ideal in enumerate(ideals):
            # Ensure epoch is within reasonable bounds before calculation
            age = max_epochs - ideal["epoch"] if ideal["epoch"] <= max_epochs else np.nan
            print(f"  Layer {i}: Ideal {ideal['ideal']:.2f}, Idolization {ideal['idolization']:.2f}, Age {age}, Decay {ideal['decay']:.4f}, Founder {ideal['founder']}")
else:
    print("No cluster founding ideals data available.")

# Optional: Export as DataFrame
founding_ideals_records = []
if hasattr(network, "cluster_founding_ideals") and network.cluster_founding_ideals:
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

# Ensure founding_ideals_df is not empty and has required columns for ideal vs current karma analysis
if founding_ideals_df is not None and not founding_ideals_df.empty and all(col in founding_ideals_df.columns for col in ['Cluster', 'Ideal', 'Epoch']):
    # Calculate the mean True Karma for each cluster from df
    # Ensure df has the 'Cluster' and 'True Karma' columns
    if "Cluster" in df.columns and "True Karma" in df.columns:
        current_mean_karma_per_cluster = df.groupby("Cluster")["True Karma"].mean().reset_index()
        current_mean_karma_per_cluster.rename(columns={"True Karma": "Current Mean True Karma"}, inplace=True)

        # Merge with founding_ideals_df
        founding_ideals_with_current = pd.merge(
            founding_ideals_df,
            current_mean_karma_per_cluster,
            on="Cluster",
            how="left" # Use left merge to keep all founding ideals
        )

        # Calculate the difference between founding ideal karma and current mean karma
        # Ensure 'Current Mean True Karma' column exists after merge
        if "Current Mean True Karma" in founding_ideals_with_current.columns:
             founding_ideals_with_current["Ideal-Current Karma Delta"] = (
                 founding_ideals_with_current["Ideal"] - founding_ideals_with_current["Current Mean True Karma"]
             )

             print("\n==== FOUNDING IDEALS VS CURRENT KARMA ====\n")
             display(founding_ideals_with_current)

             # Visualization code for ideal vs current karma
             if founding_ideals_with_current is not None and not founding_ideals_with_current.empty and all(col in founding_ideals_with_current.columns for col in ['Cluster', 'Ideal', 'Current Mean True Karma']):
                 plt.figure(figsize=(10, 6))
                 scatter = plt.scatter(
                     founding_ideals_with_current["Ideal"],
                     founding_ideals_with_current["Current Mean True Karma"],
                     c=founding_ideals_with_current["Cluster"],
                     cmap="tab20",
                     s=100,
                     edgecolors="k"
                 )
                 plt.xlabel("Founding Ideal Karma (Epoch 0)")
                 plt.ylabel("Final Mean Cluster True Karma")
                 plt.title("Founding Ideal Karma vs. Final Mean Cluster True Karma (by Founder Strategy)")
                 plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add a line at 0 karma
                 plt.axvline(0, color='grey', linestyle='--', linewidth=0.8) # Add a line at 0 karma
                 plt.grid(True, linestyle='--', alpha=0.6)
                 plt.legend(title="Founder Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
                 plt.tight_layout()
                 plt.show()
             else:
                 print("Not enough data to plot Founding Ideal Karma vs. Current Mean Cluster Karma.")
        else:
             print("Skipping Ideal vs Current Karma analysis and visualization: 'Current Mean True Karma' column not found after merging.")
    else:
        print("Skipping Ideal vs Current Karma analysis and visualization: 'Cluster' or 'True Karma' column not found in main DataFrame.")

else:
    print("Founding ideals DataFrame is empty or missing required columns. Skipping founding ideal analysis and visualization.")

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
    # Ensure df has 'Cluster' column before filtering
    clusters_without_office = [c for c in df["Cluster"].unique() if c not in clusters_with_office] if "Cluster" in df.columns else []

    def cluster_summary(cluster_id):
        # Ensure df has 'Cluster' column before filtering
        members = df[df["Cluster"] == cluster_id] if "Cluster" in df.columns else pd.DataFrame()
        # Ensure members DataFrame is not empty before calculating means
        return {
            "Cluster": cluster_id,
            "Members": len(members),
            "Mean True Karma": round(members["True Karma"].mean(), 2) if not members["True Karma"].empty else np.nan,
            "Mean Perceived Karma": round(members["Avg Perceived Karma"].mean(), 2) if not members["Avg Perceived Karma"].empty else np.nan,
            "Mean Delta": round(members["Karma-Perception Delta"].mean(), 2) if not members["Karma-Perception Delta"].empty else np.nan,
            "Mean Score": round(members["Score"].mean(), 2) if not members["Score"].empty else np.nan,
            "Total Influence": round(members["Influence Centrality"].sum(), 3) if not members["Influence Centrality"].empty else np.nan,
            "Office Agent": next((a["ID"] for a in prop_offices if a["cluster"] == cluster_id), None) # Find the ID of the office agent in this cluster
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
    # Ensure df has 'Cluster' column before filtering
    with_office = df[df["Cluster"].isin(clusters_with_office)] if "Cluster" in df.columns else pd.DataFrame()
    without_office = df[~df["Cluster"].isin(clusters_with_office)] if "Cluster" in df.columns else pd.DataFrame()

    print("\n--- Aggregate Comparison (mean per agent) ---")
    # Ensure DataFrames are not empty before calculating means
    print(f"Mean Perceived Karma (Propaganda Office): {with_office['Avg Perceived Karma'].mean():.2f}" if not with_office['Avg Perceived Karma'].empty else "N/A")
    print(f"Mean Perceived Karma (No Office): {without_office['Avg Perceived Karma'].mean():.2f}" if not without_office['Avg Perceived Karma'].empty else "N/A")
    print(f"Mean Karma-Perception Delta (Propaganda Office): {with_office['Karma-Perception Delta'].mean():.2f}" if not with_office['Karma-Perception Delta'].empty else "N/A")
    print(f"Mean Karma-Perception Delta (No Office): {without_office['Karma-Perception Delta'].mean():.2f}" if not without_office['Karma-Perception Delta'].empty else "N/A")
    print(f"Mean Score (Propaganda Office): {with_office['Score'].mean():.2f}" if not with_office['Score'].empty else "N/A")
    print(f"Mean Score (No Office): {without_office['Score'].mean():.2f}" if not without_office['Score'].empty else "N/A")

    # 4. Optionally, visualize cluster means by office status
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    # Ensure DataFrames are not empty before calculating means
    means = [
        with_office['Avg Perceived Karma'].mean() if not with_office['Avg Perceived Karma'].empty else np.nan,
        without_office['Avg Perceived Karma'].mean() if not without_office['Avg Perceived Karma'].empty else np.nan,
        with_office['Karma-Perception Delta'].mean() if not with_office['Karma-Perception Delta'].empty else np.nan,
        without_office['Karma-Perception Delta'].mean() if not without_office['Karma-Perception Delta'].empty else np.nan
    ]
    # Filter out NaN means for plotting
    plot_means = [m for m in means if not np.isnan(m)]
    bar_labels = [
        "With Office: Perceived Karma",
        "No Office: Perceived Karma",
        "With Office: Δ Karma-Percept",
        "No Office: Δ Karma-Percept"
    ]
    # Filter labels to match plot_means
    plot_labels = [label for label, mean in zip(bar_labels, means) if not np.isnan(mean)]

    if plot_means:
        ax.bar(plot_labels, plot_means, color=['#7dafff', '#c0c0c0', '#7dafff', '#c0c0c0'][:len(plot_labels)])
        ax.set_ylabel("Mean Value")
        ax.set_title("Propaganda Office Effect on Cluster Reputation")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data to plot Propaganda Office Effect on Cluster Reputation.")

    print("\n--- Accountant Laundering Report ---")
    for a in agent_population:
        if a.get("strategy") == "Accountant":
            print(f"Accountant {a['id']} | Shown Score: {a['score']}")
            total_laundered = sum(a.get("laundered_score", {}).values())
            print(f"    Total Laundered: {total_laundered}")
            print("    Laundered For:")
            for client_id, amt in a.get("laundered_score", {}).items():
                print(f"        Agent {client_id}: {amt}")

    print("\n--- Agent Laundering Received ---")
    agent_laundered_received = {a["id"]: 0 for a in agent_population}
    for acc in agent_population:
        if acc.get("strategy") == "Accountant":
            for client_id, amt in acc.get("laundered_score", {}).items():
                agent_laundered_received[client_id] += amt
    for a in agent_population:
        if agent_laundered_received[a["id"]] > 0:
            print(f"Agent {a['id']} received laundered score: {agent_laundered_received[a['id']]}")

    print("\n--- Accountant/ShadowBroker Exposure Map ---")
    for agent in agent_population:
        if agent.get("strategy") == "Accountant":
            can_see = [other["id"] for other in agent_population if other.get("strategy") == "Accountant" and other["id"] != agent["id"]]
            print(f"Accountant {agent['id']} sees true score of: {can_see}")
        if agent.get("strategy") == "ShadowBroker":
            can_see = [other["id"] for other in agent_population if other.get("strategy") == "ShadowBroker" and other["id"] != agent["id"]]
            print(f"ShadowBroker {agent['id']} sees true karma of: {can_see}")

    print("\n--- Exposure Map ---")
    for a in agent_population:
        if a.get("strategy") == "Accountant":
            print(f"Accountant {a['id']} sees true score only of Accountants: {[b['id'] for b in agent_population if b.get('strategy') == 'Accountant' and b['id'] != a['id']]}")
        if a.get("strategy") == "ShadowBroker":
            print(f"ShadowBroker {a['id']} sees true karma only of ShadowBrokers: {[b['id'] for b in agent_population if b.get('strategy') == 'ShadowBroker' and b['id'] != a['id']]}")
    
    import pandas as pd

    # Build laundering matrix (rows = Accountant, cols = Agent)
    acc_ids = [a["id"] for a in agent_population if a.get("strategy") == "Accountant"]
    agent_ids = [a["id"] for a in agent_population]
    launder_matrix = pd.DataFrame(0, index=acc_ids, columns=agent_ids)
    for acc in agent_population:
        if acc.get("strategy") == "Accountant":
            for client_id, amt in acc.get("laundered_score", {}).items():
                launder_matrix.loc[acc["id"], client_id] = amt

    print("\n--- Accountant Laundering Matrix (rows=Accountant, cols=Agent) ---")
    print(launder_matrix)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(total_laundered_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total Laundered Score")
    plt.title("Laundered Score Over Time")
    plt.show()








