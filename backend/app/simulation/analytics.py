"""
SwarmIQ — Simulation Analytics
Processes WorldState to find echo chambers, opinion drift, and influence.
"""

import logging
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

from .world import WorldState

logger = logging.getLogger("swarmiq.simulation.analytics")


class SimulationAnalytics:
    """Calculates metrics on the simulation state."""

    @staticmethod
    def detect_echo_chambers(state: WorldState) -> None:
        """
        Cluster agents by their opinion vectors to find echo chambers.
        Updates state.opinion_clusters and state.echo_chambers in place.
        """
        if not state.agents or not state.active_topics:
            return

        # 1. Build observation matrix
        agent_ids = list(state.agents.keys())
        X = np.array([state.agents[aid].opinion_vector() for aid in agent_ids])
        
        if len(agent_ids) < 3:
            return

        # 2. Cluster using K-Means
        n_clusters = min(3, len(agent_ids))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        try:
            labels = kmeans.fit_predict(X)
        except Exception as e:
            logger.warning("KMeans clustering failed: %s", e)
            return

        # 3. Group agents by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(agent_ids[idx])

        state.opinion_clusters = list(clusters.values())

        # 4. Identify echo chambers (clusters with high internal trust and extreme opinions)
        # Simplified: identify clusters with > 50% polarization on at least one topic
        echo_chambers = []
        for c_id, members in clusters.items():
            if len(members) < 2:
                continue
                
            # calc mean opinion
            c_ops = np.mean([state.agents[aid].opinion_vector() for aid in members], axis=0)
            
            # If any topic is highly polarized (> 0.7 or < -0.7)
            extreme_topics = []
            for t_idx, val in enumerate(c_ops):
                if abs(val) > 0.7:
                    extreme_topics.append((state.active_topics[t_idx], round(val, 3)))
                    
            if extreme_topics:
                # Calculate internal variance
                c_var = np.var([state.agents[aid].opinion_vector() for aid in members], axis=0).mean()
                if c_var < 0.1:  # Low variance = echo chamber
                    echo_chambers.append({
                        "size": len(members),
                        "polarized_topics": extreme_topics,
                        "members": members[:5]  # store sample
                    })
                    
        state.echo_chambers = echo_chambers

    @staticmethod
    def calculate_influence(state: WorldState) -> None:
        """
        Update influence_score for each agent based on PageRank of the trust network.
        """
        import networkx as nx
        
        if not state.agents:
            return

        # Build directed trust graph
        G = nx.DiGraph()
        for aid, agent in state.agents.items():
            G.add_node(aid)
            for target_id, trust_val in agent.trust_network.items():
                if target_id in state.agents and trust_val > 0.1:
                    G.add_edge(aid, target_id, weight=trust_val)
                    
        if G.number_of_edges() == 0:
            return

        try:
            # PageRank: nodes that are trusted by many (or heavily trusted) rank higher
            pr = nx.pagerank(G, weight="weight")
            
            # Normalize scores to 0-1 based on max
            max_pr = max(pr.values()) if pr else 1.0
            if max_pr == 0:
                max_pr = 1.0
                
            for aid, score in pr.items():
                # Smooth update
                curr = state.agents[aid].influence_score
                state.agents[aid].influence_score = 0.8 * curr + 0.2 * (score / max_pr)
        except Exception as e:
            logger.warning("PageRank influence calc failed: %s", e)

    @staticmethod
    def polarization_index(state: WorldState, topic: str) -> float:
        """
        Calculate polarization index [0.0, 1.0] for a specific topic.
        0 = consensus, 1 = perfectly polarized bimodal distribution.
        """
        if topic not in state.active_topics or not state.agents:
            return 0.0
            
        ops = np.array([a.opinions.get(topic, 0.0) for a in state.agents.values()])
        variance = np.var(ops)
        
        # Max theoretical variance of [-1, 1] is 1.0 (50% at -1, 50% at 1)
        polarization = variance / 1.0
        return float(min(1.0, polarization))
