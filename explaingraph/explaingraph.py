import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder

class ExplainGraph:
    """
    ExplainGraph is a simple interpretable model that predicts outcomes based on feature importance.
    It provides explanations for predictions and visualizes feature influence using graphs.
    """
    def __init__(self):
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """Computes feature influence dynamically using a gradient-inspired approach."""
        self.feature_importances_ = {}
        epsilon = 1e-4  # Small perturbation for numerical stability
        for feature in X.columns:
            perturbation = X.copy()
            perturbation[feature] += epsilon
            correlation = np.corrcoef(perturbation[feature], y)[0, 1]
            self.feature_importances_[feature] = abs(correlation) if not np.isnan(correlation) else 0
        total_importance = sum(self.feature_importances_.values())
        if total_importance > 0:
            for key in self.feature_importances_:
                self.feature_importances_[key] /= total_importance
    
    def predict(self, X):
        """Predicts outcomes and calculates confidence scores."""
        decision_score = np.dot(X, list(self.feature_importances_.values()))
        confidence = np.abs(decision_score) / np.max(np.abs(decision_score)) if np.max(np.abs(decision_score)) > 0 else 0
        return (decision_score > np.median(decision_score)).astype(int), confidence
    
    def explain(self, X):
        """Generates human-readable explanations for predictions."""
        explanations = []
        for _, sample in X.iterrows():
            contributions = {feat: sample[feat] * self.feature_importances_[feat] for feat in X.columns}
            explanation = " + ".join([f"{feat} ({contrib:.2f})" for feat, contrib in contributions.items()])
            max_feat = max(contributions, key=contributions.get)
            explanations.append(f"Decision heavily influenced by {max_feat}. Breakdown: {explanation}")
        return explanations
    
    def visualize_explanation(self, X):
        """Visualizes feature influence using a directed graph."""
        G = nx.DiGraph()
        cmap = plt.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        
        for _, sample in X.iterrows():
            for feature, importance in self.feature_importances_.items():
                weight = sample[feature] * importance
                if weight != 0:
                    G.add_edge(feature, 'Prediction', weight=weight)
        
        pos = nx.spring_layout(G)
        edges = G.edges(data=True)
        weights = [d['weight'] for (_, _, d) in edges]
        edge_colors = [cmap(norm(w)) for w in weights]
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, arrows=True, edge_cmap=cmap)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='black')
        plt.title("Feature Influence on Prediction")
        
        if weights:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(weights)
            plt.colorbar(sm, label="Feature Contribution Strength")
        plt.show()
