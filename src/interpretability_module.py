"""
Interpretability Module for ED-AI Triage System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class InterpretabilityEngine:
    """Comprehensive interpretability engine for triage predictions"""

    def __init__(self, model, feature_names, training_data=None):
        """
        Initialize interpretability engine

        Args:
            model: Trained ML model
            feature_names: List of feature names
            training_data: Training data for LIME (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.integrated_gradients = None

    def setup_shap(self, background_data=None, max_evals=1000):
        """Setup SHAP explainer"""
        try:
            if background_data is None and self.training_data is not None:
                # Use subset of training data as background
                background_size = min(100, len(self.training_data))
                background_indices = np.random.choice(
                    len(self.training_data), background_size, replace=False
                )
                background_data = self.training_data[background_indices]

            self.shap_explainer = shap.Explainer(
                self.model, background_data, max_evals=max_evals
            )
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print(f"SHAP setup failed: {e}")
            # Fallback to TreeExplainer if available
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                print("Using TreeExplainer as fallback")
            except:
                print("SHAP explainer setup failed completely")

    def setup_lime(self):
        """Setup LIME explainer"""
        try:
            if self.training_data is not None:
                self.lime_explainer = LimeTabularExplainer(
                    training_data=self.training_data,
                    feature_names=self.feature_names,
                    class_names=['Non-urgent', 'Urgent'],
                    mode='classification'
                )
                print("LIME explainer initialized successfully")
            else:
                print("Training data required for LIME")
        except Exception as e:
            print(f"LIME setup failed: {e}")

    def setup_integrated_gradients(self, baseline=None):
        """Setup Integrated Gradients for neural networks"""
        try:
            if hasattr(self.model, 'predict_proba'):
                # For sklearn-like models, create a wrapper
                def model_wrapper(x):
                    return self.model.predict_proba(x)[:, 1]

                self.integrated_gradients = IntegratedGradients(model_wrapper)

                if baseline is None:
                    baseline = np.zeros((1, len(self.feature_names)))

                self.baseline = baseline
                print("Integrated Gradients initialized successfully")
        except Exception as e:
            print(f"Integrated Gradients setup failed: {e}")

    def explain_prediction_shap(self, sample_data, max_evals=1000):
        """
        Generate SHAP explanation for a single prediction

        Args:
            sample_data: Input sample (numpy array)
            max_evals: Maximum evaluations for SHAP

        Returns:
            SHAP explanation object
        """
        if self.shap_explainer is None:
            self.setup_shap()

        if self.shap_explainer is not None:
            try:
                explanation = self.shap_explainer(sample_data.reshape(1, -1),
                                                max_evals=max_evals)
                return explanation
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
                return None
        return None

    def explain_prediction_lime(self, sample_data, num_features=10):
        """
        Generate LIME explanation for a single prediction

        Args:
            sample_data: Input sample (numpy array)
            num_features: Number of features to include in explanation

        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            self.setup_lime()

        if self.lime_explainer is not None:
            try:
                explanation = self.lime_explainer.explain_instance(
                    data_row=sample_data,
                    predict_fn=self.model.predict_proba,
                    num_features=num_features
                )
                return explanation
            except Exception as e:
                print(f"LIME explanation failed: {e}")
                return None
        return None

    def explain_prediction_integrated_gradients(self, sample_data):
        """
        Generate Integrated Gradients explanation

        Args:
            sample_data: Input sample (numpy array)

        Returns:
            Attribution scores
        """
        if self.integrated_gradients is None:
            self.setup_integrated_gradients()

        if self.integrated_gradients is not None:
            try:
                sample_tensor = torch.tensor(sample_data, dtype=torch.float32)
                baseline_tensor = torch.tensor(self.baseline, dtype=torch.float32)

                attributions = self.integrated_gradients.attribute(
                    sample_tensor, baseline_tensor, target=0
                )
                return attributions.detach().numpy()
            except Exception as e:
                print(f"Integrated Gradients explanation failed: {e}")
                return None
        return None

    def generate_comprehensive_explanation(self, sample_data, sample_idx=0):
        """
        Generate comprehensive explanation using multiple methods

        Args:
            sample_data: Input sample
            sample_idx: Sample index for display

        Returns:
            Dictionary with all explanations
        """
        explanations = {
            'sample_index': sample_idx,
            'feature_names': self.feature_names,
            'sample_values': sample_data
        }

        # SHAP explanation
        shap_exp = self.explain_prediction_shap(sample_data)
        if shap_exp is not None:
            explanations['shap_values'] = shap_exp.values[0]
            explanations['shap_base_value'] = shap_exp.base_values[0]
            explanations['shap_data'] = shap_exp.data[0]

        # LIME explanation
        lime_exp = self.explain_prediction_lime(sample_data)
        if lime_exp is not None:
            explanations['lime_explanation'] = lime_exp.as_list()
            explanations['lime_prediction'] = lime_exp.predict_proba

        # Integrated Gradients
        ig_attr = self.explain_prediction_integrated_gradients(sample_data)
        if ig_attr is not None:
            explanations['integrated_gradients'] = ig_attr[0]

        return explanations

    def plot_comprehensive_explanation(self, explanations, save_path=None):
        """
        Create comprehensive explanation visualization

        Args:
            explanations: Explanation dictionary from generate_comprehensive_explanation
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comprehensive Explanation - Sample {explanations["sample_index"]}',
                    fontsize=16)

        # Plot 1: SHAP Waterfall
        ax1 = axes[0, 0]
        if 'shap_values' in explanations:
            shap_values = explanations['shap_values']
            feature_names = explanations['feature_names']
            base_value = explanations['shap_base_value']

            # Create waterfall plot data
            sorted_idx = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10 features
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_values = shap_values[sorted_idx]

            cumulative = base_value
            bars = []
            for i, (feature, value) in enumerate(zip(sorted_features, sorted_values)):
                bars.append((feature, value, cumulative))
                cumulative += value

            colors = ['red' if v < 0 else 'blue' for v in sorted_values]
            ax1.barh(range(len(bars)), [b[1] for b in bars],
                    left=[b[2] - b[1] for b in bars], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(bars)))
            ax1.set_yticklabels([b[0] for b in bars])
            ax1.set_xlabel('SHAP Value')
            ax1.set_title('SHAP Waterfall Plot')
            ax1.axvline(x=base_value, color='black', linestyle='--', alpha=0.5,
                       label=f'Base Value: {base_value:.3f}')
            ax1.legend()

        # Plot 2: LIME Explanation
        ax2 = axes[0, 1]
        if 'lime_explanation' in explanations:
            lime_exp = explanations['lime_explanation']
            features, values = zip(*lime_exp)

            colors = ['red' if v < 0 else 'blue' for v in values]
            ax2.barh(range(len(features)), values, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_xlabel('LIME Weight')
            ax2.set_title('LIME Feature Contributions')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Plot 3: Feature Values
        ax3 = axes[1, 0]
        feature_names = explanations['feature_names']
        sample_values = explanations['sample_values']

        # Show top contributing features
        if 'shap_values' in explanations:
            shap_values = explanations['shap_values']
            top_indices = np.argsort(np.abs(shap_values))[::-1][:10]
            top_features = [feature_names[i] for i in top_indices]
            top_values = sample_values[top_indices]

            ax3.barh(range(len(top_features)), top_values, alpha=0.7)
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features)
            ax3.set_xlabel('Feature Value')
            ax3.set_title('Top Feature Values')
        else:
            # Fallback to all features
            ax3.barh(range(len(feature_names)), sample_values, alpha=0.7)
            ax3.set_yticks(range(len(feature_names)))
            ax3.set_yticklabels(feature_names)
            ax3.set_xlabel('Feature Value')
            ax3.set_title('Feature Values')

        # Plot 4: Integrated Gradients
        ax4 = axes[1, 1]
        if 'integrated_gradients' in explanations:
            ig_values = explanations['integrated_gradients']
            top_indices = np.argsort(np.abs(ig_values))[::-1][:10]
            top_features = [feature_names[i] for i in top_indices]
            top_ig = ig_values[top_indices]

            colors = ['red' if v < 0 else 'blue' for v in top_ig]
            ax4.barh(range(len(top_features)), top_ig, color=colors, alpha=0.7)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features)
            ax4.set_xlabel('Integrated Gradients')
            ax4.set_title('Integrated Gradients Attribution')
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive explanation plot saved to {save_path}")

        plt.show()

    def generate_explanation_report(self, explanations):
        """
        Generate textual explanation report

        Args:
            explanations: Explanation dictionary

        Returns:
            Formatted explanation report
        """
        report = []
        report.append("# Prediction Explanation Report")
        report.append("=" * 50)
        report.append("")

        sample_idx = explanations['sample_index']
        report.append(f"## Sample {sample_idx} Explanation")
        report.append("")

        # Model prediction
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(
                explanations['sample_values'].reshape(1, -1)
            )[0]
            report.append("### Model Prediction")
            report.append(f"- Urgent Probability: {pred_proba[1]:.3f}")
            report.append(f"- Non-urgent Probability: {pred_proba[0]:.3f}")
            report.append(f"- Predicted Class: {'Urgent' if pred_proba[1] > 0.5 else 'Non-urgent'}")
            report.append("")

        # SHAP Explanation
        if 'shap_values' in explanations:
            report.append("### SHAP Explanation")
            shap_values = explanations['shap_values']
            base_value = explanations['shap_base_value']
            feature_names = explanations['feature_names']

            report.append(f"- Base Value (Expected Log-Odds): {base_value:.3f}")
            report.append("- Top Contributing Features:")

            # Sort by absolute SHAP value
            sorted_idx = np.argsort(np.abs(shap_values))[::-1]
            for i in range(min(10, len(sorted_idx))):
                idx = sorted_idx[i]
                feature = feature_names[idx]
                shap_val = shap_values[idx]
                direction = "increases" if shap_val > 0 else "decreases"
                report.append(f"  - {feature}: {shap_val:.3f} ({direction} urgency)")

            report.append("")

        # LIME Explanation
        if 'lime_explanation' in explanations:
            report.append("### LIME Explanation")
            lime_exp = explanations['lime_explanation']

            report.append("- Local Feature Contributions:")
            for feature, weight in lime_exp:
                direction = "supports" if weight > 0 else "opposes"
                report.append(f"  - {feature}: {weight:.3f} ({direction} urgent prediction)")

            report.append("")

        # Key Insights
        report.append("### Key Insights")
        report.append("")

        if 'shap_values' in explanations:
            # Find most important features
            shap_values = explanations['shap_values']
            feature_names = explanations['feature_names']

            max_shap_idx = np.argmax(np.abs(shap_values))
            max_feature = feature_names[max_shap_idx]
            max_shap = shap_values[max_shap_idx]

            report.append(f"- **Most Influential Feature**: {max_feature}")
            report.append(f"  - Contribution: {max_shap:.3f}")
            report.append(f"  - Direction: {'Increases' if max_shap > 0 else 'Decreases'} urgency")

            # Clinical interpretation
            if 'shock_index' in max_feature.lower():
                report.append("  - **Clinical Note**: Shock index > 0.7 indicates potential shock")
            elif 'temperature' in max_feature.lower():
                report.append("  - **Clinical Note**: Fever (>38°C) may indicate infection")
            elif 'pain' in max_feature.lower():
                report.append("  - **Clinical Note**: Severe pain requires immediate attention")

        return "\n".join(report)


def generate_counterfactual_explanation(sample_data, model, feature_names,
                                      target_class=1, max_changes=3):
    """
    Generate counterfactual explanation

    Args:
        sample_data: Original sample
        model: Trained model
        feature_names: Feature names
        target_class: Target class to flip to
        max_changes: Maximum number of feature changes

    Returns:
        Dictionary with counterfactual explanation
    """
    original_pred = model.predict_proba(sample_data.reshape(1, -1))[0]
    original_class = np.argmax(original_pred)

    if original_class == target_class:
        return {"message": "Already in target class"}

    counterfactual = sample_data.copy()
    changes = []

    # Try changing features one by one
    feature_importance = np.abs(model.feature_importances_) if hasattr(model, 'feature_importances_') else np.ones(len(feature_names))

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]

    for idx in sorted_indices:
        if len(changes) >= max_changes:
            break

        feature_name = feature_names[idx]
        original_value = sample_data[idx]

        # Try different perturbations
        for direction in [-1, 1]:
            for scale in [0.1, 0.5, 1.0]:
                new_value = original_value + direction * scale * np.std(sample_data)
                temp_sample = counterfactual.copy()
                temp_sample[idx] = new_value

                new_pred = model.predict_proba(temp_sample.reshape(1, -1))[0]
                new_class = np.argmax(new_pred)

                if new_class == target_class:
                    changes.append({
                        'feature': feature_name,
                        'original_value': original_value,
                        'new_value': new_value,
                        'change': new_value - original_value
                    })
                    counterfactual = temp_sample
                    break
            if len(changes) > len(changes) - 1:  # If we added a change
                break

    return {
        'original_prediction': original_pred,
        'target_class': target_class,
        'changes': changes,
        'counterfactual_prediction': model.predict_proba(counterfactual.reshape(1, -1))[0],
        'success': len(changes) > 0
    }


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create interpretability engine
    engine = InterpretabilityEngine(model, feature_names, X)

    # Setup explainers
    engine.setup_shap()
    engine.setup_lime()

    # Generate explanation for a sample
    sample_idx = 0
    sample_data = X[sample_idx]

    explanations = engine.generate_comprehensive_explanation(sample_data, sample_idx)

    # Generate report
    report = engine.generate_explanation_report(explanations)
    print(report)

    # Plot comprehensive explanation
    engine.plot_comprehensive_explanation(explanations, save_path='../reports/explanation_sample_0.png')

    # Generate counterfactual
    counterfactual = generate_counterfactual_explanation(
        sample_data, model, feature_names, target_class=1
    )
    print("\nCounterfactual Explanation:")
    print(counterfactual)
