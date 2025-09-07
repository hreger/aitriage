"""
Fairness Assessment and Debiasing Module for ED-AI Triage System
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FairnessAssessor:
    """Comprehensive fairness assessment for triage predictions"""

    def __init__(self, protected_attributes: List[str] = None):
        """
        Initialize fairness assessor

        Args:
            protected_attributes: List of protected attribute names
        """
        if protected_attributes is None:
            protected_attributes = ['gender', 'race', 'age_group', 'insurance_type']
        self.protected_attributes = protected_attributes

    def calculate_disparate_impact(self, predictions: np.ndarray,
                                 protected_attr: np.ndarray,
                                 privileged_group: str = None) -> Dict:
        """
        Calculate disparate impact ratio

        Args:
            predictions: Model predictions (0/1)
            protected_attr: Protected attribute values
            privileged_group: Reference group for comparison

        Returns:
            Dictionary with disparate impact metrics
        """
        unique_groups = np.unique(protected_attr)
        group_rates = {}

        for group in unique_groups:
            mask = protected_attr == group
            if np.sum(mask) > 0:
                positive_rate = np.mean(predictions[mask])
                group_rates[group] = positive_rate

        # Calculate disparate impact ratios
        di_ratios = {}
        if privileged_group and privileged_group in group_rates:
            priv_rate = group_rates[privileged_group]
            if priv_rate > 0:
                for group, rate in group_rates.items():
                    if group != privileged_group:
                        di_ratios[f"{group}/{privileged_group}"] = rate / priv_rate

        return {
            'group_rates': group_rates,
            'disparate_impact_ratios': di_ratios,
            'fairness_threshold': 0.8  # Common threshold for disparate impact
        }

    def calculate_equalized_odds(self, predictions: np.ndarray,
                               true_labels: np.ndarray,
                               protected_attr: np.ndarray) -> Dict:
        """
        Calculate equalized odds metrics

        Args:
            predictions: Model predictions (0/1)
            true_labels: True labels (0/1)
            protected_attr: Protected attribute values

        Returns:
            Dictionary with equalized odds metrics
        """
        unique_groups = np.unique(protected_attr)
        group_metrics = {}

        for group in unique_groups:
            mask = protected_attr == group
            if np.sum(mask) > 0:
                group_pred = predictions[mask]
                group_true = true_labels[mask]

                tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate

                group_metrics[group] = {
                    'true_positive_rate': tpr,
                    'false_positive_rate': fpr,
                    'sample_size': np.sum(mask)
                }

        return group_metrics

    def assess_fairness(self, predictions: np.ndarray,
                       true_labels: np.ndarray,
                       protected_attrs: Dict[str, np.ndarray],
                       privileged_groups: Dict[str, str] = None) -> Dict:
        """
        Comprehensive fairness assessment

        Args:
            predictions: Model predictions
            true_labels: True labels
            protected_attrs: Dictionary of protected attributes
            privileged_groups: Dictionary mapping attribute names to privileged groups

        Returns:
            Comprehensive fairness assessment results
        """
        if privileged_groups is None:
            privileged_groups = {
                'gender': 'Male',
                'race': 'White',
                'age_group': '31-50',
                'insurance_type': 'Private'
            }

        results = {}

        for attr_name, attr_values in protected_attrs.items():
            print(f"Assessing fairness for {attr_name}...")

            # Disparate impact
            di_results = self.calculate_disparate_impact(
                predictions, attr_values,
                privileged_groups.get(attr_name)
            )

            # Equalized odds
            eo_results = self.calculate_equalized_odds(
                predictions, true_labels, attr_values
            )

            results[attr_name] = {
                'disparate_impact': di_results,
                'equalized_odds': eo_results
            }

        return results

    def plot_fairness_analysis(self, fairness_results: Dict,
                             save_path: str = None):
        """
        Create comprehensive fairness visualization

        Args:
            fairness_results: Results from assess_fairness
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fairness Analysis Dashboard', fontsize=16)

        # Plot 1: Disparate Impact Ratios
        ax1 = axes[0, 0]
        di_data = []
        for attr, results in fairness_results.items():
            for ratio_name, ratio in results['disparate_impact']['disparate_impact_ratios'].items():
                di_data.append({'attribute': attr, 'comparison': ratio_name, 'ratio': ratio})

        if di_data:
            di_df = pd.DataFrame(di_data)
            sns.barplot(data=di_df, x='comparison', y='ratio', ax=ax1)
            ax1.axhline(y=0.8, color='red', linestyle='--', label='Fairness threshold')
            ax1.set_title('Disparate Impact Ratios')
            ax1.set_ylabel('Ratio')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Group Positive Rates
        ax2 = axes[0, 1]
        group_data = []
        for attr, results in fairness_results.items():
            for group, rate in results['disparate_impact']['group_rates'].items():
                group_data.append({'attribute': attr, 'group': group, 'rate': rate})

        if group_data:
            group_df = pd.DataFrame(group_data)
            sns.barplot(data=group_df, x='group', y='rate', hue='attribute', ax=ax2)
            ax2.set_title('Positive Prediction Rates by Group')
            ax2.set_ylabel('Positive Rate')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)

        # Plot 3: True Positive Rates
        ax3 = axes[1, 0]
        tpr_data = []
        for attr, results in fairness_results.items():
            for group, metrics in results['equalized_odds'].items():
                tpr_data.append({
                    'attribute': attr,
                    'group': group,
                    'tpr': metrics['true_positive_rate'],
                    'sample_size': metrics['sample_size']
                })

        if tpr_data:
            tpr_df = pd.DataFrame(tpr_data)
            sns.barplot(data=tpr_df, x='group', y='tpr', hue='attribute', ax=ax3)
            ax3.set_title('True Positive Rates by Group')
            ax3.set_ylabel('True Positive Rate')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)

        # Plot 4: False Positive Rates
        ax4 = axes[1, 1]
        fpr_data = []
        for attr, results in fairness_results.items():
            for group, metrics in results['equalized_odds'].items():
                fpr_data.append({
                    'attribute': attr,
                    'group': group,
                    'fpr': metrics['false_positive_rate'],
                    'sample_size': metrics['sample_size']
                })

        if fpr_data:
            fpr_df = pd.DataFrame(fpr_data)
            sns.barplot(data=fpr_df, x='group', y='fpr', hue='attribute', ax=ax4)
            ax4.set_title('False Positive Rates by Group')
            ax4.set_ylabel('False Positive Rate')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fairness analysis plot saved to {save_path}")

        plt.show()


class FairnessAwareDebiasing:
    """Fairness-aware debiasing strategies"""

    def __init__(self):
        pass

    def reweight_samples(self, X: np.ndarray, y: np.ndarray,
                        protected_attr: np.ndarray,
                        target_fairness: float = 0.8) -> np.ndarray:
        """
        Reweight samples to achieve target fairness

        Args:
            X: Feature matrix
            y: Target labels
            protected_attr: Protected attribute
            target_fairness: Target disparate impact ratio

        Returns:
            Sample weights for training
        """
        unique_groups = np.unique(protected_attr)
        weights = np.ones(len(X))

        # Calculate current positive rates
        group_rates = {}
        for group in unique_groups:
            mask = protected_attr == group
            if np.sum(mask) > 0:
                group_rates[group] = np.mean(y[mask])

        # Find privileged and unprivileged groups
        rates = list(group_rates.values())
        privileged_rate = max(rates)
        unprivileged_rate = min(rates)

        if privileged_rate > 0:
            current_di = unprivileged_rate / privileged_rate

            if current_di < target_fairness:
                # Need to upweight unprivileged group
                adjustment_factor = target_fairness / current_di

                for i, group in enumerate(protected_attr):
                    if group_rates[group] == unprivileged_rate:
                        weights[i] *= adjustment_factor

        return weights

    def adversarial_debiasing_preprocessing(self, X: np.ndarray,
                                          protected_attr: np.ndarray,
                                          lambda_param: float = 0.1) -> np.ndarray:
        """
        Adversarial debiasing preprocessing

        Args:
            X: Feature matrix
            protected_attr: Protected attribute
            lambda_param: Regularization parameter

        Returns:
            Debiased feature matrix
        """
        from sklearn.preprocessing import StandardScaler

        # Encode protected attribute
        protected_encoded = np.where(protected_attr == np.unique(protected_attr)[0], 1, 0)

        # Simple adversarial debiasing (simplified version)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Remove correlation with protected attribute
        # This is a simplified approach - in practice, you'd use adversarial training
        correlation = np.corrcoef(X_scaled.T, protected_encoded.reshape(1, -1))[0, :-1]

        # Reduce features correlated with protected attribute
        debiased_X = X_scaled.copy()
        for i, corr in enumerate(correlation):
            if abs(corr) > 0.1:  # Threshold for correlation
                debiased_X[:, i] *= (1 - lambda_param * abs(corr))

        return debiased_X


def generate_fairness_report(fairness_results: Dict,
                           predictions: np.ndarray,
                           true_labels: np.ndarray) -> str:
    """
    Generate comprehensive fairness report

    Args:
        fairness_results: Results from fairness assessment
        predictions: Model predictions
        true_labels: True labels

    Returns:
        Formatted fairness report
    """
    report = []
    report.append("# Fairness Assessment Report")
    report.append("=" * 50)
    report.append("")

    # Overall model performance
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    report.append("## Overall Model Performance")
    report.append(f"- Accuracy: {accuracy:.3f}")
    report.append(f"- Precision: {precision:.3f}")
    report.append(f"- Recall: {recall:.3f}")
    report.append(f"- F1-Score: {f1:.3f}")
    report.append("")

    # Fairness analysis
    report.append("## Fairness Analysis")
    report.append("")

    for attr_name, results in fairness_results.items():
        report.append(f"### {attr_name.upper()}")
        report.append("")

        # Disparate impact
        di_ratios = results['disparate_impact']['disparate_impact_ratios']
        if di_ratios:
            report.append("**Disparate Impact Ratios:**")
            for comparison, ratio in di_ratios.items():
                status = "✅ FAIR" if ratio >= 0.8 else "⚠️ UNFAIR"
                report.append(f"- {comparison}: {ratio:.3f} {status}")
            report.append("")

        # Group rates
        group_rates = results['disparate_impact']['group_rates']
        report.append("**Group Positive Rates:**")
        for group, rate in group_rates.items():
            report.append(f"- {group}: {rate:.3f}")
        report.append("")

        # Equalized odds
        eo_results = results['equalized_odds']
        report.append("**Equalized Odds:**")
        for group, metrics in eo_results.items():
            report.append(f"- {group}:")
            report.append(f"  - True Positive Rate: {metrics['true_positive_rate']:.3f}")
            report.append(f"  - False Positive Rate: {metrics['false_positive_rate']:.3f}")
            report.append(f"  - Sample Size: {metrics['sample_size']}")
        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    unfair_attributes = []
    for attr_name, results in fairness_results.items():
        di_ratios = results['disparate_impact']['disparate_impact_ratios']
        if any(ratio < 0.8 for ratio in di_ratios.values()):
            unfair_attributes.append(attr_name)

    if unfair_attributes:
        report.append("⚠️ **Fairness Issues Detected**")
        report.append("")
        report.append("Consider implementing the following debiasing strategies:")
        report.append("- Sample reweighting")
        report.append("- Adversarial debiasing")
        report.append("- Fair representation learning")
        report.append("- Post-processing calibration")
        report.append("")
        report.append("Affected attributes: " + ", ".join(unfair_attributes))
    else:
        report.append("✅ **No Major Fairness Issues Detected**")
        report.append("")
        report.append("The model appears to be fair across protected attributes.")

    return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    n_samples = 1000

    # Simulated predictions and labels
    predictions = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    true_labels = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

    # Simulated protected attributes
    protected_attrs = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples),
        'age_group': np.random.choice(['18-30', '31-50', '51-70', '71+'], n_samples)
    }

    # Initialize fairness assessor
    assessor = FairnessAssessor()

    # Assess fairness
    fairness_results = assessor.assess_fairness(
        predictions, true_labels, protected_attrs
    )

    # Generate report
    report = generate_fairness_report(fairness_results, predictions, true_labels)
    print(report)

    # Plot fairness analysis
    assessor.plot_fairness_analysis(fairness_results, save_path='../reports/fairness_analysis.png')
