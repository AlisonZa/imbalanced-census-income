import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union


class InteractiveEDAPlotter:
    def __init__(self, env_config, feature_def):
        """
        Initialize the EDA plotter with environment configuration and feature definitions.
        
        Args:
            env_config (EnviromentConfiguration): Configuration for folder paths
            feature_def (FeatureDefinition): Feature definitions for the dataset
        """
        self.env_config = env_config
        self.feature_def = feature_def
        
        # Ensure plot directories exist
        os.makedirs(self.env_config.univariate_plots_folder, exist_ok=True)
        os.makedirs(self.env_config.bivariate_plots_folder, exist_ok=True)
    
    def _save_plot(self, filename: str, subfolder: str = ''):
        """
        Save the current plot to the specified folder.
        
        Args:
            filename (str): Name of the file to save
            subfolder (str, optional): Subfolder within plots directory
        """
        full_path = os.path.join(
            self.env_config.plots_folder, 
            subfolder, 
            filename
        )
        plt.tight_layout()
        plt.savefig(full_path)
        plt.close()
    
    def univariate_quantitative_plots(self, feature: str, is_ordinal: bool = False):
        """
        Create plots for a single quantitative feature, with special handling for ordinal features.
        
        Args:
            feature (str): Name of the quantitative feature
            is_ordinal (bool): Whether the feature is an ordinal numerical feature
        """
        data = self.feature_def.data_frame[feature]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(f'Univariate Analysis: {feature}')
        
        # Rug plot
        sns.rugplot(data, ax=axs[0, 0])
        axs[0, 0].set_title('Rug Plot')
        
        # Histogram with special handling for ordinal features
        if is_ordinal:
            # For ordinal features, use discrete bins
            unique_values = sorted(data.unique())
            sns.histplot(data, discrete=True, ax=axs[0, 1])
        else:
            # For continuous features, use default histogram
            sns.histplot(data, kde=False, ax=axs[0, 1])
        axs[0, 1].set_title('Histogram')
        
        # Density curve
        # For ordinal features, use fewer points in KDE
        if is_ordinal:
            sns.kdeplot(data, bw_method='silverman', cut=0, ax=axs[1, 0])
        else:
            sns.kdeplot(data, ax=axs[1, 0])
        axs[1, 0].set_title('Density Curve')
        
        # Box plot
        sns.boxplot(x=data, ax=axs[1, 1])
        axs[1, 1].set_title('Box Plot')
        
        # Violin plot
        sns.violinplot(x=data, ax=axs[2, 0])
        axs[2, 0].set_title('Violin Plot')
        
        # Additional summary statistics plot
        summary_stats = {
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Unique Values': len(data.unique()) if is_ordinal else 'N/A'
        }
        
        summary_text = '\n'.join([f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' 
                                for k, v in summary_stats.items()])
        
        axs[2, 1].text(0.5, 0.5, 
                    f'Summary Statistics:\n{summary_text}', 
                    horizontalalignment='center', 
                    verticalalignment='center')
        axs[2, 1].axis('off')
        axs[2, 1].set_title('Summary')
        
        self._save_plot(f'{feature}_univariate_analysis.png', 'univariate')

    
    def univariate_qualitative_plots(self, feature: str):
        """
        Create plots for a single qualitative feature.
        
        Args:
            feature (str): Name of the qualitative feature
        """
        data = self.feature_def.data_frame[feature]
        value_counts = data.value_counts()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Univariate Analysis: {feature}')
        
        # Bar plot
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axs[0, 0])
        axs[0, 0].set_title('Bar Plot')
        axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)
        
        # Dot plot
        sns.stripplot(x=data, ax=axs[0, 1])
        axs[0, 1].set_title('Dot Plot')
        axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)
        
        # Line plot
        value_counts.plot(kind='line', ax=axs[1, 0])
        axs[1, 0].set_title('Line Plot')
        
        # Pie chart
        axs[1, 1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        axs[1, 1].set_title('Pie Chart')
        
        self._save_plot(f'{feature}_univariate_analysis.png', 'univariate')
    
    def bivariate_quantitative_plots(self, feature1: str, feature2: str):
        """
        Create plots for two quantitative features.
        
        Args:
            feature1 (str): First quantitative feature
            feature2 (str): Second quantitative feature
        """
        data1 = self.feature_def.data_frame[feature1]
        data2 = self.feature_def.data_frame[feature2]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Bivariate Analysis: {feature1} vs {feature2}')
        
        # Scatterplot
        sns.scatterplot(x=data1, y=data2, ax=axs[0, 0])
        axs[0, 0].set_title('Scatter Plot')
        
        # Smooth curve
        sns.regplot(x=data1, y=data2, ax=axs[0, 1])
        axs[0, 1].set_title('Smooth Curve')
        
        # Contour plot
        sns.kdeplot(x=data1, y=data2, fill=True, ax=axs[1, 0])
        axs[1, 0].set_title('Contour Plot')
        
        # Heatmap
        sns.heatmap(self.feature_def.data_frame[[feature1, feature2]].corr(), 
                    annot=True, cmap='coolwarm', ax=axs[1, 1])
        axs[1, 1].set_title('Correlation Heatmap')
        
        self._save_plot(f'{feature1}_vs_{feature2}_bivariate_analysis.png', 'bivariate')
    
    def bivariate_qualitative_plots(self, feature1: str, feature2: str):
        """
        Create plots for two qualitative features.
        
        Args:
            feature1 (str): First qualitative feature
            feature2 (str): Second qualitative feature
        """
        data1 = self.feature_def.data_frame[feature1]
        data2 = self.feature_def.data_frame[feature2]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Bivariate Analysis: {feature1} vs {feature2}')
        
        # Side-by-side bar plots
        grouped_data = self.feature_def.data_frame.groupby([feature1, feature2]).size().unstack()
        grouped_data.plot(kind='bar', stacked=False, ax=axs[0, 0])
        axs[0, 0].set_title('Side-by-Side Bar Plots')
        axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)
        
        # Stacked bar plot as an alternative to mosaic plot
        grouped_data.plot(kind='bar', stacked=True, ax=axs[0, 1])
        axs[0, 1].set_title('Stacked Bar Plot')
        axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)
        
        # Overlaid lines (frequency of combination)
        combination_counts = self.feature_def.data_frame.groupby([feature1, feature2]).size()
        combination_counts.unstack().plot(kind='line', ax=axs[1, 0])
        axs[1, 0].set_title('Overlaid Lines')
        axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)
        
        # Contingency table heatmap
        contingency_table = pd.crosstab(data1, data2, normalize='all')
        sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', ax=axs[1, 1])
        axs[1, 1].set_title('Contingency Table')
        
        self._save_plot(f'{feature1}_vs_{feature2}_bivariate_analysis.png', 'bivariate')
        
    def bivariate_mixed_plots(self, quantitative_feature: str, categorical_feature: str):
        """
        Create plots for mixed feature types (quantitative vs. categorical).
        """
        # Ensure data types are correct
        df = self.feature_def.data_frame.copy()
        df[categorical_feature] = df[categorical_feature].astype(str)
        
        quant_data = df[quantitative_feature]
        cat_data = df[categorical_feature]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Mixed Feature Analysis: {quantitative_feature} vs {categorical_feature}')
        
        # Overlaid density curves (if you want to show the distribution)
        for category in cat_data.unique():
            subset = quant_data[cat_data == category]
            sns.kdeplot(subset, label=str(category), ax=axs[0, 0])
        axs[0, 0].set_title('Overlaid Density Curves')
        axs[0, 0].legend()
        
        # Side-by-side box plots (visualize quantitative feature distribution for each category)
        sns.boxplot(x=cat_data, y=quant_data, ax=axs[0, 1])
        axs[0, 1].set_title('Side-by-Side Box Plots')
        axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)
        
        # Violin plot (shows distribution of the quantitative feature)
        sns.violinplot(x=cat_data, y=quant_data, ax=axs[1, 0])
        axs[1, 0].set_title('Violin Plot')
        axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)
        
        # Dummy plot for summary
        summary_text = f'Summary:\n'
        for category in cat_data.unique():
            subset = quant_data[cat_data == category]
            summary_text += f'{category}:\n'
            summary_text += f'  Mean: {subset.mean():.2f}\n'
            summary_text += f'  Std: {subset.std():.2f}\n'
        
        axs[1, 1].text(0.5, 0.5, summary_text, 
                    horizontalalignment='center', 
                    verticalalignment='center')
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Category Summaries')
        
        self._save_plot(f'{quantitative_feature}_vs_{categorical_feature}_mixed_analysis.png', 'bivariate')


    def perform_eda(self):
        """
        Perform Exploratory Data Analysis based on feature types.
        """
        # Univariate Analysis
        if self.feature_def.numeric:
            for feature in self.feature_def.numeric:
                self.univariate_quantitative_plots(feature, is_ordinal=False)
        
        categorical_features = (self.feature_def.categorical_nominals or []) + \
                            (self.feature_def.categorical_binary or []) + \
                            (self.feature_def.categorical_ordinals or [])
        
        if categorical_features:
            for feature in categorical_features:
                self.univariate_qualitative_plots(feature)
        
        # Bivariate Analysis
        # Quantitative vs Quantitative
        if self.feature_def.numeric and len(self.feature_def.numeric) > 1:
            for i in range(len(self.feature_def.numeric)):
                for j in range(i+1, len(self.feature_def.numeric)):
                    self.bivariate_quantitative_plots(
                        self.feature_def.numeric[i], 
                        self.feature_def.numeric[j]
                    )
        
        # Categorical vs Categorical
        if len(categorical_features) > 1:
            for i in range(len(categorical_features)):
                for j in range(i+1, len(categorical_features)):
                    self.bivariate_qualitative_plots(
                        categorical_features[i], 
                        categorical_features[j]
                    )
        
        # Mixed Features
        if self.feature_def.numeric and categorical_features:
            for quant_feature in self.feature_def.numeric:
                for cat_feature in categorical_features:
                    self.bivariate_mixed_plots(quant_feature, cat_feature)
