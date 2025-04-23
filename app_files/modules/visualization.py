"""Visualization service for topic modeling results."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

from .core_types import (
    DataFrameType, ModelProtocol, StatusProtocol,
    TopicVisualizationData
)
from . import config

logger = logging.getLogger(__name__)

class VisualizationService:
    """Centralized service for generating and saving visualizations."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualization service."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message."""
        try:
            fig = go.Figure()
            fig.add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating empty plot: {e}")
            # Create absolute minimum plot as fallback
            return go.Figure()
    
    def get_topic_visualization(self, model: ModelProtocol) -> go.Figure:
        """Generate topic distribution visualization."""
        try:
            topic_info = model.get_topic_info()
            valid_topics = topic_info[topic_info['Topic'] != -1]
            
            if valid_topics.empty:
                return self._create_empty_plot("No topics to visualize")
            
            sizes = valid_topics['Count'].tolist()
            labels = valid_topics['Name'].tolist()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=sizes,
                    text=valid_topics['Top_Words'].tolist(),
                    textposition='auto',
                    hovertemplate="<b>%{x}</b><br>" +
                                "Documents: %{y}<br>" +
                                "Top Words: %{text}<br>" +
                                "<extra></extra>"
                )
            ])
            
            fig.update_layout(
                title="Topic Distribution",
                xaxis_title="Topics",
                yaxis_title="Number of Documents",
                showlegend=False,
                xaxis={'tickangle': 45},
                margin=dict(b=100),
                width=config.VISUALIZATION_CONFIG['PLOT_DIMENSIONS']['width'],
                height=config.VISUALIZATION_CONFIG['PLOT_DIMENSIONS']['height']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating topic visualization: {e}")
            return self._create_empty_plot(str(e))
    
    def get_topic_hierarchy(self, model: ModelProtocol) -> go.Figure:
        """Generate topic hierarchy visualization."""
        try:
            topic_info = model.get_topic_info()
            valid_topics = topic_info[topic_info['Topic'] != -1]
            
            if valid_topics.empty:
                return self._create_empty_plot("No topics to visualize")
            
            return model.model.visualize_hierarchy(width=800, height=600)
            
        except Exception as e:
            logger.error(f"Error generating hierarchy: {e}")
            return self._create_empty_plot(str(e))
    
    def get_topic_wordcloud(self, model: ModelProtocol) -> plt.Figure:
        """Generate word cloud visualization."""
        try:
            topic_info = model.get_topic_info()
            valid_topics = topic_info[topic_info['Topic'] != -1]
            
            if valid_topics.empty:
                return self._create_empty_plot("No topics to visualize")
            
            # Collect words and weights from all topics
            word_freqs = {}
            for topic_id in valid_topics['Topic']:
                for word, weight in model.get_topic(topic_id):
                    if isinstance(word, str) and word.strip():
                        word_freqs[word.strip()] = word_freqs.get(word.strip(), 0) + abs(float(weight))
            
            if not word_freqs:
                return self._create_empty_plot("No words to visualize")
            
            # Create word cloud
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freqs)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating wordcloud: {e}")
            return self._create_empty_plot(str(e))
    
    async def save_visualizations(
        self,
        model: ModelProtocol,
        status_manager: Optional[StatusProtocol] = None
    ) -> Dict[str, Path]:
        """Generate and save all visualizations."""
        results = {}
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define visualization tasks
            visualizations = {
                'distribution': (self.get_topic_visualization, 'html'),
                'hierarchy': (self.get_topic_hierarchy, 'html'),
                'wordcloud': (self.get_topic_wordcloud, 'png')
            }
            
            # Generate and save each visualization
            total = len(visualizations)
            for i, (viz_type, (viz_func, suffix)) in enumerate(visualizations.items(), 1):
                if status_manager:
                    progress = (i / total) * 100
                    status_manager.update_status("Visualization", progress, f"Generating {viz_type}...")
                
                try:
                    # Generate visualization
                    viz = viz_func(model)
                    path = self.output_dir / f"topic_{viz_type}.{suffix}"
                    
                    # Save visualization
                    if suffix == 'html':
                        viz.write_html(str(path))
                    else:
                        viz.savefig(path, bbox_inches='tight', dpi=300)
                        plt.close()
                    
                    results[viz_type] = path
                    logger.info(f"Saved {viz_type} visualization to {path}")
                    
                except Exception as e:
                    logger.error(f"Error saving {viz_type}: {e}")
            
            if status_manager:
                status_manager.update_status("Visualization", 100, "Visualizations complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")
            if status_manager:
                status_manager.set_error(f"Visualization error: {e}")
            return results