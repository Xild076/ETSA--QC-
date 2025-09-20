"""
Configurable parameters for the ETSA pipeline.
This allows tuning without code changes for different domains and datasets.
"""
import os
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict, field

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    # Ensemble weights for different analyzers
    vader_weight: float = 1.0
    textblob_weight: float = 1.0
    flair_weight: float = 1.5
    nlptown_weight: float = 1.5
    bertweet_weight: float = 1.5
    finbert_weight: float = 1.5
    
    # Neutral threshold - scores below this are considered neutral
    neutral_threshold: float = 0.15
    
    # Cache settings
    sentiment_cache_size: int = 1000
    
    # Parallel processing
    max_workers: int = 3
    timeout_seconds: int = 10

@dataclass
class EntityConfig:
    """Configuration for entity processing."""
    # Similarity threshold for semantic merging
    similarity_threshold: float = 0.75
    
    # Distance threshold for pronoun resolution (characters)
    pronoun_resolution_distance: int = 300
    
    # Cache settings
    category_mapping_enabled: bool = True
    
    # Aspect detection keywords
    aspect_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'price': ['overpriced', 'expensive', 'cheap', 'costly', 'affordable', 'pricey', 'price', 'cost', 'pricing', 'rates'],
        'quality': ['excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad'],
        'service': ['service', 'server', 'staff', 'waiter', 'waitress', 'serving', 'served'],
        'food': ['food', 'dish', 'meal', 'cuisine', 'recipe', 'cooking'],
        'ambience': ['atmosphere', 'ambience', 'ambiance', 'environment', 'setting', 'mood']
    })

@dataclass
class GraphConfig:
    """Configuration for graph processing."""
    # Batch processing settings
    enable_batch_processing: bool = True
    batch_size: int = 50
    
    # Memory optimization
    sentiment_cache_size: int = 500
    enable_clause_caching: bool = True

@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Component configurations
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    entity: EntityConfig = field(default_factory=EntityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    # Performance settings
    enable_parallel_processing: bool = True
    memory_cleanup_threshold: int = 1000
    
    # Domain-specific settings
    domain: str = "restaurant"  # restaurant, tech, general
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            # Return default config if file doesn't exist
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def get_domain_optimized_config(self) -> 'PipelineConfig':
        """Get domain-optimized configuration."""
        if self.domain == "restaurant":
            # Optimize for restaurant reviews
            self.sentiment.neutral_threshold = 0.15
            self.entity.similarity_threshold = 0.75
            
        elif self.domain == "tech":
            # Optimize for tech product reviews
            self.sentiment.neutral_threshold = 0.10  # More sensitive
            self.entity.similarity_threshold = 0.80  # Higher precision
            
        elif self.domain == "general":
            # Conservative settings for general text
            self.sentiment.neutral_threshold = 0.20
            self.entity.similarity_threshold = 0.70
        
        return self

# Global configuration instance
DEFAULT_CONFIG = PipelineConfig()

def get_config(config_path: str = None) -> PipelineConfig:
    """Get configuration from file or return default."""
    if config_path:
        return PipelineConfig.load_from_file(config_path)
    return DEFAULT_CONFIG

def update_config(updates: Dict[str, Any], config_path: str = None) -> PipelineConfig:
    """Update configuration with new values."""
    config = get_config(config_path)
    
    # Apply updates recursively
    for key, value in updates.items():
        if hasattr(config, key):
            if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                # Update nested configuration
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested_config, nested_key):
                        setattr(nested_config, nested_key, nested_value)
            else:
                setattr(config, key, value)
    
    return config

# Domain-specific presets
RESTAURANT_CONFIG = PipelineConfig(
    domain="restaurant",
    sentiment=SentimentConfig(
        neutral_threshold=0.15,
        vader_weight=1.0,
        textblob_weight=1.0,
        flair_weight=1.5
    ),
    entity=EntityConfig(
        similarity_threshold=0.75,
        aspect_keywords={
            'price': ['overpriced', 'expensive', 'cheap', 'costly', 'affordable', 'pricey', 'price', 'cost'],
            'quality': ['excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad', 'delicious', 'tasty'],
            'service': ['service', 'server', 'staff', 'waiter', 'waitress', 'friendly', 'rude', 'slow', 'fast'],
            'food': ['food', 'dish', 'meal', 'cuisine', 'flavor', 'taste', 'fresh', 'stale', 'hot', 'cold'],
            'ambience': ['atmosphere', 'ambience', 'environment', 'noisy', 'quiet', 'crowded', 'cozy']
        }
    )
)

TECH_CONFIG = PipelineConfig(
    domain="tech",
    sentiment=SentimentConfig(
        neutral_threshold=0.10,
        vader_weight=1.2,
        flair_weight=1.8
    ),
    entity=EntityConfig(
        similarity_threshold=0.80,
        aspect_keywords={
            'performance': ['fast', 'slow', 'quick', 'laggy', 'responsive', 'smooth', 'choppy'],
            'quality': ['excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'solid', 'flimsy'],
            'design': ['beautiful', 'ugly', 'sleek', 'bulky', 'elegant', 'cheap-looking', 'premium'],
            'features': ['useful', 'useless', 'innovative', 'outdated', 'convenient', 'confusing'],
            'value': ['overpriced', 'affordable', 'worth it', 'expensive', 'cheap', 'good value']
        }
    )
)
