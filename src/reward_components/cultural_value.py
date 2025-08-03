import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class AwardTier(Enum):
    TIER_1 = 1.0  # Oscar, Palme d'Or, Golden Lion
    TIER_2 = 0.8  # BAFTA, Golden Globe, Berlinale
    TIER_3 = 0.6  # National awards, major festival selections
    TIER_4 = 0.4  # Regional awards, smaller festivals

@dataclass
class CulturalMetrics:
    """Metrics for computing cultural value of a film"""
    awards: List[tuple]  # (award_name, year, tier_weight)
    festival_selections: List[str]  # Festival names
    critical_score: float  # 0-100 from aggregated reviews
    academic_citations: int  # Scholarly references
    cultural_themes: List[str]  # ["historical", "social_issues", "art", etc.]
    origin_country: str
    language: str
    year: int
    director_prestige: float  # 0-1 based on career achievements
    educational_alignment: float  # 0-1 how well it fits educational mission

class CulturalValueCalculator:
    def __init__(self, 
                 local_country: str = "your_country",
                 heritage_languages: List[str] = None,
                 current_year: int = 2024):
        self.local_country = local_country
        self.heritage_languages = heritage_languages or []
        self.current_year = current_year
        
        # Cultural theme weights for public TV mission
        self.theme_weights = {
            "historical": 0.9,
            "social_issues": 0.85,
            "art_culture": 0.8,
            "science_nature": 0.75,
            "human_rights": 0.9,
            "education": 0.85,
            "diversity": 0.8,
            "local_heritage": 1.0
        }
    
    def compute_cultural_value(self, metrics: CulturalMetrics) -> float:
        """
        Compute cultural value score (0-1) for a film
        """
        # 1. Awards and Recognition Score (0-0.3)
        awards_score = self._compute_awards_score(metrics.awards)
        
        # 2. Critical and Academic Recognition (0-0.2)
        recognition_score = self._compute_recognition_score(
            metrics.critical_score, metrics.academic_citations
        )
        
        # 3. Cultural Relevance and Themes (0-0.25)
        theme_score = self._compute_theme_score(metrics.cultural_themes)
        
        # 4. Origin and Language Diversity (0-0.15)
        diversity_score = self._compute_diversity_score(
            metrics.origin_country, metrics.language
        )
        
        # 5. Educational and Public Service Alignment (0-0.1)
        education_score = metrics.educational_alignment * 0.1
        
        # 6. Director Prestige and Artistic Merit (0-0.1)
        artistic_score = metrics.director_prestige * 0.1
        
        total_score = (awards_score + recognition_score + theme_score + 
                      diversity_score + education_score + artistic_score)
        
        return min(1.0, total_score)  # Cap at 1.0
    
    def _compute_awards_score(self, awards: List[tuple]) -> float:
        """Compute score based on awards and recognitions"""
        if not awards:
            return 0.0
            
        score = 0.0
        for award_name, year, tier_weight in awards:
            # Recent awards get higher weight
            recency_factor = max(0.5, 1.0 - (self.current_year - year) / 20)
            score += tier_weight * recency_factor * 0.1
            
        return min(0.3, score)  # Cap at 0.3
    
    def _compute_recognition_score(self, critical_score: float, citations: int) -> float:
        """Compute score based on critical and academic recognition"""
        # Normalize critical score (assuming 0-100 scale)
        critical_component = (critical_score / 100) * 0.15
        
        # Academic citations (with diminishing returns)
        citation_component = min(0.05, np.log1p(citations) / 100)
        
        return critical_component + citation_component
    
    def _compute_theme_score(self, themes: List[str]) -> float:
        """Compute score based on cultural themes alignment"""
        if not themes:
            return 0.0
            
        theme_scores = [self.theme_weights.get(theme, 0.3) for theme in themes]
        # Average of top themes to avoid over-rewarding many themes
        top_themes = sorted(theme_scores, reverse=True)[:3]
        
        return (sum(top_themes) / len(top_themes)) * 0.25
    
    def _compute_diversity_score(self, origin_country: str, language: str) -> float:
        """Compute score based on cultural diversity"""
        diversity_bonus = 0.0
        
        # Non-local content gets diversity bonus
        if origin_country != self.local_country:
            diversity_bonus += 0.1
            
        # Heritage languages get special consideration
        if language in self.heritage_languages:
            diversity_bonus += 0.05
        # Other non-local languages get smaller bonus
        elif language != "your_primary_language":
            diversity_bonus += 0.03
            
        return min(0.15, diversity_bonus)

# Example usage
def example_cultural_value_computation():
    """Example of how to compute cultural value for different films"""
    
    calculator = CulturalValueCalculator(
        local_country="France",
        heritage_languages=["French", "Arabic", "Portuguese"]
    )
    
    # Example 1: Award-winning social drama
    social_drama = CulturalMetrics(
        awards=[("César", 2022, AwardTier.TIER_3.value), 
                ("Cannes Selection", 2022, AwardTier.TIER_2.value)],
        festival_selections=["Cannes", "Toronto"],
        critical_score=82.0,
        academic_citations=5,
        cultural_themes=["social_issues", "diversity", "human_rights"],
        origin_country="France",
        language="French",
        year=2022,
        director_prestige=0.7,
        educational_alignment=0.8
    )
    
    # Example 2: Foreign arthouse film
    arthouse_film = CulturalMetrics(
        awards=[("Berlin Golden Bear", 2023, AwardTier.TIER_1.value)],
        festival_selections=["Berlinale", "New York"],
        critical_score=88.0,
        academic_citations=12,
        cultural_themes=["art_culture", "historical"],
        origin_country="Iran",
        language="Persian",
        year=2023,
        director_prestige=0.9,
        educational_alignment=0.9
    )
    
    # Example 3: Commercial blockbuster
    blockbuster = CulturalMetrics(
        awards=[],
        festival_selections=[],
        critical_score=65.0,
        academic_citations=0,
        cultural_themes=[],
        origin_country="USA",
        language="English",
        year=2023,
        director_prestige=0.4,
        educational_alignment=0.2
    )
    
    print("Cultural Value Scores:")
    print(f"Social Drama: {calculator.compute_cultural_value(social_drama):.3f}")
    print(f"Arthouse Film: {calculator.compute_cultural_value(arthouse_film):.3f}")
    print(f"Blockbuster: {calculator.compute_cultural_value(blockbuster):.3f}")

if __name__ == "__main__":
    example_cultural_value_computation()