"""
Discovery Engine — Smart Recommendations & Method Navigation
==============================================================
Provides:
- Recommendation engine (related methods, prerequisites)
- Method comparison utilities
- Learning path suggestions
- Advanced filtering & discovery
"""

from .equations import METHOD_METADATA
from .catalog import CATEGORIES


# ════════════════════════════════════════════════════════════
# Recommendation Engine
# ════════════════════════════════════════════════════════════

def get_related_methods(method_filename, relation_type="all"):
    """
    Get related methods for a given method.
    
    Parameters:
        method_filename: e.g., "bisection_method.py"
        relation_type: "prerequisites", "related", "prepares_for", "alternatives", "all"
    
    Returns:
        List of (filename, title, reason) tuples
    """
    results = []
    basename = method_filename.replace('.py', '')
    
    if basename not in METHOD_METADATA:
        return []
    
    metadata = METHOD_METADATA[basename]
    
    # Prerequisites (methods to learn first)
    if relation_type in ["prerequisites", "all"]:
        prereqs = metadata.get("prerequisites", [])
        for method_title in prereqs:
            method_key = method_title.lower().replace(" ", "_")
            # Find matching file
            for cat in CATEGORIES:
                for fname, title, desc in cat["files"]:
                    if title.lower().replace(" ", "_") == method_key or title == method_title:
                        results.append((fname, title, "Prerequisite - learn this first"))
    
    # Related methods (same category or complementary)
    if relation_type in ["related", "all"]:
        related = metadata.get("related_methods", [])
        for method_title in related:
            method_key = method_title.lower().replace(" ", "_")
            for cat in CATEGORIES:
                for fname, title, desc in cat["files"]:
                    if title.lower().replace(" ", "_") == method_key or title == method_title:
                        results.append((fname, title, "Related method - similar approach"))
    
    # Prepares for (dependent methods)
    if relation_type in ["prepares_for", "all"]:
        for key, meta in METHOD_METADATA.items():
            for prereq in meta.get("prerequisites", []):
                if prereq.lower() in basename.lower() or basename.lower() in prereq.lower():
                    # Find this method's file
                    for cat in CATEGORIES:
                        for fname, title, desc in cat["files"]:
                            if fname.replace('.py', '') == key:
                                results.append((fname, title, "Next step - builds on this"))
    
    return results


def get_recommendations(method_filename):
    """
    Get personalized recommendations for learning next.
    
    Returns:
        {
            "prerequisites": [...],
            "related": [...],
            "next_steps": [...],
            "alternatives": [...]
        }
    """
    results = {
        "prerequisites": get_related_methods(method_filename, "prerequisites"),
        "related": get_related_methods(method_filename, "related"),
        "next_steps": get_related_methods(method_filename, "prepares_for"),
        "alternatives": get_similar_methods(method_filename),
    }
    return results


def get_similar_methods(method_filename, limit=5):
    """
    Get methods that solve similar problems.
    
    Returns:
        List of (filename, title, reason) tuples
    """
    basename = method_filename.replace('.py', '')
    
    if basename not in METHOD_METADATA:
        return []
    
    target_category = None
    
    # Find category for this method
    for cat in CATEGORIES:
        for fname, title, desc in cat["files"]:
            if fname == method_filename:
                target_category = cat["name"]
                break
    
    # Find other methods in same category
    similar = []
    if target_category:
        for cat in CATEGORIES:
            if cat["name"] == target_category:
                for fname, title, desc in cat["files"]:
                    if fname != method_filename:
                        similar.append((fname, title, f"Alternative in {target_category}"))
    
    return similar[:limit]


def get_complexity_comparison(method1, method2):
    """
    Compare complexity of two methods.
    
    Returns:
        Dict with convergence rate, time complexity, space complexity
    """
    base1 = method1.replace('.py', '')
    base2 = method2.replace('.py', '')
    
    meta1 = METHOD_METADATA.get(base1, {})
    meta2 = METHOD_METADATA.get(base2, {})
    
    return {
        "method1": {
            "name": base1.replace("_", " ").title(),
            "complexity": meta1.get("complexity", "Unknown"),
            "convergence": _infer_convergence(base1),
        },
        "method2": {
            "name": base2.replace("_", " ").title(),
            "complexity": meta2.get("complexity", "Unknown"),
            "convergence": _infer_convergence(base2),
        }
    }


def _infer_convergence(method_name):
    """Infer convergence type from method name."""
    method_lower = method_name.lower()
    
    if "newton" in method_lower or "raphson" in method_lower:
        return "Quadratic"
    elif "secant" in method_lower:
        return "Super-linear"
    elif "bisection" in method_lower or "fixed_point" in method_lower:
        return "Linear"
    elif "rk4" in method_lower or "runge_kutta" in method_lower:
        return "4th order"
    elif "euler" in method_lower:
        return "1st order"
    elif "gradient" in method_lower:
        return "Linear (depends on problem)"
    elif "monte_carlo" in method_lower:
        return "O(1/√N)"
    else:
        return "Method-dependent"


# ════════════════════════════════════════════════════════════
# Learning Advisor (determines next steps)
# ════════════════════════════════════════════════════════════

class LearningAdvisor:
    """Suggests optimal learning sequence based on prerequisites."""
    
    def __init__(self):
        # Build prerequisite graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build method dependency graph."""
        graph = {}
        for cat in CATEGORIES:
            for fname, title, desc in cat["files"]:
                basename = fname.replace('.py', '')
                meta = METHOD_METADATA.get(basename, {})
                
                graph[fname] = {
                    "title": title,
                    "prerequisites": meta.get("prerequisites", []),
                    "category": cat["name"],
                    "difficulty": meta.get("difficulty", "intermediate"),
                }
        return graph
    
    def get_next_recommendations(self, current_method, learned_methods=None):
        """
        Suggest 3-5 next methods to learn.
        
        Parameters:
            current_method: Currently viewing this method
            learned_methods: List of already-learned methods
        
        Returns:
            List of (filename, title, reason, readiness_score) sorted by readiness
        """
        if learned_methods is None:
            learned_methods = []
        
        recommendations = []
        current_category = None
        
        # Find current method's category
        for fname, info in self.graph.items():
            if fname == current_method:
                current_category = info["category"]
                break
        
        # Score each method for recommendation
        for fname, info in self.graph.items():
            if fname in learned_methods or fname == current_method:
                continue
            
            # Check if prerequisites are met
            prerequisites_met = True
            missing_prereqs = []
            
            for prereq in info["prerequisites"]:
                prereq_met = False
                for learned in learned_methods:
                    learned_title = self.graph[learned].get("title", "")
                    if prereq.lower() in learned_title.lower() or learned_title.lower() in prereq.lower():
                        prereq_met = True
                        break
                
                if not prereq_met:
                    prerequisites_met = False
                    missing_prereqs.append(prereq)
            
            # Calculate readiness score
            score = 0
            reason = ""
            
            if prerequisites_met:
                score += 100  # All prerequisites met
                reason = "✓ Ready to learn"
            else:
                score += 50  # Some prerequisites missing
                reason = f"Learn prerequisites first: {', '.join(missing_prereqs[:2])}"
            
            # Bonus for same category
            if info["category"] == current_category:
                score += 20
                reason += " (same category)"
            
            # Bonus for intermediate difficulty if just started
            if info["difficulty"] == "intermediate":
                score += 10
            
            recommendations.append((fname, info["title"], reason, score))
        
        # Sort by readiness score (descending)
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        return recommendations[:5]  # Return top 5
    
    def get_difficulty_distribution(self):
        """Get count of methods by difficulty level."""
        distribution = {"beginner": 0, "intermediate": 0, "advanced": 0}
        for info in self.graph.values():
            diff = info["difficulty"].lower()
            if diff in distribution:
                distribution[diff] += 1
        return distribution
    
    def is_ready_for(self, method_filename, learned_methods):
        """Check if user is ready for this method."""
        if method_filename not in self.graph:
            return True, []
        
        info = self.graph[method_filename]
        missing = []
        
        for prereq in info["prerequisites"]:
            found = False
            for learned in learned_methods:
                learned_title = self.graph[learned].get("title", "")
                if prereq.lower() in learned_title.lower():
                    found = True
                    break
            if not found:
                missing.append(prereq)
        
        return len(missing) == 0, missing


# ════════════════════════════════════════════════════════════
# Search & Discovery
# ════════════════════════════════════════════════════════════

def search_methods(query, filters=None):
    """
    Advanced search with optional filters.
    
    Parameters:
        query: Search string ("root finding", "fast methods", etc.)
        filters: {
            "category": [...],
            "difficulty": ["beginner", "intermediate", "advanced"],
            "min_complexity": "Linear" | "Quadratic" | etc.,
            "application": ["solve_equations", "optimize", etc.]
        }
    
    Returns:
        List of (filename, title, description, match_score) tuples
    """
    if filters is None:
        filters = {}
    
    results = []
    query_lower = query.lower()
    
    for cat in CATEGORIES:
        # Check category filters
        if filters.get("category"):
            if cat["name"] not in filters["category"]:
                continue
        
        for fname, title, desc in cat["files"]:
            basename = fname.replace('.py', '')
            meta = METHOD_METADATA.get(basename, {})
            
            # Check difficulty filter
            if filters.get("difficulty"):
                if meta.get("difficulty", "intermediate") not in filters["difficulty"]:
                    continue
            
            # Calculate match score
            score = 0
            searchable = f"{title} {desc} {fname}".lower()
            
            # Exact match in title
            if query_lower in title.lower():
                score += 100
            
            # Match in description/metadata
            if query_lower in desc.lower():
                score += 50
            
            # Match in applications
            apps = meta.get("applications", [])
            for app in apps:
                if query_lower in app.lower():
                    score += 30
            
            # Match in category
            if query_lower in cat["name"].lower():
                score += 20
            
            if score > 0:
                results.append((fname, title, desc, score))
    
    # Sort by match score
    results.sort(key=lambda x: x[3], reverse=True)
    
    return results


def get_method_info(method_filename):
    """Get detailed info for a method including all relationships."""
    basename = method_filename.replace('.py', '')
    meta = METHOD_METADATA.get(basename, {})
    
    # Find in catalog
    category = None
    description = None
    for cat in CATEGORIES:
        for fname, title, desc in cat["files"]:
            if fname == method_filename:
                category = cat["name"]
                description = desc
                break
    
    return {
        "filename": method_filename,
        "title": basename.replace("_", " ").title(),
        "category": category,
        "description": description,
        "difficulty": meta.get("difficulty", "intermediate"),
        "complexity": meta.get("complexity", "Unknown"),
        "prerequisites": meta.get("prerequisites", []),
        "applications": meta.get("applications", []),
        "related_methods": meta.get("related_methods", []),
        "convergence": _infer_convergence(basename),
    }


def get_category_methods(category_name):
    """Get all methods in a category."""
    methods = []
    for cat in CATEGORIES:
        if cat["name"] == category_name:
            for fname, title, desc in cat["files"]:
                methods.append({
                    "filename": fname,
                    "title": title,
                    "description": desc,
                })
    return methods
