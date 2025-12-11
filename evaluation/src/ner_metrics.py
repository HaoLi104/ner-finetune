#!/usr/bin/env python3
"""
Named Entity Recognition (NER) evaluation metrics module.
Handles evaluation of NER predictions in JSON format.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
import json
try:
    from .similarity_utils import is_texts_similar
except ImportError:
    from similarity_utils import is_texts_similar


@dataclass
class Entity:
    """Represents a named entity."""
    start: int
    end: int
    type: str
    text: str
    
    def __hash__(self):
        # 如果text为空，则不包含text在哈希中（与__eq__保持一致）
        if not self.text:
            return hash((self.start, self.end, self.type))
        return hash((self.start, self.end, self.type, self.text))
    
    def __eq__(self, other):
        # 类型检查：如果other不是Entity类型，则返回False
        if not isinstance(other, Entity):
            return False
        # 只有当两个text字段都不为空且相等时，才认为是相等的（TP）
        if self.text and other.text:
            return (self.start == other.start and 
                    self.end == other.end and 
                    self.type == other.type and 
                    self.text == other.text)
        # 其他情况都不相等
        return False


@dataclass
class SemiStructuredEntity:
    """Represents a semi-structured entity with text, start, and end."""
    start: int
    end: int
    text: str
    
    def __hash__(self):
        return hash((self.text, self.start, self.end))
    
    def __eq__(self, other):
        # 类型检查：如果other不是SemiStructuredEntity类型，则返回False
        if not isinstance(other, SemiStructuredEntity):
            return False
        return (self.text == other.text and 
                self.start == other.start and 
                self.end == other.end)

class SemiStructuredMetrics:
    """Calculate semi-structured entity extraction metrics (no type classification)."""
    
    def __init__(self, matching_method: str = "position", text_match_mode: str = "exact", 
                 similarity_threshold: float = 0.8, similarity_model: str = "tiny", enable_validation: bool = False):
        """
        Initialize semi-structured metrics calculator.
        
        Args:
            matching_method: Method for matching entities
                - "position": Compare start and end positions (exact match only)
                - "text": Compare text content
            text_match_mode: Mode for text matching (only used when matching_method="text")
                - "exact": Exact text match
                - "overlap": Overlapping text match
                - "similarity": Similarity-based match using BERT embeddings
            similarity_threshold: Similarity threshold for similarity matching (0-1, default 0.8)
            similarity_model: BERT model to use for similarity ("tiny", "small", "medium", "large", or custom model name)
            enable_validation: If True, validate entity consistency
                              If False, skip validation to allow normal evaluation scenarios
        """
        if matching_method not in ["position", "text"]:
            raise ValueError("matching_method must be 'position' or 'text'")
        if text_match_mode not in ["exact", "overlap", "similarity"]:
            raise ValueError("text_match_mode must be 'exact', 'overlap', or 'similarity'")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        self.matching_method = matching_method
        self.text_match_mode = text_match_mode
        self.similarity_threshold = similarity_threshold
        self.similarity_model = similarity_model
        self.enable_validation = enable_validation
    
    def parse_entities(self, data: Dict[str, Any]) -> Set[SemiStructuredEntity]:
        """
        Parse semi-structured entities from JSON format.
        
        Args:
            data: Dictionary with "entities" key containing list of entity dicts
        
        Returns:
            Set of SemiStructuredEntity objects
        """
        entities = set()
        
        if "entities" not in data:
            return entities
        
        for entity_dict in data["entities"]:
            try:
                # 检查必需字段
                if "start" not in entity_dict:
                    print(f"Warning: Missing required field 'start' in entity: {entity_dict}")
                    continue
                if "end" not in entity_dict:
                    print(f"Warning: Missing required field 'end' in entity: {entity_dict}")
                    continue
                
                # 只支持text字段
                if "text" not in entity_dict:
                    print(f"Warning: Missing required field 'text' in entity: {entity_dict}")
                    continue
                
                entity = SemiStructuredEntity(
                    start=entity_dict["start"],
                    end=entity_dict["end"],
                    text=entity_dict["text"]
                )
                entities.add(entity)
            except (KeyError, ValueError) as e:
                print(f"Warning: Error processing entity {entity_dict}: {e}")
                continue
        
        return entities
    
    def calculate_metrics(self, y_true: List[Dict[str, Any]], 
                         y_pred: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate semi-structured entity extraction metrics.
        
        Args:
            y_true: List of ground truth entity dictionaries
            y_pred: List of predicted entity dictionaries
        
        Returns:
            Dictionary containing precision, recall, F1 score, and all metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
    
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for true_data, pred_data in zip(y_true, y_pred):
            true_entities = self.parse_entities(true_data)
            pred_entities = self.parse_entities(pred_data)
            
            if self.matching_method == "position":
                tp, fp, fn = self._calculate_position_metrics(true_entities, pred_entities)
            elif self.matching_method == "text":
                if self.text_match_mode == "exact":
                    tp, fp, fn = self._calculate_text_exact_metrics(true_entities, pred_entities)
                elif self.text_match_mode == "overlap":
                    tp, fp, fn = self._calculate_text_overlap_metrics(true_entities, pred_entities)
                else:  # similarity
                    tp, fp, fn = self._calculate_text_similarity_metrics(true_entities, pred_entities)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate precision, recall, and F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "accuracy": 0.0,  # Not meaningful for entity extraction
            "true_negatives": 0  # Not meaningful for entity extraction
        }
    

    
    def _validate_entity_consistency(self, true_entities: Set[SemiStructuredEntity], 
                                   pred_entities: Set[SemiStructuredEntity]) -> None:
        """
        Validate that all entities have matching start and end positions.
        If any entity has different start or end, raise an error.
        """
        # Create sets of (start, end) tuples for comparison
        true_positions = {(e.start, e.end) for e in true_entities}
        pred_positions = {(e.start, e.end) for e in pred_entities}
        
        # Check for conflicting positions (same start but different end, or same end but different start)
        for true_entity in true_entities:
            for pred_entity in pred_entities:
                # If they share the same start position, they must have the same end
                if true_entity.start == pred_entity.start:
                    if true_entity.end != pred_entity.end:
                        raise ValueError(f"Position conflict at start {true_entity.start}: true ({true_entity.start}, {true_entity.end}) vs pred ({pred_entity.start}, {pred_entity.end})")
                
                # If they share the same end position, they must have the same start
                if true_entity.end == pred_entity.end:
                    if true_entity.start != pred_entity.start:
                        raise ValueError(f"Position conflict at end {true_entity.end}: true ({true_entity.start}, {true_entity.end}) vs pred ({pred_entity.start}, {pred_entity.end})")
    
    def _entities_overlap(self, entity1: SemiStructuredEntity, entity2: SemiStructuredEntity) -> bool:
        """Check if two entities overlap."""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def _calculate_position_metrics(self, true_entities: Set[SemiStructuredEntity], 
                                  pred_entities: Set[SemiStructuredEntity]) -> Tuple[int, int, int]:
        """Calculate metrics using position matching (exact start and end match)."""
        # Validate entity consistency if enabled
        if self.enable_validation:
            self._validate_entity_consistency(true_entities, pred_entities)
        
        # Calculate TP: entities that match exactly in position
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        for pred_entity in pred_entities:
            for true_entity in true_entities:
                if (true_entity not in matched_true and 
                    pred_entity not in matched_pred and
                    pred_entity.start == true_entity.start and 
                    pred_entity.end == true_entity.end):
                    tp += 1
                    matched_true.add(true_entity)
                    matched_pred.add(pred_entity)
                    break
        
        # Calculate FP and FN
        fp = len(pred_entities) - tp
        fn = len(true_entities) - tp
        
        return tp, fp, fn
    
    def _calculate_text_exact_metrics(self, true_entities: Set[SemiStructuredEntity], 
                                    pred_entities: Set[SemiStructuredEntity]) -> Tuple[int, int, int]:
        """Calculate metrics using exact text matching."""
        # Validate entity consistency if enabled
        if self.enable_validation:
            self._validate_entity_consistency(true_entities, pred_entities)
        
        # Calculate TP: entities that match exactly in text content
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        for pred_entity in pred_entities:
            for true_entity in true_entities:
                if (true_entity not in matched_true and
                    pred_entity not in matched_pred and
                    pred_entity.text == true_entity.text):
                    tp += 1
                    matched_true.add(true_entity)
                    matched_pred.add(pred_entity)
                    break
        
        # Calculate FP and FN
        fp = len(pred_entities) - tp
        fn = len(true_entities) - tp
        
        return tp, fp, fn
    
    def _calculate_text_overlap_metrics(self, true_entities: Set[SemiStructuredEntity], 
                                      pred_entities: Set[SemiStructuredEntity]) -> Tuple[int, int, int]:
        """Calculate metrics using overlapping text matching."""
        # Validate entity consistency if enabled
        if self.enable_validation:
            self._validate_entity_consistency(true_entities, pred_entities)
        
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        # Check each predicted entity
        for pred_entity in pred_entities:
            for true_entity in true_entities:
                if (true_entity not in matched_true and
                    pred_entity not in matched_pred and
                    self._texts_overlap(pred_entity.text, true_entity.text)):
                    tp += 1
                    matched_true.add(true_entity)
                    matched_pred.add(pred_entity)
                    break
        
        # Calculate FP and FN
        fp = len(pred_entities) - tp
        fn = len(true_entities) - tp
        
        return tp, fp, fn
    
    def _texts_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts overlap (have common substrings)."""
        # Simple overlap check: if one text is a substring of the other
        # For example: "患者名" overlaps with "患者姓名"
        return text1 in text2 or text2 in text1
    
    def _calculate_text_similarity_metrics(self, true_entities: Set[SemiStructuredEntity], 
                                         pred_entities: Set[SemiStructuredEntity]) -> Tuple[int, int, int]:
        """Calculate metrics using similarity-based text matching."""
        # Validate entity consistency if enabled
        if self.enable_validation:
            self._validate_entity_consistency(true_entities, pred_entities)
        
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        # Check each predicted entity
        for pred_entity in pred_entities:
            for true_entity in true_entities:
                if (true_entity not in matched_true and
                    pred_entity not in matched_pred and
                    is_texts_similar(pred_entity.text, true_entity.text, self.similarity_threshold, self.similarity_model)):
                    tp += 1
                    matched_true.add(true_entity)
                    matched_pred.add(pred_entity)
                    break
        
        # Calculate FP and FN
        fp = len(pred_entities) - tp
        fn = len(true_entities) - tp
        
        return tp, fp, fn


class NERMetrics:
    """Calculate NER evaluation metrics including F1 score."""
    
    def __init__(self, strict_match: bool = True, enable_validation: bool = False):
        """
        Initialize NER metrics calculator.
        
        Args:
            strict_match: If True, entities must match exactly (start, end, type, text)
                         If False, entities match if they overlap and have same type
            enable_validation: If True, validate entity consistency (start, end, type must match)
                              If False, skip validation to allow normal NER evaluation scenarios
        """
        self.strict_match = strict_match
        self.enable_validation = enable_validation
    
    def parse_entities(self, data: Dict[str, Any]) -> Set[Entity]:
        """
        Parse entities from JSON format.
        
        Args:
            data: Dictionary with "entities" key containing list of entity dicts
        
        Returns:
            Set of Entity objects
        """
        entities = set()
        
        if "entities" not in data:
            return entities
        
        for entity_dict in data["entities"]:
            try:
                # 检查必需字段
                if "start" not in entity_dict:
                    print(f"Warning: Missing required field 'start' in entity: {entity_dict}")
                    continue
                if "end" not in entity_dict:
                    print(f"Warning: Missing required field 'end' in entity: {entity_dict}")
                    continue
                if "type" not in entity_dict:
                    print(f"Warning: Missing required field 'type' in entity: {entity_dict}")
                    continue
                
                # text字段可以为空，使用get方法提供默认值
                text = entity_dict.get("text", "")
                
                entity = Entity(
                    start=entity_dict["start"],
                    end=entity_dict["end"],
                    type=entity_dict["type"],
                    text=text
                )
                entities.add(entity)
            except (KeyError, ValueError) as e:
                print(f"Warning: Error processing entity {entity_dict}: {e}")
                continue
        
        return entities
    
    def calculate_metrics(self, y_true: List[Dict[str, Any]], 
                         y_pred: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate NER evaluation metrics.
        
        Args:
            y_true: List of ground truth entity dictionaries
            y_pred: List of predicted entity dictionaries
        
        Returns:
            Dictionary containing precision, recall, F1 score, and all metrics
        """
        if len(y_true) != len(y_pred):
            # True and pred must have the same length (the same number of report pages)
            raise ValueError("y_true and y_pred must have the same length")
    
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        for true_data, pred_data in zip(y_true, y_pred):
            true_entities = self.parse_entities(true_data)
            pred_entities = self.parse_entities(pred_data)
            
            if self.strict_match:
                tp, fp, fn, tn = self._calculate_strict_metrics(true_entities, pred_entities)
            else:
                tp, fp, fn, tn = self._calculate_overlap_metrics(true_entities, pred_entities)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
        
        # Calculate precision, recall, and F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Note: Accuracy cannot be properly calculated in NER scenarios
        # because TN (True Negatives) cannot be determined
        # We set it to 0.0 to indicate it's not meaningful
        accuracy = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "true_negatives": total_tn
        }
    
    def _calculate_strict_metrics(self, true_entities: Set[Entity], 
                                 pred_entities: Set[Entity]) -> Tuple[int, int, int, int]:
        """Calculate metrics using strict matching (exact match)."""
        # Validate that all entities have matching start, end, and type (if enabled)
        if self.enable_validation:
            self._validate_entity_consistency(true_entities, pred_entities)
        
        # Separate entities with text and without text
        true_with_text = {e for e in true_entities if e.text}
        true_without_text = {e for e in true_entities if not e.text}
        pred_with_text = {e for e in pred_entities if e.text}
        pred_without_text = {e for e in pred_entities if not e.text}
        
        # Calculate TP: entities with text that match exactly
        tp = len(true_with_text & pred_with_text)
        
        # Note: TN (True Negatives) cannot be calculated in NER scenarios
        # because we don't track non-entity regions
        tn = 0
        
        # Calculate FP and FN using the correct logic
        fp = 0
        fn = 0
        
        # For each predicted entity, check if it matches any true entity
        for pred_entity in pred_entities:
            matched = False
            for true_entity in true_entities:
                # Check if they match (same position, type, and text handling)
                if (pred_entity.start == true_entity.start and 
                    pred_entity.end == true_entity.end and 
                    pred_entity.type == true_entity.type):
                    
                    # Both have text and text matches
                    if pred_entity.text and true_entity.text and pred_entity.text == true_entity.text:
                        matched = True
                        break
                    # Both have no text
                    elif not pred_entity.text and not true_entity.text:
                        matched = True
                        break
            if not matched:
                # This is a false positive - predicted entity doesn't match any true entity
                fp += 1
        
        # For each true entity, check if it matches any predicted entity
        for true_entity in true_entities:
            matched = False
            for pred_entity in pred_entities:
                # Check if they match (same position, type, and text handling)
                if (true_entity.start == pred_entity.start and 
                    true_entity.end == pred_entity.end and 
                    true_entity.type == pred_entity.type):
                    
                    # Both have text and text matches
                    if true_entity.text and pred_entity.text and true_entity.text == pred_entity.text:
                        matched = True
                        break
                    # Both have no text
                    elif not true_entity.text and not pred_entity.text:
                        matched = True
                        break
            if not matched:
                # This is a false negative - true entity doesn't match any predicted entity
                fn += 1
        
        return tp, fp, fn, tn
    
    def _calculate_overlap_metrics(self, true_entities: Set[Entity], 
                                  pred_entities: Set[Entity]) -> Tuple[int, int, int, int]:
        """Calculate metrics using overlap matching."""
        # Separate entities with text and without text
        true_with_text = {e for e in true_entities if e.text}
        true_without_text = {e for e in true_entities if not e.text}
        pred_with_text = {e for e in pred_entities if e.text}
        pred_without_text = {e for e in pred_entities if not e.text}
        
        tp = 0
        fp = 0
        fn = 0
        # Note: TN cannot be calculated in NER scenarios
        tn = 0
        
        # Check each predicted entity with text
        for pred_entity in pred_with_text:
            matched = False
            for true_entity in true_with_text:
                if (pred_entity.type == true_entity.type and 
                    self._entities_overlap(pred_entity, true_entity)):
                    tp += 1
                    matched = True
                    break
            if not matched:
                fp += 1
        
        # Check each predicted entity without text
        for pred_entity in pred_without_text:
            matched = False
            for true_entity in true_without_text:
                if (pred_entity.type == true_entity.type and 
                    self._entities_overlap(pred_entity, true_entity)):
                    tn += 1
                    matched = True
                    break
            if not matched:
                fp += 1
        
        # Count false negatives (true entities not matched)
        for true_entity in true_with_text:
            matched = False
            for pred_entity in pred_with_text:
                if (true_entity.type == pred_entity.type and 
                    self._entities_overlap(true_entity, pred_entity)):
                    matched = True
                    break
            if not matched:
                fn += 1
        
        for true_entity in true_without_text:
            matched = False
            for pred_entity in pred_without_text:
                if (true_entity.type == pred_entity.type and 
                    self._entities_overlap(true_entity, pred_entity)):
                    matched = True
                    break
            if not matched:
                fn += 1
        
        return tp, fp, fn, tn
    
    def _validate_entity_consistency(self, true_entities: Set[Entity], pred_entities: Set[Entity]) -> None:
        """
        Validate that all entities have matching start, end, and type.
        If any entity has different start, end, or type, raise an error.
        """
        # Create sets of (start, end, type) tuples for comparison
        true_positions = {(e.start, e.end, e.type) for e in true_entities}
        pred_positions = {(e.start, e.end, e.type) for e in pred_entities}
        
        # Check for conflicting positions (same start/end but different type)
        true_start_end = {(e.start, e.end) for e in true_entities}
        pred_start_end = {(e.start, e.end) for e in pred_entities}
        
        # Find positions that exist in both but might have different types
        common_positions = true_start_end & pred_start_end
        
        for start, end in common_positions:
            true_types = {e.type for e in true_entities if e.start == start and e.end == end}
            pred_types = {e.type for e in pred_entities if e.start == start and e.end == end}
            
            # If there's a type mismatch at the same position, raise an error
            if true_types != pred_types:
                raise ValueError(f"Type mismatch at position ({start}, {end}): true types {true_types} vs pred types {pred_types}")
        
        # Check for conflicting positions (same start but different end, or same end but different start)
        # This ensures that if a position exists in both true and pred, it must have exactly the same boundaries
        for true_entity in true_entities:
            for pred_entity in pred_entities:
                # If they share the same start position, they must have the same end and type
                if true_entity.start == pred_entity.start:
                    if true_entity.end != pred_entity.end or true_entity.type != pred_entity.type:
                        raise ValueError(f"Position conflict at start {true_entity.start}: true ({true_entity.start}, {true_entity.end}, {true_entity.type}) vs pred ({pred_entity.start}, {pred_entity.end}, {pred_entity.type})")
                
                # If they share the same end position, they must have the same start and type
                if true_entity.end == pred_entity.end:
                    if true_entity.start != pred_entity.start or true_entity.type != pred_entity.type:
                        raise ValueError(f"Position conflict at end {true_entity.end}: true ({true_entity.start}, {true_entity.end}, {true_entity.type}) vs pred ({pred_entity.start}, {pred_entity.end}, {pred_entity.type})")
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap."""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def calculate_metrics_by_type(self, y_true: List[Dict[str, Any]], 
                                 y_pred: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each entity type separately.
        
        Args:
            y_true: List of ground truth entity dictionaries
            y_pred: List of predicted entity dictionaries
        
        Returns:
            Dictionary mapping entity types to their metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Collect all entity types
        all_types = set()
        for data in y_true + y_pred:
            if "entities" in data:
                for entity in data["entities"]:
                    if "type" in entity:
                        all_types.add(entity["type"])
        
        # Calculate metrics for each type
        type_metrics = {}
        for entity_type in all_types:
            type_tp = 0
            type_fp = 0
            type_fn = 0
            
            for true_data, pred_data in zip(y_true, y_pred):
                true_entities = {e for e in self.parse_entities(true_data) if e.type == entity_type}
                pred_entities = {e for e in self.parse_entities(pred_data) if e.type == entity_type}
                
                if self.strict_match:
                    tp, fp, fn, tn = self._calculate_strict_metrics(true_entities, pred_entities)
                else:
                    tp, fp, fn, tn = self._calculate_overlap_metrics(true_entities, pred_entities)
                
                type_tp += tp
                type_fp += fp
                type_fn += fn
            
            # Calculate metrics for this type
            precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
            recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            type_metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": type_tp,
                "false_positives": type_fp,
                "false_negatives": type_fn
            }
        
        return type_metrics 