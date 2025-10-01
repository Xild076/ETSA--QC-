from typing import Callable, List, Dict, Tuple, Optional, Any
import logging
import math
import networkx as nx
from combiners import COMBINERS, _modifier_polarity_summary

logger = logging.getLogger(__name__)

ENTITY_ROLES: Dict[str, Dict[str, str]] = {
    "none": {"description": "No specific role"},
    "actor": {"description": "Initiates an action"},
    "target": {"description": "Receives an action"},
    "parent": {"description": "Owns a child entity"},
    "child": {"description": "Belongs to a parent"},
    "associate": {"description": "Is connected to another entity"},
}

RELATION_TYPES: Dict[str, Dict[str, str]] = {
    "temporal": {"description": "Indicates a temporal relationship"},
    "action": {"description": "Indicates an action between entities"},
    "belonging": {"description": "Indicates a belonging relationship"},
    "association": {"description": "Indicates an association between entities"},
}

class RelationGraph:
    def __init__(
        self,
        text: str = "",
        clauses: List[str] = [],
        sentiment_analyzer_system=None,
        combiner: str = "contextual_v3",
        combiner_params: Optional[Dict[str, Any]] = None,
    ):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.text: str = text
        self.clauses: List[str] = clauses
        self.sentiment_analyzer_system = sentiment_analyzer_system
        self.entity_ids: set[int] = set()
        self.aggregate_sentiments: Dict[int, float] = {}
        self.combiner = combiner
        self.combiner_params = combiner_params or {}
        self._last_combiner_signature: Tuple[str, Tuple[Tuple[str, Any], ...]] = (
            combiner,
            tuple(sorted((combiner_params or {}).items())),
        )


    def _validate_role(self, role: str) -> None:
        if role not in ENTITY_ROLES:
            raise ValueError(f"Invalid entity role: {role}. Valid roles are: {list(ENTITY_ROLES.keys())}")

    def _validate_relation(self, relation: str) -> None:
        if relation not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation}. Valid relations are: {list(RELATION_TYPES.keys())}")

    def _node_key(self, entity_id: int, clause_layer: int) -> Tuple[int, int]:
        return (entity_id, clause_layer)

    def _assert_node_exists(self, entity_id: int, clause_layer: int) -> None:
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        key = self._node_key(entity_id, clause_layer)
        if key not in self.graph.nodes:
            raise ValueError(f"Node {key} not found in the graph.")

    def _sent(self, text: str, aspect: str = "", modifiers: List[str] = None) -> float:
        """Enhanced sentiment analysis with optional aspect-focused context."""
        if not self.sentiment_analyzer_system:
            return 0.0
        try:
            # If aspect and modifiers provided, use focused analysis
            if aspect and modifiers is not None:
                from sentiment.sentiment import analyze_aspect_sentiment_focused
                # Create a wrapper function for the sentiment analyzer
                def analyzer_wrapper(text_input):
                    return self.sentiment_analyzer_system.analyze(text_input or "")
                
                result = analyze_aspect_sentiment_focused(
                    text or "", aspect, modifiers or [], analyzer_wrapper
                )
                return self._coerce_sentiment_score(result)
            else:
                # Standard sentiment analysis
                raw_result = self.sentiment_analyzer_system.analyze(text or "")
                return self._coerce_sentiment_score(raw_result)
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return 0.0

    def _coerce_sentiment_score(self, result: Any) -> float:
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, dict):
            for key in ("aggregate", "score", "compound", "polarity", "sentiment", "value", "confidence", "confidence_score"):
                val = result.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
            for val in result.values():
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, dict):
                    try:
                        return self._coerce_sentiment_score(val)
                    except Exception:
                        continue
        raise ValueError("Unable to extract sentiment score")

    def compute_text_sentiment(self, text: Optional[str] = None) -> float:
        target = text if text is not None else self.text
        if not target:
            return 0.0
        try:
            return self._sent(target)
        except Exception:
            return 0.0

    def get_new_unique_entity_id(self) -> int:
        new_id = 1
        while new_id in self.entity_ids:
            new_id += 1
        return new_id

    def add_entity_node(
        self,
        id: int,
        head: str,
        modifier: List[str],
        entity_role: str,
        clause_layer: int,
        threshold: tuple = (0.1, -0.1),
    ) -> None:
        logger.info("Adding entity node %s at layer %s", id, clause_layer)
        self._validate_role(entity_role)
        self.entity_ids.add(id)
        key = self._node_key(id, clause_layer)

        def _unique_modifiers(spans: List[str]) -> List[str]:
            unique: List[str] = []
            seen = set()
            for span in spans or []:
                if not isinstance(span, str):
                    continue
                cleaned = span.strip()
                if not cleaned:
                    continue
                norm = cleaned.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                unique.append(cleaned)
            return unique

        def _compute_modifier_sentiment(head_text: str, spans: List[str]) -> Tuple[float, Dict[str, float], Optional[float]]:
            if not spans:
                return 0.0, {}, None
            per_scores: Dict[str, float] = {}
            weighted_sum = 0.0
            magnitude_sum = 0.0
            for span in spans:
                score_raw = self._sent(span)
                if not isinstance(score_raw, (int, float)):
                    continue
                score = float(score_raw)
                per_scores[span] = score
                weight = abs(score)
                if weight > 0:
                    weighted_sum += score * weight
                    magnitude_sum += weight
            if not per_scores:
                return 0.0, {}, None
            mean_score = (
                weighted_sum / magnitude_sum if magnitude_sum > 0 else sum(per_scores.values()) / len(per_scores)
            )
            context_score: Optional[float] = None
            if head_text and spans:
                context_text = ". ".join(f"{head_text} {span}" for span in spans)
                ctx_raw = self._sent(context_text)
                if isinstance(ctx_raw, (int, float)):
                    context_score = float(ctx_raw)
            if context_score is not None:
                combined = 0.7 * mean_score + 0.3 * context_score
            else:
                combined = mean_score
            return combined, per_scores, context_score

        clean_head = (head or "").strip()
        modifiers_clean = _unique_modifiers(modifier)
        # Use focused sentiment analysis for head with its modifiers
        head_sentiment = float(self._sent(clean_head, aspect=clean_head, modifiers=modifiers_clean)) if clean_head else 0.0
        modifier_sentiment, per_scores, context_score = _compute_modifier_sentiment(clean_head, modifiers_clean)

        final_sentiment, base_justification, extras = self._combine_sentiments(
            clean_head,
            head_sentiment,
            modifiers_clean,
            modifier_sentiment,
            threshold,
            context_score,
            per_scores,
            self.text,
        )

        head_adjusted = extras.get("head_adjusted", head_sentiment)
        modifier_adjusted = extras.get("modifier_adjusted", modifier_sentiment)
        heuristic_notes = extras.get("heuristic_notes", []) or []

        polarity_counts = _modifier_polarity_summary(per_scores, threshold)
        detail_parts = [
            f"head={head_sentiment:+.2f}",
            f"modifier_mean={modifier_sentiment:+.2f}",
        ]
        if head_adjusted != head_sentiment:
            detail_parts.append(f"head_adj={head_adjusted:+.2f}")
        if modifier_adjusted != modifier_sentiment:
            detail_parts.append(f"modifier_adj={modifier_adjusted:+.2f}")
        if context_score is not None:
            detail_parts.append(f"context={context_score:+.2f}")
        if per_scores:
            ordered = [f"{span}: {per_scores[span]:+.2f}" for span in modifiers_clean if span in per_scores]
            detail_parts.append("per_modifiers={" + ", ".join(ordered) + "}")
        if heuristic_notes:
            detail_parts.append("heuristics=" + ", ".join(heuristic_notes))
        sentiment_justification = base_justification
        if detail_parts:
            sentiment_justification = f"{base_justification} Details: {'; '.join(detail_parts)}."

        self.graph.add_node(
            key,
            head=clean_head,
            modifier=modifiers_clean,
            text=self.text,
            entity_role=entity_role,
            clause_layer=clause_layer,
            init_sentiment=final_sentiment,
            head_sentiment=head_sentiment,
            head_sentiment_adjusted=head_adjusted,
            modifier_sentiment=modifier_sentiment,
            modifier_sentiment_adjusted=modifier_adjusted,
            modifier_context_sentiment=context_score,
            modifier_sentiment_components=per_scores,
            modifier_polarity_counts=polarity_counts,
            modifier_conflict=polarity_counts["positive"] > 0 and polarity_counts["negative"] > 0,
            sentiment_justification=sentiment_justification,
            sentiment_strategy=self.combiner,
            sentiment_heuristics=heuristic_notes,
            sentiment_threshold=threshold,
        )

    def get_entities_at_layer(self, clause_layer: int) -> List[Dict]:
        entities = []
        for (entity_id, layer), data in self.graph.nodes(data=True):
            if layer == clause_layer:
                entity_info = {
                    "entity_id": entity_id,
                    "head": data.get("head", ""),
                    "modifier": data.get("modifier", []),
                    "entity_role": data.get("entity_role", ""),
                    "clause_layer": layer,
                    "init_sentiment": data.get("init_sentiment", 0.0),
                    "compound_sentiment": data.get("compound_sentiment", None),
                }
                entities.append(entity_info)
        return entities
    
    def get_all_entity_mentions(self, entity_id: int) -> List[Dict]:
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        mentions = []
        for (eid, layer), data in self.graph.nodes(data=True):
            if eid == entity_id:
                mentions.append(data.get("head", ""))
        return mentions

    def add_entity_modifier(self, entity_id: int, modifier: List[str], clause_layer: int) -> None:
        logger.info("Adding entity modifiers for %s at layer %s", entity_id, clause_layer)
        self._assert_node_exists(entity_id, clause_layer)
        key = self._node_key(entity_id, clause_layer)
        node = self.graph.nodes[key]

        def _unique_modifiers(spans: List[str]) -> List[str]:
            unique: List[str] = []
            seen = set()
            for span in spans or []:
                if not isinstance(span, str):
                    continue
                cleaned = span.strip()
                if not cleaned:
                    continue
                norm = cleaned.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                unique.append(cleaned)
            return unique

        def _compute_modifier_sentiment(head_text: str, spans: List[str]) -> Tuple[float, Dict[str, float], Optional[float]]:
            if not spans:
                return 0.0, {}, None
            per_scores: Dict[str, float] = {}
            weighted_sum = 0.0
            magnitude_sum = 0.0
            for span in spans:
                score_raw = self._sent(span)
                if not isinstance(score_raw, (int, float)):
                    continue
                score = float(score_raw)
                per_scores[span] = score
                weight = abs(score)
                if weight > 0:
                    weighted_sum += score * weight
                    magnitude_sum += weight
            if not per_scores:
                return 0.0, {}, None
            mean_score = (
                weighted_sum / magnitude_sum if magnitude_sum > 0 else sum(per_scores.values()) / len(per_scores)
            )
            context_score: Optional[float] = None
            if head_text and spans:
                context_text = ". ".join(f"{head_text} {span}" for span in spans)
                ctx_raw = self._sent(context_text)
                if isinstance(ctx_raw, (int, float)):
                    context_score = float(ctx_raw)
            if context_score is not None:
                combined = 0.7 * mean_score + 0.3 * context_score
            else:
                combined = mean_score
            return combined, per_scores, context_score

        existing_modifiers = _unique_modifiers(node.get("modifier", []))
        incoming_modifiers = _unique_modifiers(modifier)
        previous_norm = {span.lower() for span in existing_modifiers}
        combined_modifiers: List[str] = []
        seen = set()
        for span in existing_modifiers + incoming_modifiers:
            norm = span.lower()
            if norm in seen:
                continue
            seen.add(norm)
            combined_modifiers.append(span)
        added_modifiers = [span for span in combined_modifiers if span.lower() not in previous_norm]

        node['modifier'] = combined_modifiers
        head_text = (node.get('head') or '').strip()
        head_sentiment = node.get('head_sentiment')
        if head_sentiment is None:
            head_sentiment = float(self._sent(head_text)) if head_text else 0.0

        modifier_sentiment, per_scores, context_score = _compute_modifier_sentiment(head_text, combined_modifiers)
        threshold = node.get('sentiment_threshold', (0.1, -0.1))
        final_sentiment, base_justification, extras = self._combine_sentiments(
            head_text,
            head_sentiment,
            combined_modifiers,
            modifier_sentiment,
            threshold,
            context_score,
            per_scores,
            self.text,
        )

        head_adjusted = extras.get("head_adjusted", head_sentiment)
        modifier_adjusted = extras.get("modifier_adjusted", modifier_sentiment)
        heuristic_notes = extras.get("heuristic_notes", []) or []

        polarity_counts = _modifier_polarity_summary(per_scores, threshold)
        detail_parts = [
            f"head={head_sentiment:+.2f}",
            f"modifier_mean={modifier_sentiment:+.2f}",
        ]
        if head_adjusted != head_sentiment:
            detail_parts.append(f"head_adj={head_adjusted:+.2f}")
        if modifier_adjusted != modifier_sentiment:
            detail_parts.append(f"modifier_adj={modifier_adjusted:+.2f}")
        if context_score is not None:
            detail_parts.append(f"context={context_score:+.2f}")
        if per_scores:
            ordered = [f"{span}: {per_scores[span]:+.2f}" for span in combined_modifiers if span in per_scores]
            detail_parts.append("per_modifiers={" + ", ".join(ordered) + "}")
        if added_modifiers:
            detail_parts.append("added=" + ", ".join(added_modifiers))
        if heuristic_notes:
            detail_parts.append("heuristics=" + ", ".join(heuristic_notes))
        sentiment_justification = base_justification
        if detail_parts:
            sentiment_justification = f"{base_justification} Details: {'; '.join(detail_parts)}."

        node['modifier'] = combined_modifiers
        node['head_sentiment'] = head_sentiment
        node['head_sentiment_adjusted'] = head_adjusted
        node['modifier_sentiment'] = modifier_sentiment
        node['modifier_sentiment_adjusted'] = modifier_adjusted
        node['modifier_context_sentiment'] = context_score
        node['modifier_sentiment_components'] = per_scores
        node['modifier_polarity_counts'] = polarity_counts
        node['modifier_conflict'] = polarity_counts['positive'] > 0 and polarity_counts['negative'] > 0
        node['init_sentiment'] = final_sentiment
        node['sentiment_justification'] = sentiment_justification
        node['last_added_modifiers'] = added_modifiers
        node['sentiment_strategy'] = self.combiner
        node['sentiment_heuristics'] = heuristic_notes

    def set_entity_role(self, entity_id: int, entity_role: str, clause_layer: int) -> None:
        logger.info("Setting entity role...")
        self._validate_role(entity_role)
        self._assert_node_exists(entity_id, clause_layer)
        key = self._node_key(entity_id, clause_layer)
        self.graph.nodes[key]["entity_role"] = entity_role

    def add_temporal_edge(self, entity_id: int) -> None:
        logger.info("Adding temporal edge...")
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        layers = sorted(layer for (eid, layer) in self.graph.nodes if eid == entity_id)
        for i in range(len(layers) - 1):
            u = self._node_key(entity_id, layers[i])
            v = self._node_key(entity_id, layers[i + 1])
            self.graph.add_edge(u, v, relation="temporal")

    def _combine_sentiments(
        self,
        head: str,
        head_sentiment: float,
        modifier: List[str],
        modifier_sentiment: float,
        threshold: tuple = (0.1, -0.1),
        context_score: Optional[float] = None,
        per_scores: Optional[Dict[str, float]] = None,
        clause_text: str = "",
    ) -> Tuple[float, str, Dict[str, Any]]:
        combiner_cls = COMBINERS.get(self.combiner)
        if combiner_cls is None:
            logger.warning(f"Unknown combiner '{self.combiner}'; falling back to contextual_v3.")
            combiner_cls = COMBINERS["contextual_v3"]
        
        combiner_obj = combiner_cls.__class__(**self.combiner_params)
        try:
            return combiner_obj.combine(
                head,
                head_sentiment,
                modifier,
                modifier_sentiment,
                threshold,
                context_score,
                per_scores or {},
                clause_text,
            )
        except Exception:
            logger.exception(f"{self.combiner} sentiment combiner failed; falling back to legacy blend.")
            from .combiners import _legacy_blend, _clamp_score
            score, justification = _legacy_blend(head_sentiment, modifier_sentiment, bool(modifier), threshold)
            return _clamp_score(score), justification, {
                "head_adjusted": head_sentiment,
                "modifier_adjusted": modifier_sentiment,
                "heuristic_notes": ["combiner_fallback"],
            }

    def refresh_with_combiner(
        self,
        combiner: str,
        combiner_params: Optional[Dict[str, Any]] = None,
        action_function: Optional[Callable[..., Any]] = None,
        association_function: Optional[Callable[..., Any]] = None,
        belonging_function: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Recompute node- and relation-level sentiments for a combiner configuration."""
        self.combiner = combiner
        self.combiner_params = combiner_params or {}
        self._last_combiner_signature = (
            combiner,
            tuple(sorted((self.combiner_params or {}).items())),
        )

        base_instance = COMBINERS.get(self.combiner)
        if base_instance is None:
            raise ValueError(f"Unknown combiner '{self.combiner}'")

        combiner_cls = base_instance.__class__
        combiner_obj = combiner_cls(**self.combiner_params)

        for node_key, data in list(self.graph.nodes(data=True)):
            clause_idx = int(data.get("clause_layer", 0) or 0)
            if self.clauses and 0 <= clause_idx < len(self.clauses):
                clause_text = self.clauses[clause_idx]
            else:
                clause_text = self.text

            threshold = data.get("sentiment_threshold", (0.1, -0.1))
            init_sentiment, justification, extras = combiner_obj.combine(
                head_text=data.get("head", ""),
                head_sentiment=float(data.get("head_sentiment", 0.0) or 0.0),
                modifiers=list(data.get("modifier", [])),
                modifier_sentiment=float(data.get("modifier_sentiment", 0.0) or 0.0),
                threshold=threshold,
                context_score=data.get("modifier_context_sentiment"),
                per_scores=data.get("modifier_sentiment_components", {}),
                clause_text=clause_text,
            )
            extras = extras or {}

            head_adjusted = extras.get("head_adjusted", data.get("head_sentiment_adjusted", data.get("head_sentiment", 0.0)))
            modifier_adjusted = extras.get("modifier_adjusted", data.get("modifier_sentiment_adjusted", data.get("modifier_sentiment", 0.0)))

            data["init_sentiment"] = init_sentiment
            data["head_sentiment_adjusted"] = head_adjusted
            data["modifier_sentiment_adjusted"] = modifier_adjusted
            data["sentiment_justification"] = justification
            data["sentiment_strategy"] = self.combiner
            data["sentiment_heuristics"] = extras.get("heuristic_notes", [])
            if "combiner_debug" in extras:
                data["combiner_debug"] = extras["combiner_debug"]

        # Ensure edge payloads that depend on node sentiment are refreshed.
        for _, _, edge_data in self.graph.edges(data=True):
            if edge_data.get("relation") == "action":
                action_node = edge_data.get("action")
                if action_node in self.graph.nodes:
                    edge_data["init_sentiment"] = float(self.graph.nodes[action_node].get("init_sentiment", 0.0))

        # Clear previous compound / aggregate caches to avoid stale values.
        for node_key in list(self.graph.nodes):
            node_data = self.graph.nodes[node_key]
            node_data.pop("compound_sentiment", None)
        self.aggregate_sentiments.clear()

        if action_function is not None:
            self.run_compound_action_sentiment_calculations(action_function)
        if association_function is not None:
            self.run_compound_association_sentiment_calculations(association_function)
        if belonging_function is not None:
            self.run_compound_belonging_sentiment_calculations(belonging_function)



    def add_belonging_edge(self, parent_id: int, child_id: int, clause_layer: int) -> None:
        logger.info(f"Adding belonging edge between {parent_id} and {child_id}...")
        if parent_id not in self.entity_ids or child_id not in self.entity_ids:
            raise ValueError(f"Parent ID {parent_id} or Child ID {child_id} not found in the graph.")
        parent = self._node_key(parent_id, clause_layer)
        child = self._node_key(child_id, clause_layer)
        if parent not in self.graph.nodes or child not in self.graph.nodes:
            raise ValueError(f"Layer {clause_layer} missing parent or child node.")
        self.graph.add_edge(parent, child, relation="belonging", parent=parent, child=child)

    def add_action_edge(
        self,
        actor_id: int,
        target_id: int,
        clause_layer: int,
        head: str = "",
        modifier: Optional[List[str]] = None,
        action_id: Optional[int] = None,
    ) -> None:
        logger.info(f"Adding action edge actor={actor_id} target={target_id} at layer {clause_layer}...")
        if actor_id not in self.entity_ids or target_id not in self.entity_ids:
            raise ValueError(f"Actor ID {actor_id} or Target ID {target_id} not found in the graph.")
        if action_id is not None and action_id not in self.entity_ids:
            raise ValueError(f"Action ID {action_id} not found in the graph.")

        actor = self._node_key(actor_id, clause_layer)
        target = self._node_key(target_id, clause_layer)
        if actor not in self.graph.nodes or target not in self.graph.nodes:
            raise ValueError(f"Layer {clause_layer} missing actor or target node.")

        action = None
        if action_id is not None:
            action = self._node_key(action_id, clause_layer)
            if action not in self.graph.nodes:
                raise ValueError(f"Layer {clause_layer} missing action node.")

        modifiers = list(modifier or [])
        action_text = " ".join(part for part in [(head or "").strip()] + modifiers if part).strip()

        if action is not None:
            init_sent = float(self.graph.nodes[action].get("init_sentiment", 0.0))
        elif action_text:
            init_sent = float(self._sent(action_text))
        else:
            init_sent = 0.0

        edge_payload = {
            "relation": "action",
            "actor": actor,
            "target": target,
            "init_sentiment": init_sent,
        }
        if action is not None:
            edge_payload["action"] = action
        if head:
            edge_payload["action_head"] = head
        if modifiers:
            edge_payload["action_modifier"] = modifiers

        self.graph.add_edge(actor, target, **edge_payload)

    def add_association_edge(self, entity1_id: int, entity2_id: int, clause_layer: int) -> None:
        logger.info(f"Adding association edge between {entity1_id} and {entity2_id}...")
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity1 ID {entity1_id} or Entity2 ID {entity2_id} not found in the graph.")
        e1 = self._node_key(entity1_id, clause_layer)
        e2 = self._node_key(entity2_id, clause_layer)
        if e1 not in self.graph.nodes or e2 not in self.graph.nodes:
            raise ValueError(f"Layer {clause_layer} missing association endpoints.")
        self.graph.add_edge(e1, e2, relation="association", entity1=e1, entity2=e2)

    def run_compound_action_sentiment_calculations(self, function: Optional[Callable] = None) -> None:
        logger.info("Running compound action sentiment calculations...")
        if function is None:
            raise ValueError("No function provided for compound sentiment calculation")
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "action":
                actor = data["actor"]
                target = data["target"]
                actor_init = float(self.graph.nodes[actor].get("init_sentiment", 0.0))
                action_init = float(data.get("init_sentiment", 0.0))
                target_init = float(self.graph.nodes[target].get("init_sentiment", 0.0))
                result = function(actor_init, action_init, target_init)
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    actor_score, target_score = result[0], result[1]
                else:
                    actor_score = result
                    target_score = result
                actor_value = actor_score if isinstance(actor_score, (int, float)) else 0.0
                target_value = target_score if isinstance(target_score, (int, float)) else 0.0
                self.graph.nodes[actor]["compound_sentiment"] = float(actor_value)
                self.graph.nodes[target]["compound_sentiment"] = float(target_value)

    def run_compound_belonging_sentiment_calculations(self, function: Optional[Callable] = None) -> None:
        logger.info("Running compound belonging sentiment calculations...")
        if function is None:
            raise ValueError("No function provided for compound sentiment calculation")
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "belonging":
                parent = data["parent"]
                child = data["child"]
                p_init = float(self.graph.nodes[parent].get("init_sentiment", 0.0))
                c_init = float(self.graph.nodes[child].get("init_sentiment", 0.0))
                result = function(p_init, c_init)
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    parent_score, child_score = result[0], result[1]
                else:
                    parent_score = result
                    child_score = result
                parent_value = parent_score if isinstance(parent_score, (int, float)) else 0.0
                child_value = child_score if isinstance(child_score, (int, float)) else 0.0
                self.graph.nodes[parent]["compound_sentiment"] = float(parent_value)
                self.graph.nodes[child]["compound_sentiment"] = float(child_value)

    def run_compound_association_sentiment_calculations(self, function: Optional[Callable] = None) -> None:
        logger.info("Running compound association sentiment calculations...")
        if function is None:
            raise ValueError("No function provided for compound sentiment calculation")
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "association":
                e1 = data["entity1"]
                e2 = data["entity2"]
                s1 = float(self.graph.nodes[e1].get("init_sentiment", 0.0))
                s2 = float(self.graph.nodes[e2].get("init_sentiment", 0.0))
                result = function(s1, s2)
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    score_one, score_two = result[0], result[1]
                else:
                    score_one = result
                    score_two = result
                value_one = score_one if isinstance(score_one, (int, float)) else 0.0
                value_two = score_two if isinstance(score_two, (int, float)) else 0.0
                self.graph.nodes[e1]["compound_sentiment"] = float(value_one)
                self.graph.nodes[e2]["compound_sentiment"] = float(value_two)

    def run_aggregate_sentiment_calculations(self, entity_id: int, function: Optional[Callable] = None) -> float:
        logger.info(f"Running aggregate sentiment calculations for entity {entity_id}...")
        if function is None:
            raise ValueError("No function provided for aggregate sentiment calculation")
        layers = sorted(layer for (eid, layer) in self.graph.nodes if eid == entity_id)
        sentiments: List[float] = []
        for layer in layers:
            key = self._node_key(entity_id, layer)
            if "compound_sentiment" in self.graph.nodes[key]:
                sentiment = float(self.graph.nodes[key].get("compound_sentiment", 0.0))
            else:
                sentiment = float(self.graph.nodes[key].get("init_sentiment", 0.0))
            
            # Validate and clamp sentiment values
            if not math.isfinite(sentiment):
                sentiment = 0.0
            sentiment = max(-1.0, min(1.0, sentiment))
            sentiments.append(sentiment)
        
        result = float(function(sentiments)) if sentiments else 0.0
        # Ensure result is finite and within bounds
        if not math.isfinite(result):
            result = 0.0
        result = max(-1.0, min(1.0, result))
        
        self.aggregate_sentiments[entity_id] = result
        return result


class GraphVisualizer:
    def __init__(self, relation_graph: RelationGraph):
        self.relation_graph = relation_graph
        self.graph: nx.MultiDiGraph = relation_graph.graph
        self.edge_color_map: Dict[str, str] = {
            "temporal": "grey",
            "action": "blue",
            "belonging": "purple",
            "association": "orange",
        }

    def _get_node_colors(self) -> Dict[Tuple[int, int], str]:
        colors: Dict[Tuple[int, int], str] = {}
        for node, data in self.graph.nodes(data=True):
            sentiment = float(data.get("compound_sentiment", data.get("init_sentiment", 0.0)))
            if sentiment > 0.1:
                colors[node] = "green"
            elif sentiment < -0.1:
                colors[node] = "red"
            else:
                colors[node] = "lightgray"
        return colors

    def draw_graph(self, save_path: Optional[str] = None) -> None:
        logger.info("Drawing graph...")
        if not self.graph.nodes:
            logger.info("Graph is empty. Nothing to draw.")
            return
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            logger.warning("Plotly not available; cannot draw graph: %s", exc)
            return
        pos_2d = nx.spring_layout(self.graph, k=2.0, iterations=100, seed=42)
        pos_3d: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
        for node, data in self.graph.nodes(data=True):
            x, y = pos_2d[node]
            z = int(data.get("clause_layer", 0))
            pos_3d[node] = (x, y, z)
        edge_traces = []
        edges_by_relation: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {rel: [] for rel in self.edge_color_map}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if rel in edges_by_relation:
                edges_by_relation[rel].append((u, v))
        for relation, edges in edges_by_relation.items():
            if not edges:
                continue
            edge_x, edge_y, edge_z = [], [], []
            for u, v in edges:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
            trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                line=dict(width=3, color=self.edge_color_map[relation]),
                hoverinfo="none",
                mode="lines",
                name=f"{relation.capitalize()} Relation",
            )
            edge_traces.append(trace)
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_colors = []
        node_colors_dict = self._get_node_colors()
        for node, data in sorted(self.graph.nodes(data=True)):
            x, y, z = pos_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(node_colors_dict[node])
            sentiment = float(data.get("compound_sentiment", data.get("init_sentiment", 0.0)))
            sentiment_str = f"{sentiment:.2f}"
            if "compound_sentiment" in data:
                sentiment_str += f" (compounded from {float(data.get('init_sentiment', 0.0)):.2f})"
            label = f"{data.get('head','')} ({node[0]}_{node[1]})"
            hover_info = (
                f"<b>{label}</b><br>"
                f"Role: {data.get('entity_role','')}<br>"
                f"Layer: {data.get('clause_layer',0)}<br>"
                f"Sentiment: {sentiment_str}"
            )
            node_text.append(hover_info)
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(color=node_colors, size=15, line=dict(width=1, color="black")),
            name="Entities",
        )
        fig = go.Figure(data=edge_traces + [node_trace])
        max_layer = max(z for _, (_, _, z) in pos_3d.items())
        fig.update_layout(
            title="3D Relation Graph (Warning: Not using the real sentiment formulas)",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(title="Clause Layer", nticks=int(max_layer) + 1),
            ),
        )
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive graph saved to {save_path}")
        else:
            fig.show()
