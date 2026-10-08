"""Graph-query tools for flowchart QA via tool calling.

Build a directed graph from arrow_triplets.json and expose deterministic
queries the model can call from a Qwen3-style tool-call loop.
"""

import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import Levenshtein


_PUNC = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower().replace("_", " ")
    s = _PUNC.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


class FlowGraph:
    """Directed multigraph built from extracted (source, end, condition) triplets.

    Node identity is the normalized lowercase form so light formatting
    differences don't fragment nodes. We keep the original display text for
    the first occurrence to render in tool replies.
    """

    def __init__(self, triplets: List[Dict], fuzzy: float = 0.85):
        self.fuzzy = fuzzy
        self._display: Dict[str, str] = {}
        self.edges: List[Tuple[str, str, Optional[str]]] = []
        for t in triplets or []:
            s_orig = (t.get("source") or "").strip()
            d_orig = (t.get("end") or "").strip()
            cond = t.get("condition")
            if isinstance(cond, str):
                cond = cond.strip() or None
            elif cond is not None:
                cond = str(cond).strip() or None
            if not s_orig or not d_orig or d_orig.lower() == "pending":
                continue
            s, d = normalize(s_orig), normalize(d_orig)
            if not s or not d:
                continue
            self._display.setdefault(s, s_orig)
            self._display.setdefault(d, d_orig)
            self.edges.append((s, d, cond))
        self.nodes: List[str] = list(self._display.keys())

    def display(self, n_norm: Optional[str]) -> Optional[str]:
        if n_norm is None:
            return None
        return self._display.get(n_norm, n_norm)

    def _resolve(self, query: str) -> Optional[str]:
        if query is None:
            return None
        q = normalize(query)
        if not q:
            return None
        if q in self._display:
            return q
        best, best_sim = None, 0.0
        for n in self.nodes:
            sim = edit_sim(n, q)
            if sim > best_sim:
                best_sim, best = sim, n
        return best if best_sim >= self.fuzzy else None

    # ---------- tool implementations ----------
    def count_nodes(self) -> Dict:
        return {"count": len(self.nodes)}

    def count_edges(self) -> Dict:
        return {"count": len(self.edges)}

    def list_nodes(self) -> Dict:
        return {"nodes": [self.display(n) for n in self.nodes]}

    def list_edges(self) -> Dict:
        return {"edges": [
            {"source": self.display(s), "end": self.display(d),
             "condition": c if c else "connectedTo"}
            for s, d, c in self.edges
        ]}

    def find_node(self, query: str, top_k: int = 5) -> Dict:
        if not query or not self.nodes:
            return {"matches": []}
        q = normalize(query)
        scored = [(edit_sim(n, q), n) for n in self.nodes]
        scored.sort(reverse=True)
        return {"matches": [
            {"node": self.display(n), "similarity": round(s, 3)}
            for s, n in scored[:top_k]
        ]}

    def get_outgoing(self, node: str) -> Dict:
        n = self._resolve(node)
        if n is None:
            return {"node": node, "matched": None, "outgoing": []}
        out = [
            {"end": self.display(d), "condition": c if c else "connectedTo"}
            for s, d, c in self.edges if s == n
        ]
        return {"node": node, "matched": self.display(n), "outgoing": out}

    def get_incoming(self, node: str) -> Dict:
        n = self._resolve(node)
        if n is None:
            return {"node": node, "matched": None, "incoming": []}
        inc = [
            {"source": self.display(s), "condition": c if c else "connectedTo"}
            for s, d, c in self.edges if d == n
        ]
        return {"node": node, "matched": self.display(n), "incoming": inc}

    def get_neighbors(self, node: str) -> Dict:
        n = self._resolve(node)
        if n is None:
            return {"node": node, "matched": None, "neighbors": []}
        seen, nbrs = set(), []
        for s, d, _ in self.edges:
            if s == n and d not in seen:
                seen.add(d); nbrs.append(self.display(d))
            if d == n and s not in seen:
                seen.add(s); nbrs.append(self.display(s))
        return {"node": node, "matched": self.display(n), "neighbors": nbrs}

    def has_edge(self, source: str, end: str, directed: bool = True) -> Dict:
        s = self._resolve(source)
        d = self._resolve(end)
        if s is None or d is None:
            return {
                "exists": False,
                "matched_source": self.display(s),
                "matched_end": self.display(d),
                "directed": directed,
            }
        if directed:
            ok = any(es == s and ed == d for es, ed, _ in self.edges)
        else:
            ok = any(
                (es == s and ed == d) or (es == d and ed == s)
                for es, ed, _ in self.edges
            )
        return {
            "exists": ok,
            "matched_source": self.display(s),
            "matched_end": self.display(d),
            "directed": directed,
        }

    def find_path(self, source: str, end: str, directed: bool = True) -> Dict:
        s = self._resolve(source)
        d = self._resolve(end)
        if s is None or d is None:
            return {
                "found": False,
                "path": None,
                "matched_source": self.display(s),
                "matched_end": self.display(d),
            }
        adj: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
        for es, ed, c in self.edges:
            adj[es].append((ed, c))
            if not directed:
                adj[ed].append((es, c))
        prev: Dict[str, Optional[str]] = {s: None}
        prev_cond: Dict[str, Optional[str]] = {s: None}
        q: deque = deque([s])
        while q:
            x = q.popleft()
            if x == d:
                path = []
                cur: Optional[str] = d
                while cur is not None:
                    path.append({
                        "node": self.display(cur),
                        "edge_condition_in": prev_cond.get(cur)
                            if cur != s else None,
                    })
                    cur = prev[cur]
                path.reverse()
                return {"found": True, "path": path, "length": len(path) - 1,
                        "directed": directed}
            for y, c in adj[x]:
                if y not in prev:
                    prev[y] = x
                    prev_cond[y] = c
                    q.append(y)
        return {"found": False, "path": None, "directed": directed}


# ---------- OpenAI-format tool schemas (consumed by Qwen chat template) ----------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "count_nodes",
            "description": "Total number of distinct nodes (text boxes) in the flowchart graph.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_edges",
            "description": "Total number of arrows (directed edges) in the flowchart graph.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_nodes",
            "description": "Return every node text in the flowchart graph.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_edges",
            "description": "Return every directed edge as {source, end, condition}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_node",
            "description": (
                "Fuzzy-search the graph for nodes whose text matches the query. "
                "Use this to locate the canonical node label before calling other "
                "tools when the question's wording differs from the node text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text query."},
                    "top_k": {"type": "integer", "description": "Max matches.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_outgoing",
            "description": "All outgoing edges (next nodes + edge condition) from a node.",
            "parameters": {
                "type": "object",
                "properties": {"node": {"type": "string"}},
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_incoming",
            "description": "All incoming edges (previous nodes + edge condition) into a node.",
            "parameters": {
                "type": "object",
                "properties": {"node": {"type": "string"}},
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "Undirected neighbors of a node (union of outgoing and incoming).",
            "parameters": {
                "type": "object",
                "properties": {"node": {"type": "string"}},
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "has_edge",
            "description": (
                "Whether an edge from `source` to `end` exists. "
                "Set directed=false to allow either direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "end": {"type": "string"},
                    "directed": {"type": "boolean", "default": True},
                },
                "required": ["source", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_path",
            "description": (
                "A shortest path from `source` to `end`. "
                "Set directed=false to ignore arrow direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "end": {"type": "string"},
                    "directed": {"type": "boolean", "default": True},
                },
                "required": ["source", "end"],
            },
        },
    },
]


_DISPATCH = {
    "count_nodes": lambda g, a: g.count_nodes(),
    "count_edges": lambda g, a: g.count_edges(),
    "list_nodes":  lambda g, a: g.list_nodes(),
    "list_edges":  lambda g, a: g.list_edges(),
    "find_node":   lambda g, a: g.find_node(a.get("query", ""), int(a.get("top_k", 5))),
    "get_outgoing":  lambda g, a: g.get_outgoing(a.get("node", "")),
    "get_incoming":  lambda g, a: g.get_incoming(a.get("node", "")),
    "get_neighbors": lambda g, a: g.get_neighbors(a.get("node", "")),
    "has_edge":  lambda g, a: g.has_edge(a.get("source", ""), a.get("end", ""),
                                         bool(a.get("directed", True))),
    "find_path": lambda g, a: g.find_path(a.get("source", ""), a.get("end", ""),
                                          bool(a.get("directed", True))),
}


def dispatch_tool(graph: FlowGraph, name: str, args: Dict) -> Dict:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    return fn(graph, args or {})
