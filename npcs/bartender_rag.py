import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]


def load_rules(rules_path: str) -> Dict[str, Any]:
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks_from_rules(rules: Dict[str, Any]) -> List[Chunk]:
    chunks: List[Chunk] = []

    npc_name = rules.get("npc_name", "Bartender")

    # Persona chunks
    persona = rules.get("persona", {})
    if persona:
        chunks.append(
            Chunk(
                id="persona.backstory",
                text=f"NPC {npc_name} backstory: {persona.get('backstory', '')}",
                meta={"section": "persona", "field": "backstory", "npc_name": npc_name},
            )
        )
        # traits and goals as concise lists
        if persona.get("traits"):
            chunks.append(
                Chunk(
                    id="persona.traits",
                    text=f"NPC {npc_name} traits: " + ", ".join(persona["traits"]),
                    meta={"section": "persona", "field": "traits", "npc_name": npc_name},
                )
            )
        if persona.get("goals"):
            chunks.append(
                Chunk(
                    id="persona.goals",
                    text=f"NPC {npc_name} goals: " + ", ".join(persona["goals"]),
                    meta={"section": "persona", "field": "goals", "npc_name": npc_name},
                )
            )
        if persona.get("dialogue_style"):
            style_bits = []
            for k, v in persona["dialogue_style"].items():
                style_bits.append(f"{k}: {v}")
            chunks.append(
                Chunk(
                    id="persona.dialogue_style",
                    text=f"NPC {npc_name} dialogue style: " + "; ".join(style_bits),
                    meta={"section": "persona", "field": "dialogue_style", "npc_name": npc_name},
                )
            )

    # Safety chunks
    safety = rules.get("safety", {})
    if safety:
        if safety.get("refuse"):
            for i, item in enumerate(safety["refuse"], start=1):
                chunks.append(
                    Chunk(
                        id=f"safety.refuse.{i}",
                        text=f"Refuse policy {i}: {item}",
                        meta={"section": "safety", "field": "refuse", "index": i, "npc_name": npc_name},
                    )
                )
        if safety.get("deescalation"):
            for i, item in enumerate(safety["deescalation"], start=1):
                chunks.append(
                    Chunk(
                        id=f"safety.deescalation.{i}",
                        text=f"De-escalation tip {i}: {item}",
                        meta={"section": "safety", "field": "deescalation", "index": i, "npc_name": npc_name},
                    )
                )

    # Operational rules (one per rule)
    for rule in rules.get("operational_rules", []):
        rid = rule.get("id") or f"rule.{len(chunks)}"
        rtext = rule.get("text", "")
        tags = rule.get("tags", [])
        text = f"Rule {rid}: {rtext} (tags: {', '.join(tags)})"
        chunks.append(
            Chunk(
                id=f"rules.{rid}",
                text=text,
                meta={"section": "rules", "rule_id": rid, "tags": tags, "npc_name": npc_name},
            )
        )

    return chunks


def embed(text: str, ollama_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    url = f"{ollama_url}/api/embeddings"
    payload = {"model": model, "prompt": text}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Invalid embedding response from Ollama")
    return vec


def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        # try to handle different dims by trimming/padding (not ideal, but practical)
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_index(chunks: List[Chunk], ollama_url: str, embed_model: str) -> Dict[str, Any]:
    index_items = []
    for ch in chunks:
        vec = embed(ch.text, ollama_url=ollama_url, model=embed_model)
        index_items.append({
            "id": ch.id,
            "text": ch.text,
            "meta": ch.meta,
            "embedding": vec,
        })
    return {
        "ollama_url": ollama_url,
        "embed_model": embed_model,
        "items": index_items,
    }


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_index(index: Dict[str, Any], query: str, ollama_url: str, embed_model: str, top_k: int = 5) -> List[Dict[str, Any]]:
    qvec = embed(query, ollama_url=ollama_url, model=embed_model)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in index.get("items", []):
        sim = cosine_similarity(qvec, item["embedding"])  # type: ignore
        scored.append((sim, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(item, score=float(score)) for score, item in scored[:top_k]]


def compose_prompt(npc_name: str, query: str, hits: List[Dict[str, Any]]) -> str:
    rules_block = "\n".join([f"- {h['text']}" for h in hits])
    prompt = (
        f"You are roleplaying NPC {npc_name}, a bartender. Use the following retrieved rules and persona snippets to guide your response.\n"
        f"Rules and persona context (top-{len(hits)}):\n{rules_block}\n\n"
        f"Player input: {query}\n\n"
        f"Respond in JSON with fields: dialogue, actions, emotion, decision. Keep it concise and in-character."
    )
    return prompt


def cmd_build(args: argparse.Namespace) -> None:
    rules = load_rules(args.rules)
    chunks = build_chunks_from_rules(rules)
    index = build_index(chunks, ollama_url=args.ollama_url, embed_model=args.embed_model)
    save_json(args.out, index)
    print(f"Built index with {len(index['items'])} items → {args.out}")


def cmd_query(args: argparse.Namespace) -> None:
    index = load_json(args.index)
    ollama_url = args.ollama_url or index.get("ollama_url", DEFAULT_OLLAMA_URL)
    embed_model = args.embed_model or index.get("embed_model", DEFAULT_EMBED_MODEL)

    hits = search_index(index, args.query, ollama_url=ollama_url, embed_model=embed_model, top_k=args.top_k)
    npc_name = (index.get("items", [{}])[0].get("meta", {}).get("npc_name") or "Bartender") if index.get("items") else "Bartender"

    if args.json:
        out = {
            "query": args.query,
            "top_k": args.top_k,
            "results": hits,
            "composed_prompt": compose_prompt(npc_name, args.query, hits),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("Top results:")
        for i, h in enumerate(hits, start=1):
            print(f"{i}. score={h['score']:.3f} | {h['id']} :: {h['text']}")
        print()
        print("Composed prompt:")
        print(compose_prompt(npc_name, args.query, hits))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple RAG builder/query for bartender NPC rules using Ollama embeddings")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build an embeddings index from rules JSON")
    pb.add_argument("--rules", default=os.path.join(os.path.dirname(__file__), "bartender_rules.json"), help="Path to rules JSON")
    pb.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    pb.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name (Ollama)")
    pb.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "bartender_rules_index.json"), help="Output index JSON path")
    pb.set_defaults(func=cmd_build)

    pq = sub.add_parser("query", help="Query the index and produce a composed prompt")
    pq.add_argument("query", help="User/player query to retrieve relevant rules")
    pq.add_argument("--index", default=os.path.join(os.path.dirname(__file__), "bartender_rules_index.json"), help="Path to the built index JSON")
    pq.add_argument("--ollama-url", default=None, help="Override Ollama base URL (defaults to index or env)")
    pq.add_argument("--embed-model", default=None, help="Override embedding model (defaults to index or env)")
    pq.add_argument("--top-k", type=int, default=6, help="Number of results to retrieve")
    pq.add_argument("--json", action="store_true", help="Print JSON output including composed prompt")
    pq.set_defaults(func=cmd_query)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
