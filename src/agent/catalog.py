from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodCard:
    method_id: str
    content: str
    path: str


class MethodCatalog:
    def __init__(self, cards: dict[str, MethodCard]):
        self.cards = cards

    @classmethod
    def load(cls, directory: Path) -> "MethodCatalog":
        cards = {
            path.stem: MethodCard(path.stem, path.read_text(encoding="utf-8"), str(path))
            for path in sorted(directory.glob("*.md"))
        }
        required = {"bpr", "group_softmax"}
        if not required.issubset(cards):
            raise ValueError(f"Method catalog must contain {sorted(required)}; found {sorted(cards)}")
        return cls(cards)

    def prompt_text(self, family: str | None = None) -> str:
        selected = self.cards.values() if family is None else [self.cards[family]]
        return "\n\n".join(f"METHOD CARD {card.method_id}\n{card.content}" for card in selected)

