"""Generates Hungarian retrieval questions from Confluence documents using Gemini."""

import json
import re
from pathlib import Path
from typing import List, Optional, Set

import tiktoken
from openai import OpenAI
from rich.console import Console
from rich.progress import track

from src.config.settings import OpenRouterSettings
from src.models.schemas import GoldenQADataset, GoldenQAPair

console = Console()

# Token thresholds that determine how many questions to generate per document
_THRESHOLDS = [
    (150, 1),
    (600, 3),
    (1500, 5),
    (3000, 8),
]
_MAX_QUESTIONS = 8

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _question_count(token_count: int) -> int:
    """Map token count to number of questions to generate."""
    for threshold, count in _THRESHOLDS:
        if token_count < threshold:
            return count
    return _MAX_QUESTIONS


def _build_answer_prompt(question: str, document_text: str, title: str) -> str:
    return f"""Te egy vállalati tudásbázis alapján válaszoló asszisztens vagy.

Egy konkrét kérdésre kell válaszolnod, kizárólag az alábbi dokumentum tartalma alapján.

### Dokumentum adatai:
- **Cím:** {title}
- **Tartalom:** {document_text}

### Kérdés:
{question}

### Követelmények:
- Válaszolj KIZÁRÓLAG magyar nyelven
- Csak és kizárólag a fenti dokumentumban szereplő információ alapján válaszolj
- Ha a dokumentum nem tartalmaz elegendő információt, válaszolj: "A dokumentum nem tartalmaz elegendő információt a kérdés megválaszolásához."
- Legyen tömör, pontos és egyértelmű a válasz
- Ne hivatkozz magára a dokumentumra ("a dokumentum szerint..."), csak add meg a választ

Válasz:"""


def _build_prompt(document_text: str, num_questions: int, title: str) -> str:
    return f"""Te egy tapasztalt minőségbiztosítási szakember vagy, aki egy vállalati chatbot éles tesztelését végzi.

A feladatod, hogy a megadott dokumentumrészlet alapján generálj {num_questions} darab kérdés-válasz párt. Minden párhoz írj egy életszerű magyar kérdést ÉS egy pontos, dokumentum alapú választ.

### A dokumentum adatai:
- **Cím:** {title}
- **Tartalom:** {document_text}

### Irányelvek a kérdésekhez:
1. **Kerüld a "pinpoint" stílust:** Ne úgy kérdezz, mint egy vizsgáztató (pl. "Melyik évben alakult a cég?"). Ehelyett fogalmazz életszerűen (pl. "Régóta piacon vannak már, vagy új szereplők?").
2. **Használj változatos szándékokat:** Ne csak kérdő mondatokat írj! Legyenek benne kérések, problémaleírások vagy félinformációkon alapuló érdeklődések (pl. "Segíts már, nem találom, hogyan kell...").
3. **Nyelvi stílus:** Használj természetes, beszélt nyelvi fordulatokat. A kérdések legyenek udvariasak, de ne túl sterilek.
4. **Fókusz:** A kérdés vonatkozzon a dokumentumban lévő információra, de a megfogalmazás tükrözze azt a bizonytalanságot, amivel egy valódi felhasználó érkezik.
5. **Egyediség:** Mindegyik kérdésnek egyedinek kell lennie, különböző témakörökre vonatkozzon.

### Irányelvek a válaszokhoz:
1. **Csak a dokumentum alapján:** Kizárólag a fenti dokumentum tartalmára támaszkodj, ne használj külső tudást.
2. **Tömör és pontos:** A válasz legyen egyértelmű és informatív, de ne legyen feleslegesen hosszú.
3. **Magyar nyelv:** A válasz is kizárólag magyar nyelvű legyen.
4. **Ne hivatkozz a dokumentumra:** Ne kezdd azzal, hogy "a dokumentum szerint" – csak add meg a választ közvetlenül.

### Tiltások:
- TILOS olyan kérdés, amihez a válasz nincs benne a szövegben.
- TILOS a "Miről szól ez az oldal?" vagy hasonlóan általános meta-kérdés.
- TILOS olyan kérdés, ami teljesen ugyanahhoz a témakörhöz tartozik mint egy előző kérdés.

Válaszolj KIZÁRÓLAG az alábbi JSON formátumban:
{{"pairs": [{{"question": "életszerű kérdés", "ground_truth": "pontos válasz a dokumentum alapján"}}]}}"""


class DatasetGenerator:
    """Generates Hungarian retrieval questions from ingested documents using Gemini."""

    def __init__(self, settings: OpenRouterSettings, data_dir: Path) -> None:
        self._client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
        )
        self._model_name = settings.model
        self._output_path = data_dir / "evaluation" / "golden_qa.json"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, force_regenerate: bool = False) -> GoldenQADataset:
        """Generate questions for all documents, skipping already-processed ones.

        Args:
            force_regenerate: Ignore existing dataset and regenerate from scratch.

        Returns:
            The full dataset written to data/evaluation/golden_qa.json.
        """
        dataset = self._load_existing(force_regenerate)
        already_processed: Set[Optional[str]] = {p.source_page_id for p in dataset.pairs}

        md_files = self._discover_documents()
        if not md_files:
            console.print(
                "[red]Dokumentumok nem találhatók. Futtasd előbb az 'ingest' parancsot.[/red]"
            )
            return dataset

        console.print(f"[bold]Összesen {len(md_files)} dokumentum található.[/bold]")

        for md_path in track(md_files, description="Kérdések generálása..."):
            page_id = md_path.stem
            if page_id in already_processed:
                continue

            text = md_path.read_text(encoding="utf-8")
            title = self._extract_title(text, md_path.stem)
            token_count = _count_tokens(text)
            num_questions = _question_count(token_count)

            if num_questions == 0:
                console.print(f"  [dim]Kihagyva (kevés tartalom): {title}[/dim]")
                continue

            pairs = self._generate_pairs(text, title, num_questions)
            if not pairs:
                console.print(f"  [yellow]Nem sikerült kérdés-válasz párokat generálni: {title}[/yellow]")
                continue

            for pair in pairs:
                pair.source_page_id = page_id
                pair.source_title = title
                dataset.pairs.append(pair)

            console.print(f"  [green]✓[/green] {title} → {len(pairs)} kérdés-válasz pár")
            self._save(dataset)

        console.print(
            f"\n[bold green]Kész! {len(dataset.pairs)} kérdés mentve: {self._output_path}[/bold green]"
        )
        return dataset

    def generate_answers(self) -> GoldenQADataset:
        """Fill in ground_truth for all pairs that don't have one yet.

        Reads each pair's source document and asks Gemini to answer
        the question based solely on that document. Saves incrementally.

        Returns:
            The updated dataset with ground_truth populated.
        """
        if not self._output_path.exists():
            console.print("[red]Nincs meglévő kérdésadathalmaz. Futtasd előbb a generate-qa parancsot.[/red]")
            return GoldenQADataset()

        dataset = GoldenQADataset.model_validate_json(
            self._output_path.read_text(encoding="utf-8")
        )

        pending = [p for p in dataset.pairs if not p.ground_truth]
        if not pending:
            console.print("[cyan]Minden kérdésnek már van válasza.[/cyan]")
            return dataset

        console.print(f"[bold]{len(pending)} kérdéshez kell választ generálni.[/bold]")
        doc_cache: dict[str, str] = {}

        for pair in track(pending, description="Válaszok generálása..."):
            if not pair.source_page_id:
                continue

            if pair.source_page_id not in doc_cache:
                doc_path = self._find_document(pair.source_page_id)
                if not doc_path:
                    console.print(f"  [yellow]Forrás nem található: {pair.source_page_id}[/yellow]")
                    continue
                doc_cache[pair.source_page_id] = doc_path.read_text(encoding="utf-8")

            doc_text = doc_cache[pair.source_page_id]
            title = pair.source_title or self._extract_title(doc_text, pair.source_page_id)
            answer = self._generate_answer(pair.question, doc_text, title)

            if answer:
                pair.ground_truth = answer
                self._save(dataset)

        answered = sum(1 for p in dataset.pairs if p.ground_truth)
        console.print(
            f"\n[bold green]Kész! {answered}/{len(dataset.pairs)} kérdésnek van válasza.[/bold green]"
        )
        return dataset

    def _generate_answer(self, question: str, document_text: str, title: str) -> Optional[str]:
        """Call Gemini to generate a ground truth answer for one question."""
        prompt = _build_answer_prompt(question, document_text, title)
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception as e:
            console.print(f"  [red]API hiba: {e}[/red]")
            return None

    def _find_document(self, page_id: str) -> Optional[Path]:
        """Locate the markdown file for a given page_id (filename stem)."""
        confluence_dir = self._output_path.parent.parent / "confluence"
        matches = list(confluence_dir.rglob(f"{page_id}.md"))
        return matches[0] if matches else None

    def _generate_pairs(
        self, document_text: str, title: str, num_questions: int
    ) -> List[GoldenQAPair]:
        """Call Gemini and parse the returned Q&A pairs."""
        prompt = _build_prompt(document_text, num_questions, title)
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            return self._parse_pairs(raw)
        except Exception as e:
            console.print(f"  [red]API hiba: {e}[/red]")
            return []

    def _parse_pairs(self, raw: str) -> List[GoldenQAPair]:
        """Extract Q&A pairs from Gemini's JSON response."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            pairs = data.get("pairs", [])
            return [
                GoldenQAPair(
                    question=p["question"].strip(),
                    ground_truth=p["ground_truth"].strip(),
                )
                for p in pairs
                if isinstance(p, dict)
                and p.get("question", "").strip()
                and p.get("ground_truth", "").strip()
            ]
        except (json.JSONDecodeError, KeyError):
            console.print(f"  [red]Érvénytelen JSON válasz: {raw[:200]}[/red]")
            return []

    def _discover_documents(self) -> List[Path]:
        """Return all markdown files sorted by name for deterministic ordering."""
        confluence_dir = self._output_path.parent.parent / "confluence"
        if not confluence_dir.exists():
            return []
        return sorted(confluence_dir.rglob("*.md"))

    def _extract_title(self, text: str, fallback: str) -> str:
        """Extract the first H1 heading from markdown, fall back to filename."""
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else fallback

    def _load_existing(self, force_regenerate: bool) -> GoldenQADataset:
        """Load dataset from disk if it exists and regeneration is not forced."""
        if not force_regenerate and self._output_path.exists():
            raw = self._output_path.read_text(encoding="utf-8")
            dataset = GoldenQADataset.model_validate_json(raw)
            console.print(f"[cyan]Meglévő adathalmaz betöltve: {len(dataset.pairs)} kérdés.[/cyan]")
            return dataset
        return GoldenQADataset()

    def _save(self, dataset: GoldenQADataset) -> None:
        """Persist the dataset to disk after each document."""
        self._output_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
