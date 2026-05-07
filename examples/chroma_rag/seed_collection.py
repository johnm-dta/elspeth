"""Seed a ChromaDB ephemeral collection with reference documents.

Creates a persistent ChromaDB collection at examples/chroma_rag/chroma_data/
with science/health documents that the RAG pipeline will query against.

Usage:
    python examples/chroma_rag/seed_collection.py
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import chromadb

PERSIST_DIR = str(Path(__file__).resolve().parent / "chroma_data")
COLLECTION_NAME = "science-facts"

DOCUMENTS = [
    {
        "id": "doc-diabetes-overview",
        "content": (
            "Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels. "
            "Type 1 diabetes is an autoimmune condition where the immune system attacks insulin-producing "
            "beta cells in the pancreas. Type 2 diabetes involves insulin resistance, where cells don't "
            "respond effectively to insulin. Common symptoms include excessive thirst, frequent urination, "
            "unexplained weight loss, fatigue, and blurred vision."
        ),
        "metadata": {"topic": "medicine", "subtopic": "diabetes"},
    },
    {
        "id": "doc-diabetes-management",
        "content": (
            "Managing diabetes involves regular blood glucose monitoring, medication or insulin therapy, "
            "dietary management, and physical activity. HbA1c tests measure average blood sugar over "
            "2-3 months. Complications of unmanaged diabetes include neuropathy, retinopathy, kidney "
            "disease, and cardiovascular problems."
        ),
        "metadata": {"topic": "medicine", "subtopic": "diabetes"},
    },
    {
        "id": "doc-photosynthesis",
        "content": (
            "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
            "It occurs primarily in chloroplasts, using chlorophyll to absorb sunlight. The light-dependent "
            "reactions split water molecules, releasing oxygen and producing ATP and NADPH. The Calvin cycle "
            "then uses these products to fix carbon dioxide into glucose. The overall equation is: "
            "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2."
        ),
        "metadata": {"topic": "biology", "subtopic": "botany"},
    },
    {
        "id": "doc-earthquakes",
        "content": (
            "Earthquakes occur when tectonic plates along fault lines suddenly slip past each other, "
            "releasing stored elastic energy as seismic waves. The point of rupture underground is the "
            "focus (hypocenter), while the point directly above on the surface is the epicenter. "
            "Earthquakes are measured using the moment magnitude scale (Mw), which replaced the older "
            "Richter scale. Seismographs detect ground motion and record seismograms."
        ),
        "metadata": {"topic": "geology", "subtopic": "seismology"},
    },
    {
        "id": "doc-mitochondria",
        "content": (
            "Mitochondria are double-membrane organelles found in most eukaryotic cells. They are often "
            "called the 'powerhouse of the cell' because they generate most of the cell's supply of "
            "adenosine triphosphate (ATP) through oxidative phosphorylation. The inner membrane is folded "
            "into cristae, which increase surface area for the electron transport chain. Mitochondria "
            "have their own DNA and can replicate independently."
        ),
        "metadata": {"topic": "biology", "subtopic": "cell_biology"},
    },
    {
        "id": "doc-vaccines",
        "content": (
            "Vaccines work by training the immune system to recognize and fight specific pathogens. "
            "They contain weakened or inactivated forms of a pathogen, or parts of it (like proteins "
            "or mRNA instructions). When administered, the immune system produces antibodies and memory "
            "cells. If the person later encounters the real pathogen, the immune system can mount a "
            "rapid, targeted response, preventing or reducing the severity of disease."
        ),
        "metadata": {"topic": "medicine", "subtopic": "immunology"},
    },
    {
        "id": "doc-climate-oceans",
        "content": (
            "Climate change affects ocean levels through two primary mechanisms: thermal expansion of "
            "seawater as it warms, and the melting of land-based ice sheets and glaciers. Current "
            "projections estimate sea levels could rise 0.3 to 1.0 meters by 2100 under various "
            "emission scenarios. Rising seas threaten coastal communities, increase flooding, erode "
            "shorelines, and contaminate freshwater aquifers with saltwater."
        ),
        "metadata": {"topic": "earth_science", "subtopic": "climate"},
    },
    {
        "id": "doc-machine-learning",
        "content": (
            "Machine learning differs from traditional programming in a fundamental way. In traditional "
            "programming, developers write explicit rules that map inputs to outputs. In machine learning, "
            "algorithms learn patterns from data to make predictions or decisions. Models are trained on "
            "datasets, adjusting internal parameters to minimize error. Key approaches include supervised "
            "learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning "
            "(reward signals)."
        ),
        "metadata": {"topic": "computer_science", "subtopic": "ai"},
    },
    {
        "id": "doc-dna",
        "content": (
            "DNA (deoxyribonucleic acid) is a double-helix molecule composed of two polynucleotide chains. "
            "Each nucleotide consists of a phosphate group, a deoxyribose sugar, and one of four "
            "nitrogenous bases: adenine (A), thymine (T), guanine (G), or cytosine (C). Base pairing "
            "follows Chargaff's rules: A pairs with T, and G pairs with C. The sequence of bases encodes "
            "genetic information used to build proteins."
        ),
        "metadata": {"topic": "biology", "subtopic": "genetics"},
    },
    {
        "id": "doc-immune-system",
        "content": (
            "The human immune system has two main branches: innate immunity and adaptive immunity. "
            "Innate immunity provides immediate, non-specific defense through barriers (skin, mucous "
            "membranes), phagocytes, and inflammatory responses. Adaptive immunity develops over time "
            "and provides specific, long-lasting protection through T cells and B cells that recognize "
            "specific antigens. Memory cells enable faster responses to previously encountered pathogens."
        ),
        "metadata": {"topic": "medicine", "subtopic": "immunology"},
    },
]


def main() -> None:
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Delete collection if it exists (clean seed)
    with contextlib.suppress(ValueError, chromadb.errors.NotFoundError):
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["content"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS],
    )

    print(f"Seeded {collection.count()} documents into collection '{COLLECTION_NAME}'")  # noqa: T201
    print(f"Persisted to: {PERSIST_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
