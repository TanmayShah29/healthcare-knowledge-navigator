"""Gradio web UI for the Healthcare Knowledge Navigator.

Three tabs:
  1. Ingest — upload a .txt or .pdf file, select doc_type, optionally set a
     publication date for recency weighting, and run the ingestion pipeline.
  2. Query — ask a clinical question and see vector hits, linked entities,
     graph facts, dosages, contraindications, the cited answer, and confidence.
  3. Stats — quick entity/relationship/chunk/dosage/contraindication counts.

Launch via: python -m src.main web  (or import and call launch_web()).
"""
import gradio as gr
from src.graph_db import check_connection, graph_stats, close
from src.ingest import ingest_file
from src.retrieval import retrieve
from src.agents.synthesizer import synthesize_answer
from src.confidence import score_confidence

VALID_DOC_TYPES = ["clinical_guideline", "research_paper", "treatment_protocol"]


def _ingest_handler(file, doc_type, publication_date):
    """Handle file ingestion from the web UI."""
    if file is None:
        return "Please upload a file."

    try:
        result = ingest_file(
            file.name,
            doc_type=doc_type,
            publication_date=publication_date.strip(),
        )
    except Exception as e:
        return f"Ingestion failed: {e}"

    lines = [
        f"**Ingested:** `{result.source_doc}` as `{result.doc_type}`",
        f"**Chunks processed:** {result.chunks_processed}",
        f"**Triples extracted:** {len(result.triples_extracted)} (loaded: {result.triples_loaded})",
        f"**Dosage facts:** {len(result.dosages_extracted)} (loaded: {result.dosages_loaded})",
        f"**Contraindications:** {len(result.contraindications_extracted)} (loaded: {result.contraindications_loaded})",
    ]
    if result.errors:
        lines.append("\n**Errors:**")
        for err in result.errors:
            lines.append(f"- {err}")

    stats = graph_stats()
    lines.append(
        f"\n**Graph now:** {stats['node_count']} entities, "
        f"{stats['rel_count']} relationships, {stats['chunk_count']} chunks, "
        f"{stats.get('dosage_count', 0)} dosages, "
        f"{stats.get('contra_count', 0)} contraindications."
    )
    return "\n".join(lines)


def _query_handler(question):
    """Handle a clinical question from the web UI."""
    if not question.strip():
        return "Please enter a question.", "", "", "", ""

    try:
        result = retrieve(question)
    except Exception as e:
        return f"Retrieval failed: {e}", "", "", "", ""

    # Format vector hits
    vector_lines = [f"### Vector-Matched Passages ({len(result.vector_hits)})"]
    if result.vector_hits:
        for hit in result.vector_hits:
            vector_lines.append(f"- **[{hit.score:.2f}]** {hit.citation()}")
    else:
        vector_lines.append("*(none)*")
    vector_md = "\n".join(vector_lines)

    # Format graph facts
    graph_lines = [
        f"### Linked Entities: {', '.join(result.linked_entities) or '*(none)*'}",
        f"### Graph Facts ({len(result.subgraph)})",
    ]
    for triple in result.subgraph:
        graph_lines.append(f"- {triple.as_fact_string()} *[{triple.citation()}]*")
    if not result.subgraph:
        graph_lines.append("*(none)*")

    if result.dosages:
        graph_lines.append(f"\n### Dosage Facts ({len(result.dosages)})")
        for d in result.dosages:
            graph_lines.append(f"- {d.as_fact_string()} *[{d.citation()}]*")

    if result.contraindications:
        graph_lines.append(f"\n### Contraindications ({len(result.contraindications)})")
        for c in result.contraindications:
            graph_lines.append(f"- {c.as_fact_string()} *[{c.citation()}]*")
    graph_md = "\n".join(graph_lines)

    # Synthesize answer
    try:
        answer, grounded = synthesize_answer(
            question, result.subgraph, result.vector_hits,
            result.dosages, result.contraindications,
        )
    except Exception as e:
        return vector_md, graph_md, f"Synthesis failed: {e}", "Low (0.0)", ""

    score, label = score_confidence(result.subgraph, result.vector_hits, grounded)
    confidence_str = f"**{label}** ({score})"

    return vector_md, graph_md, answer, confidence_str, ""


def _stats_handler():
    """Return current graph statistics."""
    try:
        stats = graph_stats()
    except Exception as e:
        return f"Failed to get stats: {e}"

    return (
        f"| Metric | Count |\n"
        f"|--------|-------|\n"
        f"| Entities | {stats['node_count']} |\n"
        f"| Relationships | {stats['rel_count']} |\n"
        f"| Indexed Chunks | {stats['chunk_count']} |\n"
        f"| Dosage Facts | {stats.get('dosage_count', 0)} |\n"
        f"| Contraindications | {stats.get('contra_count', 0)} |\n"
    )


def launch_web(server_name: str = "127.0.0.1", server_port: int = 7860):
    """Build and launch the Gradio interface."""
    with gr.Blocks(
        title="Healthcare Knowledge Navigator",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Healthcare Knowledge Navigator")
        gr.Markdown(
            "Hybrid (graph + vector) medical RAG assistant. "
            "Ingest clinical documents, ask questions, get cited answers with confidence scores."
        )

        with gr.Tabs():
            # ── Ingest tab ──────────────────────────────────────────────
            with gr.Tab("Ingest"):
                gr.Markdown("Upload a clinical document (.txt or .pdf) to ingest into the knowledge graph.")
                with gr.Row():
                    ingest_file_input = gr.File(
                        label="Document",
                        file_types=[".txt", ".pdf"],
                        type="filepath",
                    )
                    with gr.Column():
                        ingest_doc_type = gr.Dropdown(
                            choices=VALID_DOC_TYPES,
                            value="clinical_guideline",
                            label="Document Type",
                        )
                        ingest_date = gr.Textbox(
                            label="Publication Date (optional)",
                            placeholder="YYYY-MM-DD, YYYY-MM, or YYYY",
                        )
                ingest_btn = gr.Button("Ingest", variant="primary")
                ingest_output = gr.Markdown(label="Result")
                ingest_btn.click(
                    fn=_ingest_handler,
                    inputs=[ingest_file_input, ingest_doc_type, ingest_date],
                    outputs=ingest_output,
                )

            # ── Query tab ───────────────────────────────────────────────
            with gr.Tab("Query"):
                gr.Markdown("Ask a clinical question. The system runs hybrid retrieval (vector + graph) and synthesizes a cited answer.")
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What second-line diabetes drug should I consider for a patient with heart failure?",
                    lines=2,
                )
                query_btn = gr.Button("Ask", variant="primary")
                with gr.Row():
                    with gr.Column():
                        query_vector = gr.Markdown(label="Vector Hits")
                    with gr.Column():
                        query_graph = gr.Markdown(label="Graph Evidence")
                query_confidence = gr.Markdown(label="Confidence")
                query_answer = gr.Markdown(label="Answer")
                query_btn.click(
                    fn=_query_handler,
                    inputs=query_input,
                    outputs=[query_vector, query_graph, query_answer, query_confidence, gr.Markdown()],
                )

            # ── Stats tab ───────────────────────────────────────────────
            with gr.Tab("Stats"):
                gr.Markdown("Current state of the knowledge graph.")
                stats_btn = gr.Button("Refresh")
                stats_output = gr.Markdown()
                stats_btn.click(fn=_stats_handler, outputs=stats_output)

    app.launch(server_name=server_name, server_port=server_port)
