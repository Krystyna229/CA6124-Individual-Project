#!/usr/bin/env python3
"""Build the four-page CA6124 report from recorded experiment artifacts."""

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"


def margins(section):
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(.8)
    section.bottom_margin = Inches(.8)
    section.left_margin = Inches(.75)
    section.right_margin = Inches(.75)


def shade(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    properties.append(element)


def add_table(doc, columns, rows, widths=None, font_size=11):
    table = doc.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    for index, title in enumerate(columns):
        cell = table.rows[0].cells[index]
        cell.text = title
        shade(cell, "D9EAF7")
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = str(value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[index])
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
                    if row is table.rows[0]:
                        run.bold = True
    return table


def heading(doc, text, level=1):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(text)
    return paragraph


def body(doc, text):
    paragraph = doc.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.0
    return paragraph


def page_break(doc):
    doc.add_page_break()


def main():
    OUT.mkdir(exist_ok=True)
    diagnostics = json.loads((ROOT / "diagnostics/dataset_diagnostics.json").read_text())
    results = pd.read_csv(ROOT / "results/final_summary.csv")
    params = pd.read_csv(ROOT / "results/best_parameters.csv").fillna("")

    doc = Document()
    margins(doc.sections[0])
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(11)
    for name, size, color in (("Heading 1", 13, "17365D"),
                              ("Heading 2", 11, "365F91")):
        styles[name].font.name = "Arial"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = __import__("docx").shared.RGBColor.from_string(color)
        styles[name].paragraph_format.space_before = Pt(4)
        styles[name].paragraph_format.space_after = Pt(2)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Extending DaisyRec 2.0 with CDAE")
    run.bold = True
    run.font.size = Pt(16)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Warm-start full-ranking comparison on MovieLens 1M and Amazon Digital Music").italic = True
    ident = doc.add_paragraph()
    ident.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ident.add_run("Name: Yanyu Wen     Student ID: [YOUR STUDENT ID]").bold = True

    heading(doc, "1 Introduction")
    body(doc, "This project integrates Collaborative Denoising Auto-Encoder (CDAE) [1], an established top-N method absent from the supplied DaisyRec 2.0 framework, and compares it with two existing baselines: the closed-form linear EASE model [2] and probabilistic Multi-VAE [3]. The research question is: under one reproducible warm-start full-ranking protocol, how does CDAE compare with a shallow item-co-occurrence model and a variational neural model on a dense movie dataset and a sparse e-commerce dataset?")
    body(doc, "MovieLens 1M and the 2014 Amazon Digital Music ratings-only release were selected to create a meaningful cross-domain stress test. Their differences jointly involve domain, scale, sparsity and temporal coverage; therefore observed performance differences are described as dataset-specific trends, not attributed causally to sparsity alone.")

    heading(doc, "1.1 Data diagnosis and fixed decision rule", 2)
    drows = []
    for label, key in (("ML-1M", "ml-1m"), ("Amazon Music", "amazon-music")):
        d = next(item for item in diagnostics[key]["variants"]
                 if item["preprocessing"] == "10filter")
        drows.append([label, f"{d['users']:,}", f"{d['items']:,}",
                      f"{d['interactions']:,}", f"{100*d['density']:.3f}%",
                      f"{d['actual_train_ratio']:.3f}",
                      f"{d['warm_test_interactions']:,}",
                      f"{d['warm_test_users']:,}"])
    add_table(doc, ["Dataset", "Users", "Items", "Events", "Density", "Train", "Warm test", "Test users"], drows,
              widths=[1.15,.55,.6,.8,.65,.55,.75,.65], font_size=10)
    body(doc, "Before seeing model results, the filter rule was fixed: use DaisyRec's one-pass 10filter unless fewer than 500 warm-start test users remain. Both datasets exceed the threshold, so 10filter is retained. Amazon 5filter was rejected because 20,356 items imply a 3.09 GiB float64 item matrix before EASE's additional inverse/storage costs. The 2014 Amazon file contains 836,006 headerless rows before filtering; its SHA-256 is documented in README.")
    body(doc, "Repository: https://github.com/Krystyna229/CA6124-Individual-Project")

    page_break(doc)
    heading(doc, "2 CDAE and non-destructive DaisyRec extension")
    body(doc, "For user u with binary history xᵤ, CDAE applies training-only dropout to obtain x̃ᵤ and computes hᵤ = ReLU(W₁x̃ᵤ + vᵤ + b₁), where vᵤ is a learned user embedding. Decoder logits are zᵤ = W₂hᵤ + b₂. Training minimizes summed binary cross-entropy plus optional L1/L2 penalties. Calling eval() disables corruption, so inference is deterministic. The implementation subclasses AERecommender and reuses DaisyRec history matrices, optimizer, device handling and training loop.")
    body(doc, "The extension adds CDAERecommender.py, cdae.yaml and a registry entry. Evaluation changes are opt-in configuration branches: split_boundary=day, warm_start=True, ranking_mode=full and optimization_k=10. Defaults preserve DaisyRec's original interaction-boundary, unfiltered and sampled-candidate behaviour. Existing EASE and Multi-VAE model code is not replaced. Six regression tests verify the unchanged default splitter, deterministic day boundaries, warm filtering, seen-item masking, CDAE evaluation determinism and YAML tuning parsing.")

    heading(doc, "3 Experimental protocol")
    body(doc, "Ratings are converted to implicit interactions and duplicate user-item pairs are removed. A deterministic global timestamp boundary between complete calendar days is chosen nearest 80/20; ties are resolved deterministically. All events on one day stay on one side, important because Amazon timestamps have day precision. The actual train ratios are reported in Table 1 rather than claimed to be exactly 0.8.")
    body(doc, "Final evaluation keeps only test interactions whose user and item both occur in the 80% training set. Tuning applies the same rule relative to each training/validation split, so its evaluable item set is intentionally based on the smaller tuning-training subset. Ranking considers every training-seen item and masks the user's training positives. The final cohort is 1,152 ML-1M users / 105,699 interactions and 1,294 Amazon users / 4,065 interactions.")
    body(doc, "Each of the six groups receives 20 Optuna/TPE trials with seed 2022, optimizing validation NDCG@10. The winning setting is retrained on the complete 80% training partition, and the untouched test partition is evaluated once at K ∈ {1,5,10,20,30,50}. This uniform budget was chosen before final testing and complete trial histories are retained.")

    heading(doc, "3.1 Search spaces", 2)
    add_table(doc, ["Model", "Search space"], [
        ["CDAE", "d {16,32,64,128,256}; drop {0,.1,.3,.5,.7}; batch {64,128,256,512}; lr {5e-5,1e-4,5e-4,.001,.005,.01}; L2 {0,1e-5,1e-4,.001,.01,.1,1}"],
        ["EASE", "regularization {1,5,10,50,100,200,500,1000,2000}"],
        ["Multi-VAE", "d {32,64,128,256,512}; drop {0,.2,.5,.7,.8}; batch {64,128,256,512}; lr {5e-5,1e-4,5e-4,.001,.005,.01}; anneal {.05,.1,.2,.5}"],
    ], widths=[1.05,5.85], font_size=10)

    page_break(doc)
    heading(doc, "4 Hyperparameter tuning")
    doc.add_picture(str(ROOT / "results/tuning_curves.png"), width=Inches(6.95))
    caption = doc.add_paragraph("Figure 1. Individual trial scores and best-so-far validation NDCG@10 (20 trials per group).")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(10)

    columns = [c for c in params.columns if c not in ("dataset", "model")]
    prows = []
    for _, row in params.iterrows():
        selected = ", ".join(f"{c.replace('latent_dim','d')}={row[c]}" for c in columns
                             if row[c] != "" and c != "ndcg")
        prows.append([row.dataset, row.model, selected, f"{float(row.ndcg):.4f}"])
    add_table(doc, ["Dataset", "Model", "Selected hyperparameters", "Val NDCG@10"], prows,
              widths=[1.0,.85,4.15,.9], font_size=10)
    body(doc, "The initial ranges were expanded before inspecting final-test results because several winners lay on boundaries. The reported second-stage study therefore includes lower learning rates, zero dropout, wider latent dimensions and broader regularization. Twenty trials cannot exhaust these mixed spaces; the curves make this optimization uncertainty visible instead of treating the selected point as a global optimum.")

    page_break(doc)
    heading(doc, "5 Final test results")
    rrows = []
    for _, row in results.iterrows():
        rrows.append([row.dataset, row.model, f"{row['recall@10']:.4f}", f"{row['ndcg@10']:.4f}"])
    add_table(doc, ["Dataset", "Model", "Recall@10", "NDCG@10"], rrows,
              widths=[1.4,1.3,1.1,1.1], font_size=10)
    doc.add_picture(str(ROOT / "results/final_results.png"), width=Inches(6.25))
    caption = doc.add_paragraph("Figure 2. Test performance under identical warm-start full-ranking evaluation.")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(10)

    heading(doc, "6 Analysis and limitations")
    analyses = []
    for dataset in ("ml-1m", "amazon-music"):
        subset = results[results.dataset == dataset].set_index("model")
        best = subset["ndcg@10"].idxmax()
        cdae = subset.loc["CDAE", "ndcg@10"]
        ease = subset.loc["EASE", "ndcg@10"]
        vae = subset.loc["Multi-VAE", "ndcg@10"]
        analyses.append(f"On {dataset}, {best} ranks first by NDCG@10. CDAE is {(cdae/ease-1)*100:+.1f}% relative to EASE and {(cdae/vae-1)*100:+.1f}% relative to Multi-VAE.")
    body(doc, " ".join(analyses) + " EASE tests whether direct item co-occurrence with strong shrinkage is sufficient; CDAE adds nonlinear denoising and user embeddings; Multi-VAE adds a distributional bottleneck. The contrast across datasets indicates that model ordering is context-dependent, but two datasets cannot isolate whether domain, scale, sparsity or temporal coverage produced the change.")
    body(doc, "The study is limited to one temporal split, one random seed, a warm-start cohort, and 20 tuning trials. Warm-start evaluation is appropriate for comparing these three models, which cannot score unseen items (and CDAE has no embedding for an unseen user), but it excludes deployment cold-start performance. Full ranking removes candidate-sampling variance, while the reduced Amazon cohort means its result should not be generalized to all original users. Repeated seeds and additional domains are the clearest follow-up.")

    heading(doc, "References", 2)
    refs = [
        "[1] Y. Wu et al., ‘Collaborative Denoising Auto-Encoders for Top-N Recommender Systems,’ WSDM, 2016.",
        "[2] H. Steck, ‘Embarrassingly Shallow Autoencoders for Sparse Data,’ WWW, 2019.",
        "[3] D. Liang et al., ‘Variational Autoencoders for Collaborative Filtering,’ WWW, 2018.",
        "[4] Z. Sun et al., ‘DaisyRec 2.0: Benchmarking Recommendation for Rigorous Evaluation,’ TPAMI, 2023.",
        "[5] F. Harper and J. Konstan, ‘The MovieLens Datasets: History and Context,’ TiiS, 2015.",
        "[6] J. McAuley et al., ‘Image-based Recommendations on Styles and Substitutes,’ SIGIR, 2015.",
    ]
    for reference in refs:
        paragraph = doc.add_paragraph(reference)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.left_indent = Inches(.15)
        paragraph.paragraph_format.first_line_indent = Inches(-.15)
        for run in paragraph.runs:
            run.font.size = Pt(10)

    for section in doc.sections:
        margins(section)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.add_run("CA6124 Individual Assignment")

    path = OUT / "CA6124_CDAE_Technical_Report.docx"
    doc.save(path)
    print(path)


if __name__ == "__main__":
    main()
