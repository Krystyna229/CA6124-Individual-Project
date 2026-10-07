#!/usr/bin/env python3
"""Build the four-page CA6124 report from recorded experiment artifacts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.opc.constants import RELATIONSHIP_TYPE as RT


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


def add_hyperlink(paragraph, text, url):
    relationship = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Arial")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.extend([fonts, color, underline])
    run.append(properties)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def set_style_font(style, name, size=None, bold=None):
    style.font.name = name
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), name)


def make_ranking_curves():
    ks = [1, 5, 10, 20, 30, 50]
    models = ["EASE", "Multi-VAE", "CDAE"]
    metrics = ["NDCG", "Hit Ratio", "Precision"]
    datasets = [("ml-1m", "ML-1M"), ("amazon-music", "Amazon Music")]
    colors = {"EASE": "#4C78A8", "Multi-VAE": "#F58518", "CDAE": "#54A24B"}
    markers = {"EASE": "o", "Multi-VAE": "s", "CDAE": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 3.35), sharex=True)
    for row, (dataset, label) in enumerate(datasets):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for model in models:
                slug = model.lower().replace("multi-vae", "multi-vae")
                frame = pd.read_csv(ROOT / f"results/kpi/{dataset}_{slug}.csv", index_col=0)
                values = frame.loc[metric].astype(float).to_numpy()
                ax.plot(ks, values, marker=markers[model], linewidth=1.4,
                        markersize=3.5, color=colors[model], label=model)
            ax.grid(axis="y", color="#D9D9D9", linewidth=.55)
            ax.tick_params(labelsize=7)
            ax.set_xticks(ks)
            if row == 0:
                ax.set_title(metric, fontsize=8.5, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\nscore", fontsize=8)
            if row == 1:
                ax.set_xlabel("K", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, .94], pad=.8, w_pad=.8, h_pad=.7)
    path = ROOT / "results/ranking_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


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
    ranking_curves = make_ranking_curves()

    doc = Document()
    margins(doc.sections[0])
    styles = doc.styles
    set_style_font(styles["Normal"], "Arial", 11)
    for name, size in (("Heading 1", 13), ("Heading 2", 11)):
        set_style_font(styles[name], "Arial", size, bold=True)
        styles[name].font.color.rgb = __import__("docx").shared.RGBColor.from_string("000000")
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
    add_table(doc, ["Dataset", "Users", "Items", "Events", "Density", "Train\nratio", "Warm test\ninteractions", "Eval.\nusers"], drows,
              widths=[1.05,.55,.6,.75,.65,.55,.9,.6], font_size=9.5)
    body(doc, "Before seeing model results, the filter rule was fixed: use DaisyRec's one-pass 10filter unless fewer than 500 warm-start test users remain. Both datasets exceed the threshold, so 10filter is retained. Amazon 5filter was rejected because 20,356 items imply a 3.09 GiB float64 item matrix before EASE's additional inverse/storage costs. The 2014 Amazon file contains 836,006 headerless rows before filtering; its SHA-256 is documented in README.")
    repository = doc.add_paragraph("Repository: ")
    repository.paragraph_format.space_after = Pt(3)
    add_hyperlink(repository, "github.com/Krystyna229/CA6124-Individual-Project",
                  "https://github.com/Krystyna229/CA6124-Individual-Project")

    page_break(doc)
    heading(doc, "2 CDAE and non-destructive DaisyRec extension")
    body(doc, "For user u with binary history xᵤ, CDAE applies training-only dropout to obtain x̃ᵤ and computes hᵤ = ReLU(W₁x̃ᵤ + vᵤ + b₁), where vᵤ is a learned user embedding. Decoder logits are zᵤ = W₂hᵤ + b₂. Training minimizes full-history summed binary cross-entropy plus optional L1/L2 penalties. Calling eval() disables corruption, so inference is deterministic. Relative to the original paper's sigmoid/tanh variants and discussion of negative sampling [1], ReLU and full-history BCE are deliberate framework-integration choices, not accidental substitutions.")
    body(doc, "The implementation subclasses AERecommender and reuses DaisyRec history matrices, optimizer, device handling and training loop. Evaluation changes are opt-in configuration branches: split_boundary=day, warm_start=True, ranking_mode=full, optimization_k=10 and save_user_metrics=True. Defaults preserve DaisyRec's original interaction-boundary, unfiltered and sampled-candidate behaviour. Existing EASE and Multi-VAE model code is not replaced. Seven regression tests verify defaults and the added split, filtering, ranking, CDAE-inference, tuning and per-user-metric paths.")

    heading(doc, "3 Experimental protocol")
    body(doc, "Ratings are converted to implicit interactions and duplicate user-item pairs are removed. A deterministic global timestamp boundary between complete calendar days is chosen nearest 80/20; ties are resolved deterministically. All events on one day stay on one side, important because Amazon timestamps have day precision. The actual train ratios are reported in Table 1 rather than claimed to be exactly 0.8.")
    body(doc, "Final evaluation keeps only test interactions whose user and item both occur in the 80% training set. Tuning applies the same rule relative to each training/validation split, so its evaluable item set is intentionally based on the smaller tuning-training subset. Ranking considers every training-seen item and masks the user's training positives. The final cohort is 1,152 ML-1M users / 105,699 interactions and 1,294 Amazon users / 4,065 interactions.")
    body(doc, "All tuning uses DaisyRec's supplied tune.py, whose existing backend is Optuna with a seeded TPE sampler (seed 2022). Each of the six groups receives 20 trials optimizing validation NDCG@10. The winner is retrained on the complete training partition and the untouched test set is evaluated once at K ∈ {1,5,10,20,30,50}. The uniform budget was fixed before final testing and all trial histories are retained.")
    body(doc, "CDAE and Multi-VAE use Adam for a fixed 50 epochs with early stopping disabled, giving equal epoch budgets. CDAE minimizes summed binary cross-entropy plus tuned L1/L2 penalties. Multi-VAE minimizes multinomial reconstruction loss (softmax cross-entropy) plus KL divergence. Its annealing counter advances per mini-batch; total_anneal_steps=400 makes every searched cap (0.05-0.5) reachable within 50 epochs. EASE is solved in closed form and has no epoch or optimizer.")

    heading(doc, "3.1 Search spaces", 2)
    add_table(doc, ["Model", "Search space"], [
        ["CDAE", "d {16,32,64,128,256}; drop {0,.1,.3,.5,.7}; batch {64,128,256,512}; lr {5e-5,1e-4,5e-4,.001,.005,.01}; L2 {0,1e-5,1e-4,.001,.01,.1,1}"],
        ["EASE", "regularization {1,5,10,50,100,200,500,1000,2000}"],
        ["Multi-VAE", "d {32,64,128,256,512}; drop {0,.2,.5,.7,.8}; batch {64,128,256,512}; lr {5e-5,1e-4,5e-4,.001,.005,.01}; anneal {.05,.1,.2,.5}"],
    ], widths=[1.05,5.85], font_size=10)

    page_break(doc)
    heading(doc, "4 Hyperparameter tuning")
    doc.add_picture(str(ROOT / "results/tuning_curves.png"), width=Inches(6.05))
    caption = doc.add_paragraph("Figure 1. Individual trial scores and best-so-far validation NDCG@10 (20 trials per group).")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(10)

    columns = [c for c in params.columns if c not in ("dataset", "model")]
    prows = []
    for _, row in params.iterrows():
        values = []
        for c in columns:
            if row[c] == "" or c == "ndcg":
                continue
            value = row[c]
            if c in ("latent_dim", "batch_size"):
                value = str(int(float(value)))
            elif isinstance(value, (float, int)):
                value = f"{float(value):g}"
            short = {"latent_dim": "d", "batch_size": "batch", "dropout": "drop",
                     "anneal_cap": "anneal", "reg_1": "L1", "reg_2": "L2"}.get(c, c)
            values.append(f"{short}={value}")
        selected = ", ".join(values)
        prows.append([row.dataset, row.model, selected, f"{float(row.ndcg):.4f}"])
    add_table(doc, ["Dataset", "Model", "Selected hyperparameters", "Val NDCG@10"], prows,
              widths=[.95,.8,4.25,.9], font_size=9.5)
    body(doc, "The ranges were broadened before final testing, but residual upper-bound optima remain on Amazon: Multi-VAE selects d=512 and anneal=0.5, while CDAE selects d=256. Zero regularization is a natural lower bound, but these upper-bound selections are only the best tested settings, not proven global optima; a wider targeted search remains a limitation. With the practical KL schedule and equal 50-epoch budget, ML-1M Multi-VAE instead selects interior d=256 and anneal=0.1.")
    body(doc, "Tuning stability differs sharply. On Amazon, 8/20 Multi-VAE trials and 10/20 CDAE trials score below 0.01 NDCG@10 (4 and 8, respectively, below 0.005), whereas EASE stays within 0.0276-0.0358. No ML-1M trial falls below 0.01. Thus the neural models are much more hyperparameter-sensitive on this sparse cohort, while EASE's one-dimensional search is stable; the low scores are near-failure configurations, not software crashes.")

    heading(doc, "5 Final test results")
    rrows = []
    for _, row in results.iterrows():
        rrows.append([row.dataset, row.model, f"{row['recall@10']:.4f}", f"{row['ndcg@10']:.4f}"])
    add_table(doc, ["Dataset", "Model", "Recall@10", "NDCG@10"], rrows,
              widths=[1.4,1.3,1.1,1.1], font_size=10)
    doc.add_picture(str(ranking_curves), width=Inches(6.65))
    caption = doc.add_paragraph("Figure 2. NDCG, Hit Ratio and Precision across ranking cut-offs under the same full-ranking protocol.")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(10)

    heading(doc, "6 Analysis and limitations")
    body(doc, "On ML-1M, CDAE has the highest NDCG@10 (0.4992), narrowly ahead of Multi-VAE (0.4980), while Multi-VAE has the highest Recall@10 (0.0459 versus CDAE's 0.0433). The paired CDAE-Multi-VAE NDCG difference is only 0.00120; its 10,000-resample paired-bootstrap 95% CI [-0.0111, 0.0136] crosses zero. Both neural models reliably exceed EASE in NDCG, whose pairwise intervals exclude zero. Metric choice therefore changes the top model. Their ML-1M Hit Ratio curves are also close and rank Multi-VAE first at K=10, so HR is less discriminative than NDCG here.")
    body(doc, "On Amazon, EASE leads NDCG@10 (0.0415), followed by CDAE (0.0400) and Multi-VAE (0.0383). CDAE is 3.65% below EASE and 4.45% above Multi-VAE using unrounded scores. However, all paired-bootstrap intervals cross zero: EASE-CDAE [-0.0072, 0.0100], EASE-Multi-VAE [-0.0050, 0.0117] and CDAE-Multi-VAE [-0.0074, 0.0109]. These data support no reliable pairwise winner on Amazon despite the descriptive ordering.")
    body(doc, "After 10filter, Amazon averages 11.4 interactions per user, versus 165.3 on ML-1M. Very short histories may weaken Multi-VAE's inference of a KL-regularized latent distribution and make both neural models tuning-sensitive, while EASE directly exploits item co-occurrence. These are plausible mechanisms, not causal claims, because domain, scale, sparsity and temporal coverage all change together.")
    body(doc, "Limitations are one temporal split, one seed, 20 trials, residual boundary optima and warm-start-only scope. Full ranking avoids sampled-candidate bias [7], but evaluation retains only 1,152/6,040 (19.1%) ML-1M users and 1,294/5,729 (22.6%) Amazon users after requiring train-seen users/items. The results therefore do not represent cold-start users. Future work should repeat seeds, expand targeted ranges and evaluate models designed for cold start.")

    heading(doc, "References", 2)
    refs = [
        "[1] Y. Wu et al., ‘Collaborative Denoising Auto-Encoders for Top-N Recommender Systems,’ WSDM, 2016.",
        "[2] H. Steck, ‘Embarrassingly Shallow Autoencoders for Sparse Data,’ WWW, 2019.",
        "[3] D. Liang et al., ‘Variational Autoencoders for Collaborative Filtering,’ WWW, 2018.",
        "[4] Z. Sun et al., ‘DaisyRec 2.0: Benchmarking Recommendation for Rigorous Evaluation,’ TPAMI, 2023.",
        "[5] F. Harper and J. Konstan, ‘The MovieLens Datasets: History and Context,’ TiiS, 2015.",
        "[6] J. McAuley et al., ‘Image-based Recommendations on Styles and Substitutes,’ SIGIR, 2015.",
        "[7] W. Krichene and S. Rendle, ‘On Sampled Metrics for Item Recommendation,’ KDD, 2020.",
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

    # Apply Arial explicitly to every textual run; this prevents Word theme
    # fonts from overriding the assignment's typography requirement.
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    for section in doc.sections:
        paragraphs.extend(section.header.paragraphs)
        paragraphs.extend(section.footer.paragraphs)
    for paragraph in paragraphs:
        for run in paragraph.runs:
            run.font.name = "Arial"
            fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
            for key in ("ascii", "hAnsi", "eastAsia", "cs"):
                fonts.set(qn(f"w:{key}"), "Arial")

    path = OUT / "CA6124_CDAE_Technical_Report.docx"
    doc.save(path)
    print(path)


if __name__ == "__main__":
    main()
