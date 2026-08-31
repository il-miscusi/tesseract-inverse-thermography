# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Generate the four-page submission paper from committed result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "tesseract_inverse_thermography.pdf"
W, H = A4
M = 42
INK = colors.HexColor("#e8edf5")
MUTED = colors.HexColor("#9aa6b8")
BG = colors.HexColor("#0b0e14")
PANEL = colors.HexColor("#121824")
GRID = colors.HexColor("#2a3445")
CYAN = colors.HexColor("#4fd1c5")
AMBER = colors.HexColor("#f4a259")
WHITE = colors.white

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=8.35,
                      leading=11.2, textColor=INK, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.25, leading=9.25)
TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=23,
                       leading=25, textColor=WHITE, alignment=TA_LEFT)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=10.4, leading=13,
                          textColor=CYAN)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14,
                    leading=17, textColor=WHITE, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.2,
                    leading=12, textColor=CYAN, spaceAfter=3)
CAPTION = ParagraphStyle("caption", parent=SMALL, textColor=MUTED,
                         alignment=TA_CENTER)


def p(c: canvas.Canvas, text: str, x: float, y: float, width: float,
      style: ParagraphStyle = BODY) -> float:
    para = Paragraph(text, style)
    _, height = para.wrap(width, H)
    para.drawOn(c, x, y - height)
    return y - height


def image_contain(c: canvas.Canvas, path: Path, x: float, y: float,
                  width: float, height: float) -> None:
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    scale = min(width / iw, height / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(img, x + (width - dw) / 2, y + (height - dh) / 2,
                dw, dh, preserveAspectRatio=True, mask="auto")


def page_base(c: canvas.Canvas, number: int, section: str) -> None:
    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setStrokeColor(GRID); c.line(M, H - 30, W - M, H - 30)
    c.setFont("Helvetica-Bold", 7.2); c.setFillColor(CYAN)
    c.drawString(M, H - 24, "INVERSE RENDERING THROUGH A MULTIPHYSICS EQUILIBRIUM")
    c.setFont("Helvetica", 7.2); c.setFillColor(MUTED)
    c.drawRightString(W - M, H - 24, section.upper())
    c.line(M, 27, W - M, 27)
    c.drawString(M, 16, "Tesseract Hackathon 2026 | Track 05")
    c.drawRightString(W - M, 16, str(number))


def section_rule(c: canvas.Canvas, y: float) -> None:
    c.setStrokeColor(GRID); c.line(M, y, W - M, y)


def draw_table(c: canvas.Canvas, data, x, y, widths, row_heights=None,
               font_size=7.2) -> float:
    table = Table(data, colWidths=widths, rowHeights=row_heights)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("TEXTCOLOR", (0, 0), (-1, 0), CYAN),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    tw, th = table.wrap(sum(widths), H)
    table.drawOn(c, x, y - th)
    return y - th


def main() -> None:
    b = json.loads((ROOT / "figures/experiment_b_v2.json").read_text())
    cdata = json.loads((ROOT / "figures/experiment_c_renderer.json").read_text())
    d = json.loads((ROOT / "figures/experiment_d/experiment_d_summary.json").read_text())
    e = json.loads((ROOT / "figures/experiment_e_factorial.json").read_text())
    grad = json.loads((ROOT / "figures/e2e_gradient_check.json").read_text())
    container = json.loads((ROOT / "figures/container_e2e_gradient_check.json").read_text())
    rc, ro = b["results"]["coupled"], b["results"]["one_way"]
    cf = cdata["results"]["full"]
    cm = cdata["results"]["calibration_mismatch"]
    centroid_excess = cm["centroid_shift_cells"] - cf["centroid_shift_cells"]
    dfull = d["arms"]["full"]
    dpair = d["paired_diagnostic_error"]["total_power_relative_error"]
    dci = [100 * value for value in dpair["median_bootstrap_95_ci"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUT), pagesize=A4)
    pdf.setTitle("Inverse rendering through a multiphysics equilibrium")
    pdf.setAuthor("anonymous")
    pdf.setSubject("Tesseract Hackathon 2026 Track 05 submission paper")

    # Page 1 - thesis and contribution.
    page_base(pdf, 1, "Overview")
    y = H - 54
    y = p(pdf, "Inverse rendering through a<br/>multiphysics equilibrium", M, y, W - 2 * M, TITLE)
    y -= 5
    y = p(pdf, "A differentiable LWIR camera composed with a live Fortran/JAX/PyTorch cooling model",
          M, y, W - 2 * M, SUBTITLE)
    y -= 5
    pdf.setFont("Helvetica", 8); pdf.setFillColor(MUTED)
    pdf.drawString(M, y, "github.com/il-miscusi/tesseract-inverse-thermography")
    y -= 16
    section_rule(pdf, y); y -= 14
    y = p(pdf, "<b>Abstract.</b> A thermal image is not temperature. It is band-integrated Planck radiance, modified by emissivity, reflected ambient, projection, optical blur, vignetting, and sensor calibration. We implement that measurement model as a differentiable JAX Tesseract and pull a pixel loss through it, through a coupled flow-heat fixed point, and back to an unknown volumetric heat source. The coupled system spans a PyTorch viscosity closure, a Fortran Darcy/Brinkman solver with a hand-derived adjoint, and JAX heat transport. The complete pixels-to-source derivative matches finite differences at %.2e in-process and %.2e through four served containers. On a frozen 12-scene bank with 64x32 truth, 32x16 inversion, and a held-out camera view, the calibrated system produces useful diagnoses on 12/12 scenes; a 4%% emissivity error increases power error on 12/12 pairs by a median %.2f percentage points (95%% bootstrap interval %.2f-%.2f). The stronger preregistered harm-prevalence claim is not accepted." % (grad["best_rel_err"], container["best_rel_err"], 100 * dpair["median"], dci[0], dci[1]), M, y, W - 2 * M)
    y -= 12
    image_contain(pdf, ROOT / "figures/hero.png", M, y - 146, W - 2 * M, 146)
    y -= 154
    y = p(pdf, "<b>Figure 1.</b> One noisy LWIR frame is inverted through the camera and the coupled equilibrium to recover the hidden heat-source map. Source panels share a physical scale.", M, y, W - 2 * M, CAPTION)
    y -= 10
    y = p(pdf, "Three contributions", M, y, W - 2 * M, H1)
    col = (W - 2 * M - 14) / 2
    p(pdf, "<b>Composition.</b> Four typed Tesseracts expose apply and VJP endpoints across three native derivative regimes. A matrix-free implicit adjoint composes them without a global tape.", M, y, col)
    p(pdf, "<b>Evidence.</b> A frozen 12-scene bank uses 64x32 truth, 32x16 inversion, two training views, one held-out view, absolute residual gates, and no excluded failures.", M + col + 14, y, col)
    p(pdf, "<b>Reproducibility.</b> Fast CI verifies the API path. A separate target builds and serves all four images and stores image IDs, source commit, timing, and the finite-difference table.", M, y - 52, col)
    p(pdf, "<b>Honest limits.</b> Same-grid accuracy, optimizer limits, synthetic data, and the no-PSF negative result are all reported. Claims are bounded to what the artifacts establish.", M + col + 14, y - 52, col)
    dashboard_y = 270
    section_rule(pdf, dashboard_y + 28)
    p(pdf, "Evidence dashboard", M, dashboard_y + 15, W - 2 * M, H2)
    dashboard = [
        ["API gradient", "4-container gradient", "useful unseen scenes", "paired power bias"],
        ["%.2e" % grad["best_rel_err"], "%.2e" % container["best_rel_err"],
         "%d/12" % dfull["operationally_useful_count"], "+%.2f pp" % (100 * dpair["median"])],
    ]
    draw_table(pdf, dashboard, M, dashboard_y - 3,
               [(W - 2 * M) / 4] * 4, row_heights=[24, 34], font_size=7.8)
    p(pdf, "<b>Track 05 thesis.</b> The decisive result is not merely that a camera can be differentiated. Tesseract makes its VJP operationally composable with an implicit cross-language equilibrium, and the frozen bank tests the resulting inverse system beyond its development scenes. The negative preregistered verdict remains visible alongside the positive paired evidence.", M, dashboard_y - 78, W - 2 * M, BODY)
    pdf.showPage()

    # Page 2 - method.
    page_base(pdf, 2, "Method")
    y = H - 54
    y = p(pdf, "One chain rule across four components", M, y, W - 2 * M, H1)
    y = p(pdf, "The source q drives a steady heat solve. Temperature changes viscosity; viscosity changes flow; flow changes temperature. At equilibrium T* = G(T*; q). The camera maps T* to digital counts C = R(T*). For a pixel loss J, the camera VJP supplies dJ/dT*. The implicit adjoint solves (I - dG/dT)^T lambda = dJ/dT* with GMRES, then one heat VJP returns dJ/dq. Each matrix-vector product is a chain of component VJPs; no framework traces the whole program.", M, y, W - 2 * M)
    y -= 8
    image_contain(pdf, ROOT / "figures/chain.png", M, y - 185, W - 2 * M, 185)
    y -= 192
    y = p(pdf, "<b>Figure 2.</b> Forward fixed point and reverse implicit adjoint. The fast path uses identical APIs in-process; the interoperability gate serves the four images.", M, y, W - 2 * M, CAPTION)
    y -= 8
    col = (W - 2 * M - 16) / 2
    yl = y
    yl = p(pdf, "Thermal-camera Tesseract", M, yl, col, H2)
    yl = p(pdf, "The JAX renderer integrates Planck spectral radiance over 8-14 um with 64-node Gauss-Legendre quadrature. A grey opaque surface emits eps L(T) and reflects (1-eps)L(T_ambient). A homography and bilinear sampler project radiance to the sensor; a differentiable Gaussian PSF, cos^4 vignetting, gain, and offset complete the image.", M, yl, col)
    yl -= 6
    yl = p(pdf, "Heterogeneous physics", M, yl, col, H2)
    p(pdf, "Viscosity is a learned PyTorch closure. Flow is a compiled Fortran Darcy/Brinkman solver with a hand-derived discrete adjoint. Heat transport is JAX with an implicit derivative. Only arrays and typed derivative requests cross component boundaries.", M, yl, col)
    xr = M + col + 16
    yr = y
    yr = p(pdf, "Verification hierarchy", xr, yr, col, H2)
    table = [
        ["gate", "result"],
        ["camera tests", "18 pass"],
        ["inverse utilities", "7 pass"],
        ["pixels-to-q FD, API", "%.2e" % grad["best_rel_err"]],
        ["pixels-to-q FD, 4 containers", "%.2e" % container["best_rel_err"]],
        ["served runtime", "%.0f s" % container["seconds"]],
    ]
    yr = draw_table(pdf, table, xr, yr, [col * 0.66, col * 0.34])
    yr -= 10
    yr = p(pdf, "Why Tesseract is load-bearing", xr, yr, col, H2)
    p(pdf, "JAX cannot trace the PyTorch tape or the Fortran binary; neither tape contains the equilibrium solve. A custom callback layer or rewrite could expose the same mathematical derivative. Tesseract supplies a typed, remotely executable composition without rewriting the native solvers.", xr, yr, col)
    pdf.showPage()

    # Page 3 - decisive evidence.
    page_base(pdf, 3, "Frozen bank")
    y = H - 54
    y = p(pdf, "Generalization on twelve unseen fault scenes", M, y, W - 2 * M, H1)
    y = p(pdf, "Experiment D seals seeds 101-112 until the inverse method, two training views, held-out third view, 4% emissivity mismatch, absolute residual gates, and reporting rule are fixed. Truth uses a 64x32 coupled solve and inversion uses 32x16. A known uniform load plus one perturbation at each of 20 chip cells independently calibrates a differentiable observation-space multi-fidelity tangent; no fault scene contributes to it.", M, y, W - 2 * M)
    y -= 7
    image_contain(pdf, ROOT / "figures/experiment_d_generalization.png", M, y - 224, W - 2 * M, 224)
    y -= 230
    y = p(pdf, "<b>Figure 3.</b> Every paired point is retained. Red mismatch markers fail at least one independent plausibility condition (residual mean or whiteness can reject an arm even when RMS is below 2 sigma).", M, y, W - 2 * M, CAPTION)
    y -= 8
    rows = [
        ["arm", "plausible", "useful", "RMS counts", "centroid mm", "power error"],
        ["calibrated", "%d/12" % dfull["absolute_plausible_fit_count"],
         "%d/12" % dfull["operationally_useful_count"],
         "%.3f" % dfull["metrics"]["holdout_rms_counts"]["median"],
         "%.3f" % dfull["metrics"]["centroid_error_mm"]["median"],
         "%.3f%%" % (100 * dfull["metrics"]["total_power_relative_error"]["median"])],
        ["4% emissivity error", "%d/12" % d["arms"]["mismatch"]["absolute_plausible_fit_count"],
         "%d/12" % d["arms"]["mismatch"]["operationally_useful_count"],
         "%.3f" % d["arms"]["mismatch"]["metrics"]["holdout_rms_counts"]["median"],
         "%.3f" % d["arms"]["mismatch"]["metrics"]["centroid_error_mm"]["median"],
         "%.3f%%" % (100 * d["arms"]["mismatch"]["metrics"]["total_power_relative_error"]["median"])],
    ]
    y = draw_table(pdf, rows, M, y, [118, 72, 68, 82, 82, 89])
    y -= 12
    col = (W - 2 * M - 16) / 2
    p(pdf, "<b>Positive evidence.</b> The calibrated arm is useful on 12/12 scenes. Emissivity mismatch raises absolute power error on 12/12 pairs by a median <b>%.2f points</b>; the deterministic paired-bootstrap 95%% interval is %.2f-%.2f points." % (100 * dpair["median"], dci[0], dci[1]), M, y, col)
    p(pdf, "<b>Negative verdict retained.</b> The stronger preregistered risk claim required at least 6/12 mismatches to be both plausible and at least five points harmful. Only <b>%d/12</b> qualify, so that claim is <b>not accepted</b>." % d["materially_harmful_plausible_mismatch_count"], M + col + 16, y, col)
    evidence_y = 272
    section_rule(pdf, evidence_y + 28)
    p(pdf, "Why this bank is hard to game", M, evidence_y + 15, W - 2 * M, H2)
    p(pdf, "<b>Sealed scenes.</b> Development uses seeds 0-2; the reported bank is exactly 101-112. The summarizer refuses missing or extra seed artifacts and uses a fixed bootstrap seed.", M, evidence_y - 3, col, SMALL)
    p(pdf, "<b>Absolute validation.</b> A held-out fit must satisfy RMS <=4 counts, |mean| <=0.5 counts, and both lag-1 residual correlations <=0.10. Closeness to the calibrated arm is never a plausibility test.", M + col + 16, evidence_y - 3, col, SMALL)
    p(pdf, "<b>Representability audit.</b> The original exact-source coarse oracle missed fine-grid pixels by 51 counts. Known-load calibration reduced the development oracle to about 2 counts before the bank was opened; the source was not optimized to conceal the discrepancy.", M, evidence_y - 64, col, SMALL)
    p(pdf, "<b>Physical diagnosis.</b> Gates use millimetres and total power, not only source-vector L2. Every optimizer converged, all raw JSON/NPZ pairs and logs are committed, and the final animation uses a real bank trajectory.", M + col + 16, evidence_y - 64, col, SMALL)
    pdf.showPage()

    # Page 4 - previous result, limits, reproducibility.
    page_base(pdf, 4, "Discussion")
    y = H - 54
    y = p(pdf, "What the complete evidence establishes", M, y, W - 2 * M, H1)
    col = (W - 2 * M - 16) / 2
    yl = y
    yl = p(pdf, "Forward x gradient factorial", M, yl, col, H2)
    yl = p(pdf, "Experiment E separates Experiment B's two interventions. Exact coupled forward/gradient reaches source L2 %.4f and data loss %.3f. Keeping the coupled forward but truncating only the implicit reverse feedback converges at L2 %.4f and loss %.3f. Frozen forward/matching gradient reaches L2 %.4f and loss %.3f. Forward fidelity dominates pixel fit; exact gradient fidelity improves source shape on this fixed problem." % (rc["rel_l2"], rc["final_data_loss"], e["coupled_truncated"]["final"]["rel_l2"], e["coupled_truncated"]["final"]["data_loss"], ro["rel_l2"], ro["final_data_loss"]), M, yl, col)
    yl -= 8
    yl = p(pdf, "Historical stress test", M, yl, col, H2)
    yl = p(pdf, "Experiment C first exposed the inverse crime: with 64x32 truth and 32x16 inversion, calibrated source L2 rose to %.3f and pixel RMS to %.2f counts. Its relative calibration-shift gate passed, but the absolute noise-level gate failed. Experiment D repairs representability with independent known-load calibration rather than hiding discrepancy in the source." % (cf["rel_l2"], cf["pixel_rms_counts"]), M, yl, col)
    yl -= 8
    yl = p(pdf, "Limitations", M, yl, col, H2)
    p(pdf, "Measurements remain synthetic; the chip footprint is known; calibration uncertainty is represented by one fixed mismatch rather than a posterior; and material/camera physics are prototype-grade rather than instrument-grade. Real calibrated imagery, transient data, unknown supports, and posterior uncertainty maps are the next validation layer.", M, yl, col)

    xr = M + col + 16
    yr = y
    yr = p(pdf, "Reproduce", xr, yr, col, H2)
    box_h = 112
    pdf.setFillColor(PANEL); pdf.setStrokeColor(GRID)
    pdf.roundRect(xr, yr - box_h, col, box_h, 5, fill=1, stroke=1)
    commands = [
        "pip install -r requirements.txt",
        "make judge",
        "make verify",
        "make verify-containers",
        "make summarize-experiment-d",
        "make animation",
        "make paper landing",
    ]
    yy = yr - 18
    pdf.setFont("Courier", 7.1); pdf.setFillColor(INK)
    for command in commands:
        pdf.drawString(xr + 10, yy, "$ " + command); yy -= 15
    yr -= box_h + 12
    yr = p(pdf, "Container receipt", xr, yr, col, H2)
    yr = p(pdf, "Mode: <b>%s</b><br/>Best FD error: <b>%.2e</b><br/>Runtime: %.0f s<br/>Source: %s<br/>Four immutable image IDs are stored in the JSON artifact." % (container["execution_mode"], container["best_rel_err"], container["seconds"], container["source_commit"][:12]), xr, yr, col)
    yr -= 8
    yr = p(pdf, "Artifact map", xr, yr, col, H2)
    p(pdf, "Frozen protocol: writeup/EXPERIMENT_D_PROTOCOL.md<br/>Bank summary: figures/experiment_d/experiment_d_summary.json<br/>Twelve raw JSON/NPZ pairs: figures/experiment_d/bank/<br/>Container receipt: figures/container_e2e_gradient_check.json<br/>Code: github.com/il-miscusi/tesseract-inverse-thermography", xr, yr, col, SMALL)

    matrix_y = 420
    section_rule(pdf, matrix_y + 25)
    p(pdf, "Claim-to-artifact matrix", M, matrix_y + 13, W - 2 * M, H2)
    claims = [
        ["claim", "direct evidence", "boundary"],
        ["Cross-framework derivative is correct", "two end-to-end FD receipts", "directional checks, small grids"],
        ["Composed inverse generalizes", "Experiment D, 12/12 useful", "synthetic, known support"],
        ["4% emissivity biases inferred power", "12/12 pairs; +4.42 pp median", "harm prevalence claim failed"],
        ["Coupled model improves same-grid recovery", "Experiment B v2, 0.137 vs 0.246 L2", "both arms hit budget"],
        ["Exact adjoint improves source shape", "factorial: 0.137 vs 0.241 L2", "one fixed problem"],
    ]
    draw_table(pdf, claims, M, matrix_y - 2, [166, 190, 155], font_size=6.8)
    applications_y = 292
    p(pdf, "Transfer applications", M, applications_y, W - 2 * M, H2)
    third = (W - 2 * M - 20) / 3
    p(pdf, "<b>Electronics cooling.</b> Localize a failing die or blocked channel from the visible cold-plate surface while respecting convection and radiometry.", M, applications_y - 16, third, SMALL)
    p(pdf, "<b>Additive manufacturing.</b> Pull pyrometer residuals through a differentiable sensor and melt-pool equilibrium to infer hidden heat input.", M + third + 10, applications_y - 16, third, SMALL)
    p(pdf, "<b>Active thermography.</b> Infer delamination or subsurface defects from transient surface radiation with a swappable camera component.", M + 2 * (third + 10), applications_y - 16, third, SMALL)

    y2 = 180
    section_rule(pdf, y2); y2 -= 14
    y2 = p(pdf, "References", M, y2, W - 2 * M, H2)
    refs = (
        "[1] Pasteur Labs, <i>Tesseract: universal, autodiff-native software components</i>, 2026. "
        "[2] J. Bradbury et al., <i>JAX: composable transformations of Python+NumPy programs</i>, 2018. "
        "[3] A. Paszke et al., <i>PyTorch: An Imperative Style, High-Performance Deep Learning Library</i>, NeurIPS 2019. "
        "[4] P. Virtanen et al., <i>SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python</i>, Nature Methods 2020. "
        "[5] M. Planck, <i>On the Law of Distribution of Energy in the Normal Spectrum</i>, 1901."
    )
    p(pdf, refs, M, y2, W - 2 * M, SMALL)
    pdf.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
