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
    c.drawString(M, 16, "Usi Adia-Nimuwa | Tesseract Hackathon 2026 | Track 05")
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
    grad = json.loads((ROOT / "figures/e2e_gradient_check.json").read_text())
    container = json.loads((ROOT / "figures/container_e2e_gradient_check.json").read_text())
    rc, ro = b["results"]["coupled"], b["results"]["one_way"]
    cf = cdata["results"]["full"]
    cm = cdata["results"]["calibration_mismatch"]
    centroid_excess = cm["centroid_shift_cells"] - cf["centroid_shift_cells"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUT), pagesize=A4)
    pdf.setTitle("Inverse rendering through a multiphysics equilibrium")
    pdf.setAuthor("Usi Adia-Nimuwa")
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
    pdf.drawString(M, y, "Usi Adia-Nimuwa | github.com/il-miscusi/tesseract-inverse-thermography")
    y -= 16
    section_rule(pdf, y); y -= 14
    y = p(pdf, "<b>Abstract.</b> A thermal image is not temperature. It is band-integrated Planck radiance, modified by emissivity, reflected ambient, projection, optical blur, vignetting, and sensor calibration. We implement that measurement model as a differentiable JAX Tesseract and pull a pixel loss through it, through a coupled flow-heat fixed point, and back to an unknown volumetric heat source. The coupled system spans a PyTorch viscosity closure, a Fortran Darcy/Brinkman solver with a hand-derived adjoint, and JAX heat transport. The complete pixels-to-source derivative matches finite differences at %.2e in-process and %.2e through four served containers. Under independent truth/inversion grids, a modest camera-calibration mismatch remains within 2.14x of the calibrated pixel RMS yet moves the inferred hotspot an additional %.2f coarse cells. The renderer therefore changes the physical diagnosis; it is not a visualization layer." % (grad["best_rel_err"], container["best_rel_err"], centroid_excess), M, y, W - 2 * M)
    y -= 12
    image_contain(pdf, ROOT / "figures/hero.png", M, y - 146, W - 2 * M, 146)
    y -= 154
    y = p(pdf, "<b>Figure 1.</b> One noisy LWIR frame is inverted through the camera and the coupled equilibrium to recover the hidden heat-source map. Source panels share a physical scale.", M, y, W - 2 * M, CAPTION)
    y -= 10
    y = p(pdf, "Three contributions", M, y, W - 2 * M, H1)
    col = (W - 2 * M - 14) / 2
    p(pdf, "<b>Composition.</b> Four typed Tesseracts expose apply and VJP endpoints across three differentiation systems. A matrix-free implicit adjoint composes them without a global tape.", M, y, col)
    p(pdf, "<b>Evidence.</b> A committed-before-results renderer ablation uses a 64x32 truth grid and 32x16 inversion, reports negative controls, and makes calibration sensitivity measurable.", M + col + 14, y, col)
    p(pdf, "<b>Reproducibility.</b> Fast CI verifies the API path. A separate target builds and serves all four images and stores image IDs, source commit, timing, and the finite-difference table.", M, y - 52, col)
    p(pdf, "<b>Honest limits.</b> Same-grid accuracy, optimizer limits, synthetic data, and the no-PSF negative result are all reported. Claims are bounded to what the artifacts establish.", M + col + 14, y - 52, col)
    dashboard_y = 270
    section_rule(pdf, dashboard_y + 28)
    p(pdf, "Evidence dashboard", M, dashboard_y + 15, W - 2 * M, H2)
    dashboard = [
        ["API gradient", "4-container gradient", "same-grid source L2", "calibration shift"],
        ["%.2e" % grad["best_rel_err"], "%.2e" % container["best_rel_err"],
         "%.4f" % rc["rel_l2"], "+%.2f cells" % centroid_excess],
    ]
    draw_table(pdf, dashboard, M, dashboard_y - 3,
               [(W - 2 * M) / 4] * 4, row_heights=[24, 34], font_size=7.8)
    p(pdf, "<b>Track 05 thesis.</b> The decisive result is not that a camera can be differentiated. It is that modest calibration error changes the recovered physical cause even when the image fit remains inside the predeclared plausibility envelope. Tesseract makes that camera VJP composable with an implicit, cross-language equilibrium.", M, dashboard_y - 78, W - 2 * M, BODY)
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
    p(pdf, "JAX cannot trace the PyTorch tape or the Fortran binary; neither tape contains the equilibrium solve. Tesseract's common apply/VJP contract is the compositional abstraction. Removing it does not merely slow the system: it removes the cross-language derivative needed by the inverse problem.", xr, yr, col)
    pdf.showPage()

    # Page 3 - decisive evidence.
    page_base(pdf, 3, "Evidence")
    y = H - 54
    y = p(pdf, "The renderer changes the diagnosis", M, y, W - 2 * M, H1)
    y = p(pdf, "Experiment C was committed before its first result-producing run. Observations come from the complete camera over a 64x32 coupled solve; all recoveries invert at 32x16 with identical physics, data, TV prior, initialization, optimizer, and budget. Only the assumed camera changes.", M, y, W - 2 * M)
    y -= 7
    image_contain(pdf, ROOT / "figures/renderer_necessity.png", M, y - 276, W - 2 * M, 276)
    y -= 282
    y = p(pdf, "<b>Figure 3.</b> Calibrated and modest-mismatch recoveries see the same observation. Source panels are normalized individually; printed power ratios preserve amplitude information. The residual color scale is shared.", M, y, W - 2 * M, CAPTION)
    y -= 8
    rows = [["camera assumption", "pixel RMS", "source L2", "centroid", "power", "status"]]
    names = [("full", "calibrated"), ("blackbody", "blackbody"),
             ("no_psf", "no PSF"), ("no_vignetting", "no vignette"),
             ("calibration_mismatch", "modest mismatch")]
    for key, label in names:
        r = cdata["results"][key]
        rows.append([label, "%.2f" % r["pixel_rms_counts"], "%.3f" % r["rel_l2"],
                     "%.2f" % r["centroid_shift_cells"], "%.3f" % r["total_power_ratio"],
                     "conv" if r["optimizer"]["status"] == 0 else "limit"])
    y = draw_table(pdf, rows, M, y, [128, 72, 72, 72, 72, 64])
    y -= 10
    col = (W - 2 * M - 16) / 2
    p(pdf, "<b>Predeclared verdict: PASS.</b> A simplified arm must fit within 3x the calibrated RMS and materially worsen source L2, centroid, or power. Only the modest calibration-mismatch arm passes both: 29.82 counts RMS and an additional %.2f-cell centroid error." % centroid_excess, M, y, col)
    p(pdf, "<b>Negative evidence retained.</b> Blackbody and no-vignetting models are visibly rejectable. Removing PSF blur fits nearly as well and improves coarse-grid source L2 from %.3f to %.3f. The supported claim is calibration sensitivity, not that every optical stage is indispensable." % (cf["rel_l2"], cdata["results"]["no_psf"]["rel_l2"]), M + col + 16, y, col)
    pdf.showPage()

    # Page 4 - previous result, limits, reproducibility.
    page_base(pdf, 4, "Discussion")
    y = H - 54
    y = p(pdf, "What the complete evidence establishes", M, y, W - 2 * M, H1)
    col = (W - 2 * M - 16) / 2
    yl = y
    yl = p(pdf, "Same-grid source recovery", M, yl, col, H2)
    yl = p(pdf, "Experiment B v2 recovers the two-hotspot map at relative L2 %.4f, centroid %.2f cells, and power %.3f. A frozen-viscosity forward model ends at L2 %.4f. Both L-BFGS-B arms reach the 250-iteration limit; the comparison is a budget-matched model-mismatch endpoint, not a gradient-only intervention or convergence floor." % (rc["rel_l2"], rc["centroid_shift_cells"], rc["total_power_ratio"], ro["rel_l2"]), M, yl, col)
    yl -= 8
    yl = p(pdf, "Independent-grid stress test", M, yl, col, H2)
    yl = p(pdf, "With 64x32 truth and 32x16 inversion, calibrated source L2 rises to %.3f and pixel RMS to %.2f counts. The old %.4f accuracy is therefore not robust to discretization mismatch. Centroid and power remain useful diagnostic summaries, but exact source shape is ill-conditioned from one blurred image." % (cf["rel_l2"], cf["pixel_rms_counts"], rc["rel_l2"]), M, yl, col)
    yl -= 8
    yl = p(pdf, "Limitations", M, yl, col, H2)
    p(pdf, "Measurements remain synthetic; one image underdetermines a 512-value source; calibration uncertainty is represented by one fixed mismatch rather than a posterior; material and camera physics are prototype-grade rather than instrument-grade; and four of five Experiment C optimizers stop at their budget. Real calibrated imagery, transient or multi-view data, held-out source families, and uncertainty maps are the next validation layer.", M, yl, col)

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
        "make experiment-c",
        "make figures renderer-figure landing",
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
    p(pdf, "Protocol: writeup/RENDERER_PROTOCOL.md<br/>Renderer result: figures/experiment_c_renderer.json<br/>Container receipt: figures/container_e2e_gradient_check.json<br/>Full narrative: writeup/WRITEUP.md<br/>Code and data: github.com/il-miscusi/tesseract-inverse-thermography", xr, yr, col, SMALL)

    matrix_y = 420
    section_rule(pdf, matrix_y + 25)
    p(pdf, "Claim-to-artifact matrix", M, matrix_y + 13, W - 2 * M, H2)
    claims = [
        ["claim", "direct evidence", "boundary"],
        ["Cross-framework derivative is correct", "two end-to-end FD receipts", "directional checks, small grids"],
        ["Renderer calibration changes diagnosis", "Experiment C, +1.82-cell shift", "one fixed mismatch"],
        ["Coupled model improves same-grid recovery", "Experiment B v2, 0.137 vs 0.246 L2", "both arms hit budget"],
        ["Every optical stage is necessary", "not supported: no-PSF is negative", "claim rejected"],
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
