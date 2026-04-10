import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("output", exist_ok=True)

# ═══════════════════════════════════════════════════
# NODE DIAGRAM
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(0, 11)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

NAVY   = '#0d1b2a'
PURPLE = '#5b2c8c'
RED2   = '#a93226'
BLUE2  = '#1a5276'
GREEN2 = '#1e8449'
ORANGE = '#ca6f1e'
GRAY2  = '#2c3e50'

def rect_node(ax, cx, cy, w, h, label, color=NAVY, fs=8.2):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.07",
                       facecolor=color, edgecolor='#cccccc', linewidth=1.0, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=4,
            multialignment='center', linespacing=1.4)

def diamond(ax, cx, cy, w, h, label, color=PURPLE, fs=8.0):
    pts = [[cx, cy+h/2],[cx+w/2, cy],[cx, cy-h/2],[cx-w/2, cy]]
    poly = plt.Polygon(pts, closed=True, facecolor=color,
                       edgecolor='#cccccc', linewidth=1.0, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=4,
            multialignment='center', linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, lbl='', lx=0, ly=0, curve=None):
    style = f'arc3,rad={curve}' if curve else 'arc3,rad=0'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.3,
                                connectionstyle=style), zorder=2)
    if lbl:
        ax.text(x1+lx, y1+ly, lbl, fontsize=8, color='#c0392b',
                fontweight='bold', zorder=5)

rect_node(ax, 5.5, 9.3,  2.4, 0.55, 'N1  ENTRY',                                     color=GRAY2,  fs=8.5)
diamond  (ax, 5.5, 8.2,  3.6, 0.85, 'N2: if not retrieved_passages?\n[Condition A]',  fs=8.0)
rect_node(ax, 1.4, 6.4,  2.5, 1.15, 'N3\nS2: score = 0.0\nS3: label = Unverified\nS4-S6: empty fields', color=RED2, fs=7.5)
rect_node(ax, 8.9, 7.2,  2.5, 0.65, 'N4: compute_confidence\n_score()  [S7]',         color=BLUE2,  fs=7.5)
diamond  (ax, 8.9, 6.1,  3.2, 0.82, 'N5: if score >= 0.8?\n[Condition B]',            fs=8.0)
rect_node(ax, 8.9, 5.0,  2.5, 0.60, 'N6: label = "High - Verified"\n[S9]',            color=GREEN2, fs=7.5)
diamond  (ax, 8.9, 3.9,  3.2, 0.82, 'N7: elif score >= 0.5?\n[Condition C]',          fs=8.0)
rect_node(ax, 8.9, 2.8,  2.5, 0.60, 'N8: label = "Medium - Verified"\n[S11]',         color=ORANGE, fs=7.5)
rect_node(ax, 8.9, 1.75, 2.5, 0.60, 'N9: label = "Low - Unverified"\n[S12]',          color=RED2,   fs=7.5)
rect_node(ax, 5.5, 0.55, 2.4, 0.55, 'N10  EXIT',                                     color=GRAY2,  fs=8.5)

arrow(ax, 5.5,  9.02, 5.5,  8.63)
arrow(ax, 3.73, 7.92, 1.4,  7.0,   'T (A)', -0.6,  0.12)
arrow(ax, 7.27, 7.92, 8.9,  7.53,  'F (A)',  0.1,   0.12)
arrow(ax, 8.9,  6.87, 8.9,  6.52)
arrow(ax, 8.9,  5.72, 8.9,  5.30,  'T (B)',  0.15,  0.0)
arrow(ax, 8.9,  4.69, 8.9,  4.32,  'F (B)',  0.15,  0.0)
arrow(ax, 8.9,  3.51, 8.9,  3.10,  'T (C)',  0.15,  0.0)
arrow(ax, 8.9,  3.51, 8.9,  2.05,  'F (C)',  0.15, -0.5)
arrow(ax, 1.4,  5.82, 1.4,  0.55)
ax.annotate('', xy=(4.28, 0.55), xytext=(1.4, 0.55),
            arrowprops=dict(arrowstyle='->', color='#444444', lw=1.3), zorder=2)
for ny in [5.0, 2.8, 1.75]:
    ax.annotate('', xy=(6.72, 0.55), xytext=(7.65, ny),
                arrowprops=dict(arrowstyle='->', color='#777777', lw=1.0,
                                connectionstyle='arc3,rad=0.25'), zorder=2)

legend_items = [
    (RED2,   'P1: A=True  ->  Unverified'),
    (GREEN2, 'P2: A=F, B=True  ->  High'),
    (ORANGE, 'P3: A=F, B=F, C=True  ->  Medium'),
    (RED2,   'P4: A=F, B=F, C=False  ->  Low'),
]
for i, (c, lbl) in enumerate(legend_items):
    y = 2.1 - i * 0.42
    ax.add_patch(FancyBboxPatch((0.25, y-0.14), 0.32, 0.28,
                                boxstyle='round,pad=0.03', facecolor=c,
                                edgecolor='#aaaaaa', linewidth=0.7, zorder=5))
    ax.text(0.68, y, lbl, fontsize=7.5, va='center', color='#222222', zorder=5)

ax.text(0.25, 2.4, 'Paths', fontsize=8, fontweight='bold', color='#222222')
ax.add_patch(FancyBboxPatch((0.1, 1.5), 2.95, 1.1, boxstyle='round,pad=0.06',
                             facecolor='#f9f9f9', edgecolor='#cccccc',
                             linewidth=0.8, zorder=4))
ax.set_title('Control Flow Node Diagram — Confidence Label Assignment Block',
             fontsize=11, fontweight='bold', color='#0d1b2a', pad=10)
plt.tight_layout()
plt.savefig('output/node_diagram_v3.png', dpi=170, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Diagram saved.")

# ═══════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether, Preformatted
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H = letter

doc = SimpleDocTemplate(
    "output/Confid_AI_Testing_Final.pdf",
    pagesize=letter,
    rightMargin=0.85*inch, leftMargin=0.85*inch,
    topMargin=0.8*inch,    bottomMargin=0.8*inch
)

NAVY   = colors.HexColor('#0d1b2a')
BLUE   = colors.HexColor('#1b4f72')
LTBLUE = colors.HexColor('#d6e8f7')
TBLHDR = colors.HexColor('#1b4f72')
TBLALT = colors.HexColor('#eaf2fb')
LGRAY  = colors.HexColor('#f7f9fc')

styles = getSampleStyleSheet()
S = lambda n, **kw: ParagraphStyle(n, parent=styles['Normal'], **kw)

TITLE = S('T1', fontSize=20, fontName='Helvetica-Bold', alignment=TA_CENTER,
          textColor=NAVY, spaceAfter=3)
SUBT  = S('T2', fontSize=10, fontName='Helvetica', alignment=TA_CENTER,
          textColor=colors.HexColor('#555555'), spaceAfter=2)
H1    = S('H1', fontSize=13, fontName='Helvetica-Bold', textColor=BLUE,
          spaceBefore=8, spaceAfter=4)
H2    = S('H2', fontSize=10.5, fontName='Helvetica-Bold', textColor=NAVY,
          spaceBefore=6, spaceAfter=3)
BODY  = S('BD', fontSize=9.5, fontName='Helvetica', leading=15,
          textColor=colors.black, spaceAfter=4, alignment=TA_JUSTIFY)
BOLD  = S('BL', fontSize=9.5, fontName='Helvetica-Bold', leading=14,
          textColor=colors.black, spaceAfter=2)
MONO  = S('MN', fontSize=8,   fontName='Courier', leading=12,
          textColor=colors.black, backColor=LGRAY,
          borderColor=colors.HexColor('#cccccc'), borderWidth=0.5,
          borderPad=8, spaceAfter=5)
SM    = S('SM', fontSize=8, fontName='Helvetica', leading=11,
          textColor=colors.HexColor('#555555'), spaceAfter=3)

# Cell paragraph style for wrapping inside table cells
CELL  = S('CL', fontSize=8.5, fontName='Helvetica', leading=12,
          textColor=colors.black)
CELLB = S('CB', fontSize=8.5, fontName='Helvetica-Bold', leading=12,
          textColor=colors.white)

USABLE = W - 1.7 * inch  # 6.8 inches

def rule(t=1.2, c=BLUE):
    return HRFlowable(width='100%', thickness=t, color=c,
                      spaceAfter=5, spaceBefore=1)

def make_tbl(data, widths, hdr=TBLHDR):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1,  0), hdr),
        ('TEXTCOLOR',      (0, 0), (-1,  0), colors.white),
        ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTNAME',       (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',       (0, 0), (-1, -1), 8.5),
        ('LEADING',        (0, 0), (-1, -1), 12),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TBLALT]),
        ('GRID',           (0, 0), (-1, -1), 0.35, colors.HexColor('#aaaaaa')),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',     (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
        ('LEFTPADDING',    (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 6),
    ]))
    return t

def kv_tbl(rows, w1=1.65*inch):
    w2 = USABLE - w1
    t = Table(rows, colWidths=[w1, w2])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, -1), LTBLUE),
        ('FONTNAME',      (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',      (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 0), (-1,-1), 8.5),
        ('LEADING',       (0, 0), (-1,-1), 12),
        ('GRID',          (0, 0), (-1,-1), 0.35, colors.HexColor('#aaaaaa')),
        ('VALIGN',        (0, 0), (-1,-1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1,-1), 5),
        ('BOTTOMPADDING', (0, 0), (-1,-1), 5),
        ('LEFTPADDING',   (0, 0), (-1,-1), 6),
        ('RIGHTPADDING',  (0, 0), (-1,-1), 6),
        ('ROWBACKGROUNDS',(0, 0), (-1,-1), [colors.white, TBLALT]),
    ]))
    return t

# ── helper: wrap a list of rows with Paragraph cells ──────────────────────
def p(text, style=CELL):
    return Paragraph(text, style)

story = []

# ── COVER ─────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 0.25*inch),
    Paragraph("Confid.AI", TITLE),
    Spacer(1, 0.10*inch),
    Paragraph("Software Testing Analysis Report", TITLE),
    Spacer(1, 0.10*inch),
    rule(2, NAVY),
    Paragraph("Team Members: Jaideep  |  Dharaneesh  |  Nipun", SUBT),
    rule(0.5, colors.HexColor('#aaaaaa')),
    Spacer(1, 0.15*inch),
]
story.append(Paragraph(
    "This report presents a structured software testing analysis for a selected section of "
    "the Confid.AI backend. The section chosen is the <b>confidence label assignment block</b> "
    "inside the <font name='Courier'>submit_query()</font> endpoint "
    "(<font name='Courier'>backend/api/endpoints.py</font>). "
    "This block was selected because it contains clearly defined branches and compound predicates, "
    "and it directly controls the confidence label displayed to every user for each AI-generated answer.",
    BODY))

# ── 1. CODE ───────────────────────────────────────────────────────────────
story += [Paragraph("1. Code Section Under Test", H1), rule()]
story.append(Paragraph(
    "File: <font name='Courier'>backend/api/endpoints.py</font> — "
    "Function: <font name='Courier'>submit_query()</font>", BODY))

code = """\
# Thresholds (config.py):
#   HIGH_CONFIDENCE_THRESHOLD   = 0.8
#   MEDIUM_CONFIDENCE_THRESHOLD = 0.5

# Condition A
if not retrieved_passages:                      # [S1]
    confidence_score = 0.0                      # [S2]
    confidence_label = "Unverified - No Data"   # [S3]
    explanation      = "No documents found."    # [S4]
    citations        = []                       # [S5]
    score_breakdown  = { all zeros }            # [S6]
else:
    confidence_score, explanation,              # [S7]
    citations, score_breakdown =
        scoring_service.compute_confidence_score(
            answer, question, retrieved_passages
        )
    # Condition B 
    if confidence_score >= 0.8:                 # [S8]
        confidence_label = "High - Verified"    # [S9]
    # Condition C
    elif confidence_score >= 0.5:               # [S10]
        confidence_label = "Medium - Verified"  # [S11]
    else:
        confidence_label = "Low - Unverified"   # [S12]
"""
story.append(Preformatted(code, MONO))

# ── 2. STRUCTURE ──────────────────────────────────────────────────────────
story += [Paragraph("2. Code Structure Analysis", H1), rule()]
story.append(Paragraph("2.1  Basic Conditions", H2))
story.append(make_tbl([
    ["ID", "Expression",             "Evaluated When",                "Possible Values"],
    ["A",  "not retrieved_passages", "Always on entry",               "True / False"],
    ["B",  "confidence_score >= 0.8","Only when A = False",           "True / False"],
    ["C",  "confidence_score >= 0.5","Only when A = False, B = False","True / False"],
], [0.4*inch, 2.25*inch, 2.4*inch, 1.75*inch]))

story += [Spacer(1, 0.08*inch), Paragraph("2.2  Statements", H2)]
story.append(make_tbl([
    ["ID",  "Statement",                                    "Paths"],
    ["S1",  "if not retrieved_passages (branch check)",     "All"],
    ["S2",  "confidence_score = 0.0",                       "P1"],
    ["S3",  "confidence_label = 'Unverified - No Data'",    "P1"],
    ["S4",  "explanation = 'No documents found.'",          "P1"],
    ["S5",  "citations = [ ]",                               "P1"],
    ["S6",  "score_breakdown = { all zeros }",              "P1"],
    ["S7",  "scoring_service.compute_confidence_score(...)","P2, P3, P4"],
    ["S8",  "if confidence_score >= 0.8",                   "P2, P3, P4"],
    ["S9",  "confidence_label = 'High - Verified'",         "P2"],
    ["S10", "elif confidence_score >= 0.5",                 "P3, P4"],
    ["S11", "confidence_label = 'Medium - Verified'",       "P3"],
    ["S12", "confidence_label = 'Low - Unverified'",        "P4"],
], [0.45*inch, 3.85*inch, 2.5*inch]))

story += [Spacer(1, 0.05*inch),
          Paragraph("Total: 12 statements  ·  3 basic conditions  ·  6 branches  ·  4 executable paths", BOLD)]

# ── 3. NODE DIAGRAM ───────────────────────────────────────────────────────
story.append(PageBreak())
story += [Paragraph("3. Control Flow Node Diagram", H1), rule()]
story.append(Paragraph(
    "Rectangles represent process or terminal nodes. Diamonds represent decision nodes. "
    "Arrows are labelled T (True) or F (False). The four coloured nodes correspond to the "
    "four distinct execution paths.", BODY))

img_h = USABLE * (1682 / 1852)
story.append(Image("output/node_diagram_v3.png", width=USABLE, height=img_h))
story.append(Spacer(1, 0.08*inch))

story.append(Paragraph("3.1  Path Descriptions", H2))
story.append(make_tbl([
    ["Path","Node Sequence",                       "Condition Values",              "Label Outcome"],
    ["P1",  "N1 -> N2 -> N3 -> N10",               "A = True",                     "Unverified - No Data"],
    ["P2",  "N1 -> N2 -> N4 -> N5 -> N6 -> N10",   "A = False, B = True",              "High - Verified"],
    ["P3",  "N1 -> N2 -> N4 -> N5 -> N7 -> N8 -> N10",         "A = False, B = False, C = True",     "Medium - Verified"],
    ["P4",  "N1 -> N2 -> N4 -> N5 -> N7 -> N9 -> N10",         "A = False, B = False, C = False",    "Low - Unverified"],
], [0.45*inch, 2.4*inch, 2.3*inch, 1.65*inch]))
story.append(Paragraph(
    "All 4 paths are feasible. There is no dead code or unreachable branch in this section.", SM))

# ── 4. TEST CASES ─────────────────────────────────────────────────────────
story.append(PageBreak())
story += [Paragraph("4. Test Suite", H1), rule()]
story.append(Paragraph(
    "Four test cases are defined — one per execution path. Together they satisfy all seven "
    "coverage criteria. No test shares preconditions with another.", BODY))

TESTS = [
    {
        "id":"TC-1","path":"P1","name":"No Documents in Knowledge Base",
        "pre": "ChromaDB collection is empty. No PDFs have been uploaded to the system.",
        "inp": "question = 'What is recursion?'\nretrieved_passages = [ ]  (empty list)",
        "cond":"A = True  (not retrieved_passages evaluates to True)",
        "out": "confidence_score = 0.0\nconfidence_label = 'Unverified - No Data'\ncitations = []\nscore_breakdown = {consistency:0, semantic:0, completeness:0, precision:0}",
        "hit": "S1, S2, S3, S4, S5, S6",
    },
    {
        "id":"TC-2","path":"P2","name":"Strong Document Match — High Confidence",
        "pre": "ChromaDB contains a directly relevant PDF. scoring_service returns 0.87.",
        "inp": "question = 'Define machine learning'\nretrieved_passages = [doc1, doc2, doc3]",
        "cond":"A = False  |  B = True  (0.87 >= 0.8)",
        "out": "confidence_score = 0.87\nconfidence_label = 'High - Verified'\ncitations = [citation list]\nexplanation = set by scoring_service",
        "hit": "S1, S7, S8, S9",
    },
    {
        "id":"TC-3","path":"P3","name":"Partial Document Match — Medium Confidence",
        "pre": "ChromaDB contains a partially relevant PDF. scoring_service returns 0.65.",
        "inp": "question = 'Explain neural networks'\nretrieved_passages = [doc1]",
        "cond":"A = False  |  B = False (0.65 < 0.8)  |  C = True (0.65 >= 0.5)",
        "out": "confidence_score = 0.65\nconfidence_label = 'Medium - Verified'\ncitations = [citation list]\nexplanation = set by scoring_service",
        "hit": "S1, S7, S8, S10, S11",
    },
    {
        "id":"TC-4","path":"P4","name":"Off-Topic Query — Low Confidence",
        "pre": "ChromaDB contains an unrelated PDF. scoring_service returns 0.30.",
        "inp": "question = 'What is the capital of France?'\nretrieved_passages = [doc1]",
        "cond":"A = False  |  B = False (0.30 < 0.8)  |  C = False (0.30 < 0.5)",
        "out": "confidence_score = 0.30\nconfidence_label = 'Low - Unverified'\ncitations = [citation list]\nexplanation = set by scoring_service",
        "hit": "S1, S7, S8, S10, S12",
    },
]

for tc in TESTS:
    story.append(KeepTogether([
        Paragraph(f"{tc['id']}:  {tc['name']}  (Path {tc['path']})", H2),
        kv_tbl([
            ["Preconditions",    tc["pre"]],
            ["Inputs",           tc["inp"]],
            ["Condition Values", tc["cond"]],
            ["Expected Output",  tc["out"]],
            ["Statements Hit",   tc["hit"]],
        ]),
        Spacer(1, 0.08*inch),
    ]))

# ── 5. TRUTH TABLES ───────────────────────────────────────────────────────
story.append(PageBreak())
story += [Paragraph("5. Truth Tables", H1), rule()]

story.append(Paragraph("5.1  Basic Conditions Truth Table", H2))
story.append(make_tbl([
    ["Test", "A: not passages", "B: score >= 0.8", "C: score >= 0.5", "Label Outcome",        "Path"],
    ["TC-1", "T",               "--",              "--",              "Unverified - No Data", "P1"],
    ["TC-2", "F",               "T",               "--",              "High - Verified",      "P2"],
    ["TC-3", "F",               "F",               "T",               "Medium - Verified",    "P3"],
    ["TC-4", "F",               "F",               "F",               "Low - Unverified",     "P4"],
], [0.55*inch, 1.15*inch, 1.2*inch, 1.2*inch, 1.9*inch, 0.5*inch]))
story.append(Paragraph("'--' means the condition is not evaluated (short-circuit logic).", SM))
story.append(Spacer(1, 0.08*inch))

story.append(Paragraph("5.2  Compound Conditions Truth Table", H2))
story.append(Paragraph(
    "Each predicate in this block is atomic. Compound expressions arise from the if/elif/else chain:", BODY))
story.append(make_tbl([
    ["Compound Expression",                   "TC-1","TC-2","TC-3","TC-4","T & F Covered?"],
    ["A  (not retrieved_passages)",           "T",   "F",   "F",   "F",  "Yes"],
    ["A = F  AND  B  (score >= 0.8)",           "--",  "T",   "F",   "F",  "Yes"],
    ["A = F AND NOT B AND C  (score >= 0.5)",   "--",  "--",  "T",   "F",  "Yes"],
], [2.55*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.65*inch, 1.25*inch]))
story.append(Spacer(1, 0.08*inch))

story.append(Paragraph("5.3  MC/DC Independence Pairs", H2))
story.append(make_tbl([
    ["Cond.", "Test Pair",    "What Changes","Other Conditions Fixed",          "Outcomes Differ?"],
    ["A",     "TC-1 vs TC-2","A: T -> F",   "B = T, C = T  (implied by score = 0.87)","Yes: Unverified -> High"],
    ["B",     "TC-2 vs TC-3","B: T -> F",   "A = False, C = True",                 "Yes: High -> Medium"],
    ["C",     "TC-3 vs TC-4","C: T -> F",   "A = False, B = False",                "Yes: Medium -> Low"],
], [0.55*inch, 1.2*inch, 0.95*inch, 2.1*inch, 2.0*inch]))

# ── 6. COVERAGE METRICS ───────────────────────────────────────────────────
story.append(PageBreak())
story += [Paragraph("6. Coverage Metrics", H1), rule()]

# FIX: use Paragraph objects in cells so long text wraps within column width
# Column widths:  0.3 + 1.7 + 2.05 + 1.65 + 1.1  =  6.8 inch  (= USABLE)
W1, W2, W3, W4, W5 = 0.3*inch, 1.7*inch, 2.05*inch, 1.65*inch, 1.1*inch

def pc(txt, bold=False):
    st = S('pc_b', parent=CELL, fontName='Helvetica-Bold') if bold else CELL
    return Paragraph(txt, st)

def ph(txt):
    return Paragraph(txt, S('ph', parent=CELL, fontName='Helvetica-Bold',
                             textColor=colors.white))

metrics_data = [
    # Header row — plain strings (styled via TableStyle)
    ["#", "Coverage Type", "Formula", "Calculation", "Score"],
    # Data rows — Paragraph objects for word-wrap
    [pc("1"), pc("All Path\nTesting"),
     pc("# paths covered /\n# total paths"),
     pc("4 covered / 4 total"), pc("4/4\n= 100%")],

    [pc("2"), pc("Statement\nCoverage"),
     pc("# statements covered /\n# total statements"),
     pc("12 covered / 12 total"), pc("12/12\n= 100%")],

    [pc("3"), pc("Segment\nCoverage"),
     pc("# segments covered /\n# total segments"),
     pc("12 covered / 12 total"), pc("12/12\n= 100%")],

    [pc("4"), pc("Branch\nCoverage"),
     pc("# executed branches /\n# total branches"),
     pc("6 taken / 6 total"), pc("6/6\n= 100%")],

    [pc("5"), pc("Basic Condition\nCoverage"),
     pc("# truth values /\n(2 x # basic conditions)"),
     pc("6 values / (2 x 3 = 6)"), pc("6/6\n= 100%")],

    [pc("6"), pc("Compound Condition\nCoverage"),
     pc("# truth values /\n(2 x # compound conditions)"),
     pc("6 values / (2 x 3 = 6)"), pc("6/6\n= 100%")],

    [pc("7"), pc("MC/DC\nCoverage"),
     pc("# non-redundant combos /\n(N+1) required"),
     pc("4 tests / (3+1 = 4)\nrequired"), pc("4/4\n= 100%")],
]

metrics_tbl = Table(metrics_data, colWidths=[W1, W2, W3, W4, W5], repeatRows=1)
metrics_tbl.setStyle(TableStyle([
    ('BACKGROUND',    (0, 0), (-1,  0), TBLHDR),
    ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
    ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0, 0), (-1, -1), 8.5),
    ('LEADING',       (0, 0), (-1, -1), 12),
    ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, TBLALT]),
    ('GRID',          (0, 0), (-1, -1), 0.35, colors.HexColor('#aaaaaa')),
    ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING',    (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('LEFTPADDING',   (0, 0), (-1, -1), 6),
    ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
]))
story.append(metrics_tbl)
story.append(Spacer(1, 0.08*inch))

story.append(Paragraph(
    "<b>Note — Compound Condition Coverage:</b> Each predicate is atomic. "
    "Compound expressions are formed by the implicit AND chains in the if/elif/else structure "
    "(see Section 5.2). All 3 compound expressions take both True and False values across the "
    "4 test cases, satisfying all 6 required truth values.", SM))
story.append(Spacer(1, 0.05*inch))
story.append(Paragraph(
    "<b>Note — MC/DC:</b> With 3 independent conditions (A, B, C), the minimum required "
    "test set size is N+1 = 4. The 4 test cases each form a unique independence pair "
    "(one per condition), confirming each condition independently affects the outcome.", SM))

# ── 7. JUSTIFICATION ──────────────────────────────────────────────────────
story += [Spacer(1, 0.05*inch), Paragraph("7. Justification for Test Suite", H1), rule()]

items = [
    ("Why this code section?",
     "The confidence label assignment block is the most user-facing decision logic in Confid.AI. "
     "It determines what label (High, Medium, Low, or Unverified) is shown for every AI-generated "
     "answer, drives the admin flagging system, and triggers the red warning banner. Incorrect "
     "labelling would directly mislead users, making this the highest-priority section to test."),
    ("Why exactly 4 tests?",
     "The section has exactly 4 feasible execution paths, 3 independent conditions, no loops, and "
     "no dead code. One test per path is both the minimum and the maximum needed: it achieves "
     "100% coverage on all seven metrics simultaneously, with no redundant test."),
    ("Why MC/DC?",
     "MC/DC confirms that each individual condition independently changes the outcome, ensuring "
     "no logic is redundant. It is the gold standard for decision-intensive code and is mandated "
     "in safety-critical standards (DO-178C, IEC 61508)."),
    ("Suggested boundary tests for production use",
     "TC-5: confidence_score = 0.8 exactly (boundary for Condition B).  "
     "TC-6: confidence_score = 0.5 exactly (boundary for Condition C).  "
     "TC-7: retrieved_passages = None instead of [ ] (null-safety check)."),
]
for hdr, txt in items:
    story.append(KeepTogether([
        Paragraph(hdr, BOLD),
        Paragraph(txt, BODY),
        Spacer(1, 0.04*inch),
    ]))

doc.build(story)
print("PDF built: output/Confid_AI_Testing_Final.pdf")