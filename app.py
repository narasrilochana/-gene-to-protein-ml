
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gene to Protein | Bioinformatics Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
ASSET_DIR = ROOT / "assets"

COLORS = {
    "deep": "#0A3323",
    "midnight": "#105666",
    "beige": "#F7F4D5",
    "rose": "#D3968C",
    "moss": "#839958",
    "ink": "#0A3323",
}

# ============================================================
# GLOBAL CSS AND RESPONSIVE DESIGN
# Pinterest integration:
# Add inspiration images to assets/ and adjust this CSS section
# to incorporate their layout, borders, patterns and typography.
# ============================================================
st.markdown(
    f"""
    <style>
    html, body, [class*="css"], [data-testid], button, input,
    textarea, select, label, p, span, div, h1, h2, h3, h4,
    h5, h6 {{
        font-family: "Times New Roman", Times, serif !important;
    }}

    .stApp {{
        background: {COLORS["deep"]};
        color: {COLORS["beige"]};
    }}

    [data-testid="stHeader"] {{
        background: {COLORS["deep"]};
    }}

    [data-testid="stSidebar"] {{
        background: {COLORS["midnight"]};
        border-right: 2px solid {COLORS["rose"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {COLORS["beige"]} !important;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    .hero {{
        background: {COLORS["midnight"]};
        border: 1px solid {COLORS["rose"]};
        border-radius: 18px;
        padding: clamp(1.2rem, 4vw, 3rem);
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 28px rgba(0,0,0,.18);
    }}

    .kicker {{
        color: {COLORS["rose"]};
        text-transform: uppercase;
        letter-spacing: .28em;
        font-size: .78rem;
        font-weight: bold;
        margin-bottom: .8rem;
    }}

    .hero-title {{
        color: {COLORS["beige"]};
        font-size: clamp(2.1rem, 5vw, 4.4rem);
        line-height: 1.02;
        font-style: italic;
        margin: 0;
        font-weight: 500;
        overflow-wrap: anywhere;
    }}

    .hero-subtitle {{
        color: {COLORS["beige"]};
        font-size: 1.05rem;
        margin-top: .8rem;
    }}

    .hero-rule {{
        height: 2px;
        background: linear-gradient(
            90deg, {COLORS["rose"]}, {COLORS["moss"]}, transparent
        );
        margin-top: 1.4rem;
    }}

    .section-label {{
        color: {COLORS["rose"]};
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .78rem;
        font-weight: bold;
    }}

    .info-card {{
        background: {COLORS["midnight"]};
        color: {COLORS["beige"]};
        border: 1px solid {COLORS["moss"]};
        border-radius: 14px;
        padding: 1rem 1.15rem;
        min-height: 105px;
        overflow-wrap: anywhere;
    }}

    .info-card .label {{
        color: {COLORS["rose"]};
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .1em;
    }}

    .info-card .value {{
        color: {COLORS["beige"]};
        font-size: clamp(1.35rem, 2.8vw, 2rem);
        font-weight: bold;
        margin-top: .25rem;
    }}

    .sequence-box {{
        background: {COLORS["beige"]};
        color: {COLORS["ink"]};
        border: 1px solid {COLORS["rose"]};
        border-radius: 12px;
        padding: 1rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        word-break: break-word;
        line-height: 1.65;
    }}

    .small-note {{
        color: {COLORS["beige"]};
        opacity: .88;
        font-size: .92rem;
    }}

    div[data-testid="stMetric"] {{
        background: {COLORS["midnight"]};
        border: 1px solid {COLORS["moss"]};
        border-radius: 12px;
        padding: 12px;
    }}

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS["beige"]} !important;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        background: {COLORS["rose"]};
        color: {COLORS["deep"]};
        border: 1px solid {COLORS["beige"]};
        border-radius: 9px;
        font-weight: bold;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: {COLORS["moss"]};
        color: {COLORS["deep"]};
    }}

    input, textarea, [data-baseweb="select"] > div {{
        background: {COLORS["beige"]} !important;
        color: {COLORS["deep"]} !important;
    }}

    [data-testid="stTabs"] button {{
        color: {COLORS["beige"]} !important;
        font-size: 1rem;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background: {COLORS["beige"]};
        color: {COLORS["deep"]};
        border-radius: 10px;
        overflow: hidden;
    }}

    img {{
        max-width: 100%;
        height: auto;
    }}

    @media (max-width: 700px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .hero {{
            padding: 1.25rem;
        }}

        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STANDARD GENETIC CODE
# ============================================================
CODON_TABLE: Dict[str, str] = {
    "UUU":"F","UUC":"F","UUA":"L","UUG":"L",
    "UCU":"S","UCC":"S","UCA":"S","UCG":"S",
    "UAU":"Y","UAC":"Y","UAA":"*","UAG":"*",
    "UGU":"C","UGC":"C","UGA":"*","UGG":"W",
    "CUU":"L","CUC":"L","CUA":"L","CUG":"L",
    "CCU":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAU":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "CGU":"R","CGC":"R","CGA":"R","CGG":"R",
    "AUU":"I","AUC":"I","AUA":"I","AUG":"M",
    "ACU":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAU":"N","AAC":"N","AAA":"K","AAG":"K",
    "AGU":"S","AGC":"S","AGA":"R","AGG":"R",
    "GUU":"V","GUC":"V","GUA":"V","GUG":"V",
    "GCU":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAU":"D","GAC":"D","GAA":"E","GAG":"E",
    "GGU":"G","GGC":"G","GGA":"G","GGG":"G",
}

# ============================================================
# REFERENCE PROTEIN DATASET: 30 ENTRIES
# These are educational representative sequences/segments.
# Some large proteins are represented by fragments.
# ============================================================
PROTEINS = {
    "Human Insulin": {
        "class": "Hormone / Signaling",
        "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    },
    "Glucagon": {
        "class": "Hormone / Signaling",
        "sequence": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"
    },
    "Somatostatin": {
        "class": "Hormone / Signaling",
        "sequence": "AGCKNFFWKTFTSC"
    },
    "Oxytocin": {
        "class": "Hormone / Signaling",
        "sequence": "CYIQNCPLG"
    },
    "Vasopressin": {
        "class": "Hormone / Signaling",
        "sequence": "CYFQNCPRG"
    },
    "Leptin": {
        "class": "Hormone / Signaling",
        "sequence": "VPIQKVQDDTKTLIKTIVTRINDISHTQSVSSKQKVTGLDFIPGLHP"
    },
    "Erythropoietin": {
        "class": "Hormone / Signaling",
        "sequence": "APPRLICDSRVLERYLLEAKEAENITTGCAEHCSLNENITVP"
    },
    "Hemoglobin Alpha": {
        "class": "Transport / Structural",
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVK"
    },
    "Hemoglobin Beta": {
        "class": "Transport / Structural",
        "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAV"
    },
    "Myoglobin": {
        "class": "Transport / Structural",
        "sequence": "GLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLK"
    },
    "Albumin": {
        "class": "Transport / Structural",
        "sequence": "DAHKSEVAHRFKDLGEQHFKGLVLIAFSQYLQQCPFDEHVKLVNEVTEFAK"
    },
    "Collagen": {
        "class": "Transport / Structural",
        "sequence": "GPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPP"
    },
    "Keratin": {
        "class": "Transport / Structural",
        "sequence": "QSTSSSRSYTSGPGSRISSSSRSYTSGPGSRISSSS"
    },
    "Actin": {
        "class": "Transport / Structural",
        "sequence": "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQK"
    },
    "Tubulin": {
        "class": "Transport / Structural",
        "sequence": "MREIVHIQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLER"
    },
    "Ferritin": {
        "class": "Transport / Structural",
        "sequence": "MSQIRQNYSTKDSVNKFLDLVAQFSK"
    },
    "p53": {
        "class": "Regulatory / Immune",
        "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFT"
    },
    "BRCA1": {
        "class": "Regulatory / Immune",
        "sequence": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLK"
    },
    "Histone H4": {
        "class": "Regulatory / Immune",
        "sequence": "MSGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGLI"
    },
    "Ubiquitin": {
        "class": "Regulatory / Immune",
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    },
    "Calmodulin": {
        "class": "Regulatory / Immune",
        "sequence": "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKE"
    },
    "Transferrin": {
        "class": "Transport / Structural",
        "sequence": "VPDKTVRWCAVSEHEATKCQSFRDHMKSVIPSDGPSVACVKKASYLDCI"
    },
    "Cytochrome C": {
        "class": "Enzyme / Catalytic",
        "sequence": "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEY"
    },
    "Lysozyme": {
        "class": "Enzyme / Catalytic",
        "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDG"
    },
    "GFP": {
        "class": "Enzyme / Catalytic",
        "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"
    },
    "lacZ": {
        "class": "Enzyme / Catalytic",
        "sequence": "MTHKLLVTLKQKQKQKQGIPV"
    },
    "Amylase": {
        "class": "Enzyme / Catalytic",
        "sequence": "DNGRTTQYQVVWPGDNQ"
    },
    "Pepsinogen": {
        "class": "Enzyme / Catalytic",
        "sequence": "PQPLWILGFAD"
    },
    "Taq Polymerase": {
        "class": "Enzyme / Catalytic",
        "sequence": "MELTQFQK"
    },
    "SARS-CoV-2 Spike RBD": {
        "class": "Regulatory / Immune",
        "sequence": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCV"
    },
}

for item in PROTEINS.values():
    item["sequence"] = re.sub(r"\s+", "", item["sequence"]).upper()

# ============================================================
# CHEMICAL RESIDUE GROUPS
# ============================================================
HYDROPHOBIC = set("VLIWFYCM")
HYDROPHILIC = set("STNQ")
CHARGED = set("DEKRH")

# Deterministic reverse translation.
# One valid DNA codon is selected for each amino acid.
AA_TO_DNA = {
    "A":"GCT", "R":"CGT", "N":"AAT", "D":"GAT",
    "C":"TGT", "Q":"CAA", "E":"GAA", "G":"GGT",
    "H":"CAT", "I":"ATT", "L":"CTG", "K":"AAA",
    "M":"ATG", "F":"TTT", "P":"CCT", "S":"TCT",
    "T":"ACT", "W":"TGG", "Y":"TAT", "V":"GTT",
}

# ============================================================
# DNA VALIDATION AND PROCESSING
# ============================================================
def clean_and_validate_dna(raw: str) -> str:
    cleaned = re.sub(r"\s+", "", raw).upper()

    if not cleaned:
        raise ValueError("Please enter a DNA sequence.")

    invalid = sorted(set(cleaned) - set("ATCG"))

    if invalid:
        raise ValueError(
            "Invalid character(s): " + ", ".join(invalid)
            + ". DNA can contain only A, T, C and G."
        )

    if len(cleaned) < 3:
        raise ValueError(
            "The DNA sequence must contain at least 3 bases."
        )

    return cleaned


def transcribe_dna(dna: str) -> str:
    return dna.replace("T", "U")


def translate_mrna(mrna: str) -> Tuple[str, bool, int]:
    amino_acids = []
    stop_found = False
    complete_codons = len(mrna) // 3

    for index in range(0, complete_codons * 3, 3):
        codon = mrna[index:index + 3]
        amino_acid = CODON_TABLE.get(codon)

        if amino_acid == "*":
            stop_found = True
            break

        if amino_acid is None:
            break

        amino_acids.append(amino_acid)

    return (
        "".join(amino_acids),
        stop_found,
        len(mrna) % 3
    )


# ============================================================
# CHEMICAL PROPERTY CALCULATIONS
# ============================================================
def calculate_features(dna: str, protein: str) -> Dict[str, float]:
    length = max(len(protein), 1)

    return {
        "GC Content (%)":
            100 * (dna.count("G") + dna.count("C")) / max(len(dna), 1),
        "Hydrophobic Ratio":
            sum(aa in HYDROPHOBIC for aa in protein) / length,
        "Hydrophilic Ratio":
            sum(aa in HYDROPHILIC for aa in protein) / length,
        "Charged Ratio":
            sum(aa in CHARGED for aa in protein) / length,
        "Sequence Length": float(len(protein)),
    }


def residue_counts(protein: str) -> Dict[str, int]:
    return {
        "Hydrophobic": sum(aa in HYDROPHOBIC for aa in protein),
        "Hydrophilic": sum(aa in HYDROPHILIC for aa in protein),
        "Charged": sum(aa in CHARGED for aa in protein),
        "Other": sum(
            aa not in HYDROPHOBIC | HYDROPHILIC | CHARGED
            for aa in protein
        ),
    }


def make_reference_dna(protein_sequence: str) -> str:
    return "".join(AA_TO_DNA[aa] for aa in protein_sequence)


def feature_row(dna: str, protein: str) -> List[float]:
    values = calculate_features(dna, protein)

    return [
        values["GC Content (%)"],
        values["Hydrophobic Ratio"],
        values["Hydrophilic Ratio"],
        values["Charged Ratio"],
        values["Sequence Length"],
    ]


# ============================================================
# RANDOM FOREST MODEL
# ============================================================
@st.cache_resource
def train_model():
    rows = []
    labels = []

    for item in PROTEINS.values():
        sequence = item["sequence"]
        dna = make_reference_dna(sequence)

        rows.append(feature_row(dna, sequence))
        labels.append(item["class"])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(rows, labels)

    predictions = model.predict(rows)
    accuracy = accuracy_score(labels, predictions)

    benchmark = pd.DataFrame({
        "Protein": list(PROTEINS.keys()),
        "True Class": labels,
        "Predicted Class": predictions,
        "Correct": [
            actual == predicted
            for actual, predicted in zip(labels, predictions)
        ],
        "Sequence Length": [
            len(item["sequence"]) for item in PROTEINS.values()
        ],
    })

    return model, accuracy, benchmark


model, training_accuracy, benchmark_df = train_model()

# ============================================================
# EDITORIAL HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="kicker">
            A BIOINFORMATICS • SEQUENCE ANALYSIS • MACHINE LEARNING
        </div>
        <h1 class="hero-title">From DNA to Protein</h1>
        <div class="hero-subtitle">
            An automated sequence translator and
            machine-learning protein classifier
        </div>
        <div class="hero-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Bioinformatics Studio")

    st.write(
        "Explore how a DNA coding sequence can be transcribed, "
        "translated and profiled."
    )

    st.markdown("---")
    st.markdown("**Reference collection**")

    st.metric("Protein entries", len(PROTEINS))
    st.metric(
        "Biological classes",
        len(set(item["class"] for item in PROTEINS.values()))
    )

    st.markdown("---")
    st.caption("Classroom prototype • Standard genetic code")
    st.caption(
        "Some large proteins are represented by sequence segments."
    )

# ============================================================
# UPLOADED BIOLOGY ILLUSTRATIONS
# ============================================================
asset_names = [
    "dna_helix.png",
    "petri_dish.png",
    "molecule.png",
    "microscope.png",
]

image_cols = st.columns(4)

for col, filename, caption in zip(
    image_cols,
    asset_names,
    ["DNA double helix", "Cell culture",
     "Molecular structure", "Microscopy"]
):
    image_path = ASSET_DIR / filename

    if image_path.exists():
        with col:
            st.image(
                str(image_path),
                caption=caption,
                use_container_width=True
            )

# ============================================================
# MAIN TABS
# ============================================================
tab_workspace, tab_benchmark = st.tabs([
    "🧬 Sequence Workspace & ML Predictor",
    "📊 30-Protein ML Training Benchmark"
])

# ============================================================
# TAB 1: SEQUENCE WORKSPACE AND PREDICTOR
# ============================================================
with tab_workspace:

    st.markdown(
        '<div class="section-label">01 / Sequence workspace</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Choose a reference protein or analyse your own DNA"
    )

    source_mode = st.radio(
        "Sequence source",
        ["Reference protein", "Custom DNA sequence"],
        horizontal=True
    )

    selected_name = None

    if source_mode == "Reference protein":

        selected_name = st.selectbox(
            "Select a reference protein",
            list(PROTEINS.keys())
        )

        selected_protein = PROTEINS[selected_name]["sequence"]
        dna_input = make_reference_dna(selected_protein)

        st.info(
            "A valid coding-DNA demonstration sequence has been "
            "generated using synonymous codons. It is not the "
            "native gene sequence."
        )

        st.text_area(
            "Generated DNA coding sequence",
            dna_input,
            height=100,
            disabled=True
        )

    else:

        dna_input = st.text_area(
            "Paste your DNA sequence",
            placeholder="Example: ATGGCTGCTGAA...",
            height=150,
            help="Whitespace is ignored. Use only A, T, C and G."
        )

    analyse = st.button(
        "Analyse DNA sequence",
        type="primary"
    )

    if analyse:

        try:
            dna = clean_and_validate_dna(dna_input)
            mrna = transcribe_dna(dna)

            protein, stop_found, trailing_bases = translate_mrna(mrna)

            if not protein:
                st.error(
                    "No amino acids were translated. "
                    "Check that the sequence contains a valid codon."
                )

            else:

                features = calculate_features(dna, protein)

                row = feature_row(dna, protein)

                prediction = model.predict([row])[0]
                probabilities = model.predict_proba([row])[0]

                confidence = float(max(probabilities) * 100)

                st.session_state["analysis_result"] = {
                    "dna": dna,
                    "mrna": mrna,
                    "protein": protein,
                    "features": features,
                    "counts": residue_counts(protein),
                    "prediction": prediction,
                    "confidence": confidence,
                    "stop_found": stop_found,
                    "trailing_bases": trailing_bases,
                    "selected_name": selected_name,
                }

        except ValueError as error:
            st.session_state.pop("analysis_result", None)
            st.error(str(error))

    result = st.session_state.get("analysis_result")

    if result:

        st.markdown("---")

        st.markdown(
            '<div class="section-label">02 / Analysis results</div>',
            unsafe_allow_html=True
        )

        # KPI METRIC CARDS
        kpi_cols = st.columns(4)

        kpis = [
            ("DNA Length", f'{len(result["dna"]):,} bp'),
            (
                "GC Content",
                f'{result["features"]["GC Content (%)"]:.2f}%'
            ),
            ("Protein Length", f'{len(result["protein"]):,} aa'),
            ("ML Confidence", f'{result["confidence"]:.1f}%'),
        ]

        for col, (label, value) in zip(kpi_cols, kpis):

            with col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="label">{label}</div>
                        <div class="value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ML PREDICTION
        st.markdown("### Predicted biological function")

        st.success(f'**{result["prediction"]}**')

        st.caption(
            "Confidence is the highest Random Forest class probability. "
            "It is not a calibrated probability of biological function."
        )

        if result["stop_found"]:
            st.info("Translation stopped at the first in-frame stop codon.")
        else:
            st.info("No in-frame stop codon was found.")

        if result["trailing_bases"]:
            st.warning(
                f'{result["trailing_bases"]} trailing base(s) '
                "did not form a complete codon and were not translated."
            )

        # mRNA AND PROTEIN OUTPUT
        left, right = st.columns(2)

        with left:
            st.markdown("#### Transcribed mRNA")

            st.markdown(
                f'<div class="sequence-box">{result["mrna"]}</div>',
                unsafe_allow_html=True
            )

        with right:
            st.markdown("#### Translated protein chain")

            st.markdown(
                f'<div class="sequence-box">{result["protein"]}</div>',
                unsafe_allow_html=True
            )

        # CHEMICAL RESIDUE CHART
        st.markdown("#### Chemical residue profile")

        counts = result["counts"]

        fig, ax = plt.subplots(figsize=(8, 3.6))

        bars = ax.bar(
            list(counts.keys()),
            list(counts.values()),
            color=[
                COLORS["rose"],
                COLORS["moss"],
                "#6AAE9B",
                "#C9C6A1"
            ],
            edgecolor=COLORS["deep"],
            linewidth=1.1
        )

        ax.set_title(
            "Amino-acid residue groups",
            fontname="Times New Roman",
            color=COLORS["deep"],
            fontsize=15
        )

        ax.set_ylabel(
            "Residue count",
            fontname="Times New Roman",
            color=COLORS["deep"]
        )

        ax.set_xlabel(
            "Chemical grouping",
            fontname="Times New Roman",
            color=COLORS["deep"]
        )

        ax.tick_params(axis="x", colors=COLORS["deep"])
        ax.tick_params(axis="y", colors=COLORS["deep"])

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")

        for bar, value in zip(bars, counts.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontname="Times New Roman",
                color=COLORS["deep"]
            )

        fig.patch.set_facecolor(COLORS["beige"])
        ax.set_facecolor(COLORS["beige"])
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # NUMERICAL FEATURE TABLE
        feature_df = pd.DataFrame([
            {"Feature": key, "Value": value}
            for key, value in result["features"].items()
        ])

        with st.expander("View chemical-property measurements"):
            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

        # DOWNLOAD OUTPUT
        download_text = (
            f'>DNA\n{result["dna"]}\n\n'
            f'>mRNA\n{result["mrna"]}\n\n'
            f'>Protein\n{result["protein"]}\n'
        )

        st.download_button(
            "Download sequence analysis (.fasta)",
            data=download_text,
            file_name="sequence_analysis.fasta",
            mime="text/plain"
        )

# ============================================================
# TAB 2: ML TRAINING BENCHMARK
# ============================================================
with tab_benchmark:

    st.markdown(
        '<div class="section-label">03 / Model benchmark</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Reference dataset and Random Forest training performance"
    )

    metric_cols = st.columns(3)

    with metric_cols[0]:
        st.metric("Dataset size", f"{len(PROTEINS)} proteins")

    with metric_cols[1]:
        st.metric("Algorithm", "Random Forest")

    with metric_cols[2]:
        st.metric(
            "Training accuracy",
            f"{training_accuracy * 100:.1f}%"
        )

    st.markdown(
        """
        **Model inputs:** GC content, hydrophobic ratio,
        hydrophilic ratio, charged-residue ratio and sequence length.

        **Model settings:** 100 decision trees
        (`n_estimators=100`) and `random_state=42`.
        """
    )

    st.warning(
        "This accuracy is measured on the same 30 entries used "
        "to train the model. It is an in-sample training score, "
        "not an independent test score."
    )

    st.markdown("#### True class vs. predicted class")

    show_only_errors = st.checkbox(
        "Show only misclassified entries",
        value=False
    )

    if show_only_errors:
        display_df = benchmark_df.loc[~benchmark_df["Correct"]]
    else:
        display_df = benchmark_df

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Correct": st.column_config.CheckboxColumn(
                "Correct prediction"
            ),
            "Sequence Length": st.column_config.NumberColumn(
                "Sequence length (aa)",
                format="%d"
            ),
        }
    )

    st.markdown("#### Dataset distribution")

    distribution = (
        benchmark_df["True Class"]
        .value_counts()
        .rename_axis("Biological class")
        .reset_index(name="Protein count")
    )

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True
    )

    csv = benchmark_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download benchmark table (.csv)",
        data=csv,
        file_name="protein_classifier_benchmark.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown(
    """
    <p class="small-note">
        BIOINFORMATICS STUDIO • DNA → mRNA → PROTEIN
        • RANDOM FOREST CLASSIFICATION
    </p>
    """,
    unsafe_allow_html=True
)
