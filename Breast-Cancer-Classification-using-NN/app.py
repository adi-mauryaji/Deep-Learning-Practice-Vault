import json
from pathlib import Path

import altair as alt
import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

# Colours, also in style.css. The malignant / benign pair passes a colour-blind check on the dark surface.
TEXT = "#E7ECF7"
MUTED = "#8D97AE"
GRID = "#1E2740"
ACCENT = "#A99BFF"
EDGE = "#C9D3EA"
MALIGNANT_COLOR = "#EC5E3E"
BENIGN_COLOR = "#3896EC"

REPO = "https://github.com/adi-mauryaji/Deep-Learning-Practice-Vault"
folder = Path(__file__).parent

st.set_page_config(page_title="Breast Cancer · Neural Network", page_icon="🧬", layout="wide")
st.html(folder / "style.css")

# The data measures 10 things about each cell nucleus, and records each one 3 ways
MEASUREMENTS = {
    "radius": "Mean distance from the centre to points on the edge.",
    "texture": "Standard deviation of the grey-scale values.",
    "perimeter": "Length of the edge.",
    "area": "Area inside the edge.",
    "smoothness": "Local variation in radius lengths.",
    "compactness": "Perimeter² ÷ area − 1.",
    "concavity": "How deep the inward dents in the edge are.",
    "concave points": "How many inward dents the edge has.",
    "symmetry": "How symmetric the nucleus is.",
    "fractal dimension": "How jagged the edge is.",
}
GROUPS = {
    "Mean": ("mean {}", "Average over all the nuclei in the image."),
    "Worst": ("worst {}", "Average of the 3 largest values in the image."),
    "Standard error": ("{} error", "How much the value varies between nuclei."),
}
NAMES = {0: "malignant", 1: "benign"}


# Load everything once, not on every slider move
@st.cache_resource
def load_files():
    model = keras.saving.load_model(folder / "model.keras")
    scaler = joblib.load(folder / "scaler.pkl")
    report = json.loads((folder / "report.json").read_text())
    dataset = load_breast_cancer()  # ships with scikit-learn, no download
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    return model, scaler, report, data, dataset.target


model, scaler, report, data, target = load_files()
hidden_layer, output_layer = model.layers
W1, b1 = hidden_layer.get_weights()  # 30 x 20
W2, b2 = output_layer.get_weights()  # 20 x 2


def forward(scaled_rows):
    """The same network worked out layer by layer, so the diagram can show every neuron."""
    hidden = np.maximum(0, scaled_rows @ W1 + b1)  # Dense(20, relu)
    logits = hidden @ W2 + b2  # Dense(2) before softmax
    return hidden, logits


@st.cache_data
def quick_picks():
    """Four test cases worth showing: two easy calls, a close call and a mistake."""
    ids = np.array(report["test_ids"])
    chances = keras.ops.convert_to_numpy(model(scaler.transform(data.loc[ids].to_numpy()), training=False))[:, 0]
    actual = target[ids]
    predicted = np.where(chances >= 0.5, 0, 1)
    wrong = (predicted != actual)
    missed = wrong & (actual == 0)  # malignant called benign, the costly kind of mistake
    return {
        "Clear malignant": int(ids[np.argmax(np.where(actual == 0, chances, -1))]),
        "Clear benign": int(ids[np.argmin(np.where(actual == 1, chances, 2))]),
        "Close call": int(ids[np.argmin(abs(chances - 0.5))]),
        "A mistake": int(ids[missed if missed.any() else wrong][0]),
    }


def choose(row):
    st.session_state.record = row


def reset(row):
    for column in data.columns:
        st.session_state.pop(f"{row}-{column}", None)


def label(column):
    return column[0].upper() + column[1:]


def svg_html(markup):
    """st.html strips SVG, but st.markdown keeps it. One line, so markdown passes it through untouched."""
    st.markdown(" ".join(markup.split()), unsafe_allow_html=True)


def percent(chance):
    if chance > 0.999:
        return "&gt;99.9%"
    if chance < 0.001:
        return "&lt;0.1%"
    return f"{chance:.1%}"


def donut(confidence, color):
    radius = 46
    circle = 2 * np.pi * radius
    return f"""
    <svg class="donut" viewBox="0 0 120 120" role="img" aria-label="{confidence:.1%} sure">
      <circle cx="60" cy="60" r="{radius}" class="donut-track"/>
      <circle cx="60" cy="60" r="{radius}" class="donut-arc" stroke="{color}"
        stroke-dasharray="{circle * confidence:.1f} {circle:.1f}" transform="rotate(-90 60 60)"/>
      <text x="60" y="62" class="donut-value">{percent(confidence)}</text>
      <text x="60" y="80" class="donut-label">sure</text>
    </svg>"""


def network_svg(raw_row, scaled_row, hidden, malignant_chance):
    """Draw 30 inputs → 20 hidden neurons → 2 outputs. Brighter lines carry more signal."""
    width, height = 680, 430
    in_x, hidden_x, out_x = 70, 310, 520
    in_y = np.linspace(44, height - 16, 30)
    hidden_y = np.linspace(60, height - 32, 20)
    out_y = [height / 2 - 50, height / 2 + 70]

    # Signal on each line = value coming in x weight of the line
    signal_in = np.abs(scaled_row[:, None] * W1)
    signal_in /= signal_in.max() or 1
    alive = hidden > 0
    signal_out = np.abs(hidden[:, None] * W2)
    signal_out /= signal_out.max() or 1
    activation = hidden / (hidden.max() or 1)

    parts = [f'<svg class="net" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Diagram of the neural network for this sample">']
    for x, anchor, text in [(20, "start", "30 inputs"), (hidden_x, "middle", "20 hidden · ReLU"),
                            (width - 4, "end", "2 outputs · softmax")]:
        parts.append(f'<text x="{x}" y="18" text-anchor="{anchor}" class="net-head">{text}</text>')

    # Lines into a switched-off neuron are dimmed, because ReLU stops their signal there
    parts.append(f'<g stroke="{EDGE}" stroke-linecap="round">')
    for i in range(30):
        for j in range(20):
            strength = signal_in[i, j] * (1 if alive[j] else 0.2)
            if strength < 0.04:
                continue  # too faint to see
            parts.append(f'<line x1="{in_x}" y1="{in_y[i]:.1f}" x2="{hidden_x}" y2="{hidden_y[j]:.1f}" '
                         f'stroke-opacity="{0.04 + 0.55 * strength:.2f}" stroke-width="{0.5 + 1.4 * strength:.2f}"/>')
    parts.append("</g>")
    for k, color in enumerate([MALIGNANT_COLOR, BENIGN_COLOR]):
        parts.append(f'<g stroke="{color}" stroke-linecap="round">')
        for j in range(20):
            strength = signal_out[j, k]
            if strength < 0.03:
                continue
            parts.append(f'<line x1="{hidden_x}" y1="{hidden_y[j]:.1f}" x2="{out_x}" y2="{out_y[k]:.1f}" '
                         f'stroke-opacity="{0.1 + 0.8 * strength:.2f}" stroke-width="{0.6 + 2.6 * strength:.2f}"/>')
        parts.append("</g>")

    # Brackets for the 3 groups of 10 inputs
    for number, name in enumerate(["mean", "error", "worst"]):
        top, bottom = in_y[number * 10], in_y[number * 10 + 9]
        middle = (top + bottom) / 2
        parts.append(f'<line x1="40" y1="{top:.1f}" x2="40" y2="{bottom:.1f}" class="net-bracket"/>')
        parts.append(f'<text x="28" y="{middle:.1f}" class="net-group" transform="rotate(-90 28 {middle:.1f})">{name}</text>')

    for i, column in enumerate(data.columns):
        brightness = 0.25 + 0.75 * min(abs(scaled_row[i]) / 3, 1)
        parts.append(f'<circle cx="{in_x}" cy="{in_y[i]:.1f}" r="3.8" fill="{TEXT}" fill-opacity="{brightness:.2f}">'
                     f'<title>{label(column)}: {raw_row[i]:g}</title></circle>')
    for j in range(20):
        if alive[j]:
            parts.append(f'<circle cx="{hidden_x}" cy="{hidden_y[j]:.1f}" r="7" fill="{ACCENT}" '
                         f'fill-opacity="{0.25 + 0.75 * activation[j]:.2f}"><title>Neuron {j + 1}: {hidden[j]:.2f}</title></circle>')
        else:
            parts.append(f'<circle cx="{hidden_x}" cy="{hidden_y[j]:.1f}" r="6" class="net-off">'
                         f'<title>Neuron {j + 1}: switched off by ReLU</title></circle>')
    for k, (name, color, chance) in enumerate([
        ("Malignant", MALIGNANT_COLOR, malignant_chance),
        ("Benign", BENIGN_COLOR, 1 - malignant_chance),
    ]):
        parts.append(f'<circle cx="{out_x}" cy="{out_y[k]:.1f}" r="17" fill="{color}" fill-opacity="{0.15 + 0.85 * chance:.2f}" '
                     f'stroke="{color}" stroke-width="2"><title>{name}: {chance:.1%}</title></circle>')
        parts.append(f'<text x="{out_x + 26}" y="{out_y[k] - 4:.1f}" class="net-out">{name}</text>')
        parts.append(f'<text x="{out_x + 26}" y="{out_y[k] + 16:.1f}" class="net-out-value">{percent(chance)}</text>')
    parts.append("</svg>")
    return "".join(parts)


picks = quick_picks()
if "record" not in st.session_state:
    st.session_state.record = picks["Clear malignant"]

(caught, missed), (false_alarms, cleared) = report["confusion_matrix"]
test_size = report["test_size"]
right = caught + cleared

# ---------- Hero ----------
st.html(f"""
<section class="hero">
  <div class="chips">
    <span class="live"><i></i>Live model</span><span>Deep learning</span><span>TensorFlow · Keras</span>
  </div>
  <h1>Breast cancer, read by a <em>neural network</em>.</h1>
  <p class="lede">A {model.count_params()}-parameter network reads 30 measurements of cell nuclei from a biopsy
  image and calls the tumour malignant or benign. Pick a real test case, change any number,
  and watch every neuron react.</p>
  <div class="kpis">
    <div><b>{right / test_size:.1%}</b><span>Test accuracy</span><small>{right} of {test_size} unseen cases</small></div>
    <div><b>{caught / (caught + missed):.1%}</b><span>Malignant caught</span><small>{caught} of {caught + missed}</small></div>
    <div><b>{cleared / (cleared + false_alarms):.1%}</b><span>Benign cleared</span><small>{cleared} of {cleared + false_alarms}</small></div>
    <div><b>{model.count_params()}</b><span>Parameters</span><small>30 → 20 → 2</small></div>
  </div>
  <div class="links">
    <a class="cta" href="{REPO}/tree/main/Breast-Cancer-Classification-using-NN" target="_blank">View the code on GitHub ↗</a>
    <span>Learning project · not a medical tool</span>
  </div>
</section>
""")

# ---------- Live console ----------
controls, results = st.columns([0.92, 1.08], gap="large")

with controls:
    with st.container(key="controls"):
        st.html("""
        <p class="step"><b>01</b>Choose a case</p>
        <p class="hint">All 114 test cases were held back from training. The network has never seen them.</p>
        """)
        names = list(picks)
        for start_at in range(0, len(names), 2):
            # A new row every 2 buttons, so phones (one column) keep the same order
            for column, name in zip(st.columns(2), names[start_at:start_at + 2]):
                column.button(
                    name, key=f"pick-{name}", on_click=choose, args=(picks[name],), width="stretch",
                    type="primary" if st.session_state.record == picks[name] else "secondary",
                )
        record = st.selectbox(
            "Or pick any test case",
            report["test_ids"],
            key="record",
            format_func=lambda row: f"Case #{row} · actually {NAMES[target[row]]}",
        )
        start = data.loc[record]

        st.html("""
        <p class="step"><b>02</b>Change the measurements</p>
        <p class="hint">Sliders cover the range seen in the data. The result updates as you move them.</p>
        """)
        values = {}
        for tab, (pattern, note) in zip(st.tabs(list(GROUPS)), GROUPS.values()):
            with tab:
                st.html(f'<p class="hint">{note}</p>')
                for number, (measurement, help_text) in enumerate(MEASUREMENTS.items()):
                    if number % 2 == 0:
                        columns = st.columns(2)
                    column = pattern.format(measurement)
                    low, high = data[column].min(), data[column].max()
                    step = 10.0 ** (np.floor(np.log10(high - low)) - 3)
                    decimals = max(0, int(round(-np.log10(step))))
                    # The key includes the case, so picking a new case resets every slider
                    values[column] = columns[number % 2].slider(
                        label(measurement), float(low), float(high), float(start[column]), float(step),
                        format=f"%.{decimals}f", help=help_text, key=f"{record}-{column}",
                    )
        changed = [column for column in data.columns if not np.isclose(values[column], start[column])]
        st.button(
            "↺ Reset to the original values", on_click=reset, args=(record,),
            disabled=not changed, width="stretch",
        )

row = np.array([[values[column] for column in data.columns]])
scaled = scaler.transform(row)
malignant_chance = float(keras.ops.convert_to_numpy(model(scaled, training=False))[0, 0])
predicted = "malignant" if malignant_chance >= 0.5 else "benign"

# Row 0 is the sample. Row k + 1 is the same sample with measurement k set to the
# training average, which is 0 after scaling. The score change is that measurement's effect.
rows = np.repeat(scaled, len(data.columns) + 1, axis=0)
rows[np.arange(1, len(rows)), np.arange(len(data.columns))] = 0
hidden, logits = forward(rows)
scores = logits[:, 0] - logits[:, 1]  # model score: above 0 means malignant
effects = scores[0] - scores[1:]

if changed:
    count = f"{len(changed)} measurement" + ("s" if len(changed) > 1 else "")
    truth_style, truth = "edited", f"✎ Edited sample · you changed {count}, so there is no real diagnosis"
elif NAMES[target[record]] == predicted:
    truth_style, truth = "right", "✓ Matches the real diagnosis"
else:
    truth_style, truth = "wrong", f"✗ Real diagnosis: {NAMES[target[record]]} · the network got this one wrong"

call_color = MALIGNANT_COLOR if predicted == "malignant" else BENIGN_COLOR
confidence = malignant_chance if predicted == "malignant" else 1 - malignant_chance

order = np.argsort(-abs(effects))
top = order[:8]
limit = abs(effects[top]).max() or 1
driver_rows = []
for k in top:
    effect = effects[k]
    width = 50 * abs(effect) / limit
    direction, left = ("up", 50) if effect > 0 else ("down", 50 - width)
    column = data.columns[k]
    driver_rows.append(f"""
    <div class="driver" title="{label(column)} = {row[0, k]:g} (training average {scaler.mean_[k]:.4g})">
      <span class="driver-name">{label(column)}</span>
      <span class="driver-track"><i class="{direction}" style="left: {left:.1f}%; width: {width:.1f}%"></i></span>
      <span class="driver-value">{effect:+.1f}</span>
    </div>""")

with results:
    with st.container(key="results"):
        svg_html(f"""
        <div class="verdict">
          <div>
            <div class="kicker">The network's call · case #{record}</div>
            <div class="call" style="color: {call_color}">{"▲" if predicted == "malignant" else "▼"} {predicted.capitalize()}</div>
            <div class="truth {truth_style}">{truth}</div>
          </div>
          {donut(confidence, call_color)}
        </div>
        """)
        st.html('<p class="step"><b>03</b>Inside the network</p>')
        svg_html(f'<div class="net-wrap">{network_svg(row[0], scaled[0], hidden[0], malignant_chance)}</div>')
        st.html(f"""
        <p class="hint">Brighter line = more signal (input × weight). Hollow neurons are switched off by ReLU
        for this sample. Hover any node for its value.</p>

        <p class="step"><b>04</b>What drove this call</p>
        <p class="hint">Each bar: how far the model score (now {scores[0]:+.1f}, above 0 means malignant) moves when that one
        measurement is swapped for the training average. Top 8 of 30. Learned patterns, not medical causes.</p>
        <div class="drivers">
          {"".join(driver_rows)}
          <div class="driver-axis"><span>◀ Benign</span><span>Malignant ▶</span></div>
        </div>
        """)
        with st.expander("All 30 measurements as a table"):
            st.dataframe(
                pd.DataFrame({
                    "Measurement": [label(column) for column in data.columns[order]],
                    "Effect": effects[order].round(2),
                    "Value": [f"{value:g}" for value in row[0, order]],
                    "Average": [f"{value:.4g}" for value in scaler.mean_[order]],
                }),
                hide_index=True,
                column_config={"Effect": st.column_config.NumberColumn(format="%+.2f")},
            )

# ---------- How it was built ----------
st.html(f"""
<h2 class="section-title">How it was built</h2>
<div class="card">
  <ol class="pipeline">
    <li><b>Data</b><span>569 biopsies × 30 measurements, Wisconsin Diagnostic Breast Cancer</span></li>
    <li><b>Split</b><span>{report["train_size"]} cases to train on, {test_size} held back for the test</span></li>
    <li><b>Scale</b><span>StandardScaler, fitted on the training cases only</span></li>
    <li><b>Train</b><span>Dense(20, ReLU) → Dense(2, softmax), Adam, 50 epochs</span></li>
    <li><b>Test</b><span>{right} of {test_size} unseen cases right</span></li>
  </ol>
</div>
""")
matrix, curve = st.columns(2, gap="medium")

with matrix:
    def cell(count, total):
        return f'<b style="background: rgba(169, 155, 255, {0.08 + 0.5 * count / total:.2f})">{count}</b>'

    biggest = max(caught, missed, false_alarms, cleared)
    st.html(f"""
    <div class="card">
      <p class="card-title">Test results</p>
      <div class="matrix">
        <span></span><span class="col">Predicted malignant</span><span class="col">Predicted benign</span>
        <span class="row">Actually malignant</span>{cell(caught, biggest)}{cell(missed, biggest)}
        <span class="row">Actually benign</span>{cell(false_alarms, biggest)}{cell(cleared, biggest)}
      </div>
      <p class="hint">{missed} malignant tumours were missed and {false_alarms} benign ones were flagged.
      In a real screening tool the missed ones matter most.</p>
    </div>
    """)

history = report["history"]
epochs = list(range(1, len(history["accuracy"]) + 1))
history_data = pd.DataFrame({
    "Epoch": epochs * 2,
    "Accuracy": history["accuracy"] + history["val_accuracy"],
    "Data": ["Training"] * len(epochs) + ["Validation"] * len(epochs),
})
nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Epoch"], empty=False)
lines = alt.Chart(history_data).mark_line(strokeWidth=2).encode(
    x=alt.X("Epoch:Q", axis=alt.Axis(tickCount=5)),
    y=alt.Y("Accuracy:Q", title=None, scale=alt.Scale(domain=[0.6, 1]), axis=alt.Axis(format="%", tickCount=5)),
    color=alt.Color(
        "Data:N",
        scale=alt.Scale(domain=["Training", "Validation"], range=[ACCENT, MUTED]),
        legend=alt.Legend(orient="top", title=None, symbolType="stroke"),
    ),
)
crosshair = (
    alt.Chart(history_data)
    .transform_pivot("Data", value="Accuracy", groupby=["Epoch"])
    .mark_rule(color=MUTED, strokeWidth=1)
    .encode(
        x="Epoch:Q",
        opacity=alt.condition(nearest, alt.value(0.7), alt.value(0)),
        tooltip=["Epoch:Q", alt.Tooltip("Training:Q", format=".1%"), alt.Tooltip("Validation:Q", format=".1%")],
    )
    .add_params(nearest)
)
history_chart = (
    alt.layer(lines, crosshair)
    .properties(height=212, background="transparent")
    .configure_view(stroke=None)
    .configure_axis(
        labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11,
        labelColor=MUTED, titleColor=MUTED, titleFontWeight=400,
        gridColor=GRID, domain=False, ticks=False,
    )
    .configure_legend(labelFont="Inter", labelFontSize=12, labelColor=TEXT)
)

with curve:
    with st.container(key="curve"):
        st.html('<p class="card-title">Accuracy while training</p>')
        st.altair_chart(history_chart, theme=None, width="stretch")

st.html(f"""
<h2 class="section-title">Engineering notes</h2>
<div class="notes">
  <div><b>No test-data leakage</b><p>The scaler learns its mean and spread from the {report["train_size"]} training
  cases only, then reuses them on the test set. Refitting it on test data would let the test set leak into the inputs.</p></div>
  <div><b>Softmax output</b><p>Two outputs that add up to 1, which is what sparse categorical cross-entropy expects.
  The confidence ring is that softmax output.</p></div>
  <div><b>Epochs picked on validation</b><p>Validation loss stops improving around epoch 50, so training stops there.
  The test set was not used to choose it.</p></div>
  <div><b>Explainable calls</b><p>The driver bars swap one measurement at a time for its training average and
  measure how far the score before softmax moves. That score does not get stuck near 0% or 100%.</p></div>
</div>
<footer>
  Built by <a href="https://github.com/adi-mauryaji" target="_blank">@adi-mauryaji</a> with TensorFlow, Keras,
  scikit-learn and Streamlit · Data: Wisconsin Diagnostic Breast Cancer, UCI Machine Learning Repository
</footer>
""")
