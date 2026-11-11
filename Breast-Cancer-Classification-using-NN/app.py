import json
from pathlib import Path

import altair as alt
import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

INK = "#1E2A2C"
MUTED = "#596462"
GRID = "#EAE5D8"
TOWARDS_MALIGNANT = "Towards malignant"
TOWARDS_BENIGN = "Towards benign"
MALIGNANT_COLOR = "#BF4A28"
BENIGN_COLOR = "#00806B"

folder = Path(__file__).parent

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🔬", layout="wide")
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


# Load the saved files once, not on every slider move
@st.cache_resource
def load_files():
    model = keras.saving.load_model(folder / "model.keras")
    scaler = joblib.load(folder / "scaler.pkl")
    report = json.loads((folder / "report.json").read_text())
    return model, scaler, report


@st.cache_data
def load_data():
    dataset = load_breast_cancer()  # ships with scikit-learn, no download
    return pd.DataFrame(dataset.data, columns=dataset.feature_names), dataset.target


def model_scores(scaled_rows):
    """Model score for each row: above 0 means the network picks malignant.

    It is the output layer before softmax (malignant minus benign). Softmax turns it into the
    chance on the meter, but the chance gets stuck near 0% or 100%, and the score does not.
    """
    hidden_layer, output_layer = model.layers
    hidden = keras.ops.convert_to_numpy(hidden_layer(scaled_rows))
    weights, bias = output_layer.get_weights()
    logits = hidden @ weights + bias
    return logits[:, 0] - logits[:, 1]


def section(number, title):
    st.html(f'<p class="section"><span>{number}</span>{title}</p>')


def signed(number):
    return f"{number:+.1f}".replace("-", "−")


def percent(chance):
    if chance > 0.999:
        return "99.9%+"
    if chance < 0.001:
        return "under 0.1%"
    return f"{chance:.1%}"


def label(column):
    return column[0].upper() + column[1:]


def style_chart(chart):
    return (
        chart.properties(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelFont="IBM Plex Sans", titleFont="IBM Plex Sans", labelFontSize=12, titleFontSize=12,
            labelColor=INK, titleColor=MUTED, titleFontWeight=400,
            gridColor=GRID, domain=False, ticks=False,
        )
        .configure_legend(labelFont="IBM Plex Sans", labelFontSize=12, labelColor=INK)
    )


model, scaler, report = load_files()
data, target = load_data()
test_ids = report["test_ids"]
accuracy = report["test_accuracy"]
(caught, missed), (false_alarms, cleared) = report["confusion_matrix"]

# Header
st.html('<p class="eyebrow">Practice deep learning project</p>')
st.title("Breast Cancer Classification")
st.html(f"""
<div class="specs">
  <div><span>Model</span>Neural network · 30 → 20 → 2</div>
  <div><span>Data</span>569 tumour samples</div>
  <div><span>Test accuracy</span>{accuracy:.1%}</div>
  <div class="disclaimer">For my deep learning practice, not medical advice.</div>
</div>
""")

inputs, result = st.columns([1.05, 1], gap="large")

with inputs:
    st.html("""
    <p class="note intro">Each sample is 30 measurements of the cell nuclei in a breast-lump biopsy image.
    Start from a test record the network never saw in training, then move any slider.
    The result updates straight away.</p>
    """)

    section("01", "Start from a test record")
    first_malignant = next(position for position, row in enumerate(test_ids) if target[row] == 0)
    record = st.selectbox(
        "Test record",
        test_ids,
        index=first_malignant,
        format_func=lambda row: f"Record {row} · actually {NAMES[target[row]]}",
        help=f"The {len(test_ids)} records held back from training.",
    )
    start = data.loc[record]

    section("02", "Cell nucleus measurements")
    values = {}
    for tab, (pattern, note) in zip(st.tabs(list(GROUPS)), GROUPS.values()):
        with tab:
            st.html(f'<p class="note">{note}</p>')
            for number, (measurement, help_text) in enumerate(MEASUREMENTS.items()):
                # A new row every 2 sliders, so phones (one column) keep the same order
                if number % 2 == 0:
                    columns = st.columns(2)
                column = pattern.format(measurement)
                low, high = data[column].min(), data[column].max()
                step = 10.0 ** (np.floor(np.log10(high - low)) - 3)
                decimals = max(0, int(round(-np.log10(step))))
                # The key includes the record, so picking a new record resets every slider
                values[column] = columns[number % 2].slider(
                    label(measurement), float(low), float(high), float(start[column]), float(step),
                    format=f"%.{decimals}f", help=help_text, key=f"{record}-{column}",
                )

    section("03", "About this data")
    st.html(f"""
    <p class="note">
      Wisconsin Diagnostic Breast Cancer data, 569 samples, as shipped with scikit-learn.
      The network trained on {report["train_size"]} of them for 50 epochs and was tested on the other {report["test_size"]}:
      {caught + cleared} right and {missed + false_alarms} wrong.
      Sliders cover the ranges seen in the data.
      <a href="https://github.com/adi-mauryaji/Deep-Learning-Practice-Vault">Source code</a>
    </p>
    """)

row = np.array([[values[column] for column in data.columns]])
changed = [column for column in data.columns if not np.isclose(values[column], start[column])]
scaled = scaler.transform(row)

# Row 0 is the sample. Row k + 1 is the same sample with measurement k set to the
# training average, which is 0 after scaling. The score change is that measurement's effect.
rows = np.repeat(scaled, len(data.columns) + 1, axis=0)
rows[np.arange(1, len(rows)), np.arange(len(data.columns))] = 0
scores = model_scores(rows)
score, effects = scores[0], scores[0] - scores[1:]

malignant_chance = float(keras.ops.convert_to_numpy(model(scaled, training=False))[0, 0])
predicted = "malignant" if malignant_chance >= 0.5 else "benign"

if predicted == "malignant":
    verdict_style, stamp = "malignant", "▲ Malignant"
else:
    verdict_style, stamp = "benign", "▼ Benign"

if changed:
    count = f"{len(changed)} measurement" + ("s" if len(changed) > 1 else "")
    check_style, check = "edited", f"✎ You changed {count}, so this is a new sample with no known diagnosis."
elif NAMES[target[record]] == predicted:
    check_style, check = "right", f"✓ Actual diagnosis: {predicted}. The network got this record right."
else:
    check_style, check = "wrong", f"✗ Actual diagnosis: {NAMES[target[record]]}. The network got this record wrong."

effect_data = pd.DataFrame({
    "Measurement": [label(column) for column in data.columns],
    "Value": [f"{value:g}" for value in row[0]],
    "Training average": [f"{value:.4g}" for value in scaler.mean_],
    "Effect on score": effects.round(2),
    "Direction": [TOWARDS_MALIGNANT if effect > 0 else TOWARDS_BENIGN for effect in effects],
    "Size": abs(effects),
}).sort_values("Size", ascending=False)
top = effect_data.head(10)
chart_limit = max(1.0, top["Size"].max())

base = alt.Chart(top).encode(
    x=alt.X(
        "Effect on score:Q",
        title="Effect on model score",
        scale=alt.Scale(domain=[-chart_limit, chart_limit], nice=False),
        axis=alt.Axis(tickCount=5, labelExpr="datum.value > 0 ? '+' + datum.label : datum.label"),
    ),
    y=alt.Y("Measurement:N", title=None, sort=top["Measurement"].tolist(), axis=alt.Axis(minExtent=150)),
    color=alt.Color(
        "Direction:N",
        scale=alt.Scale(domain=[TOWARDS_BENIGN, TOWARDS_MALIGNANT], range=[BENIGN_COLOR, MALIGNANT_COLOR]),
        legend=None,  # drawn in HTML above the chart, so narrow screens never cut it off
    ),
    tooltip=["Measurement", "Value", "Effect on score"],
)
# Round only the far end of each bar, so the bars start flat on the zero line
bars_right = base.transform_filter(alt.datum.Direction == TOWARDS_MALIGNANT).mark_bar(
    size=14, cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
)
bars_left = base.transform_filter(alt.datum.Direction == TOWARDS_BENIGN).mark_bar(
    size=14, cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4,
)
zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x="x:Q")
effect_chart = style_chart(alt.layer(bars_left, bars_right, zero_line).properties(height=300))

history = report["history"]
epochs = range(1, len(history["accuracy"]) + 1)
history_data = pd.DataFrame({
    "Epoch": [*epochs, *epochs],
    "Accuracy": history["accuracy"] + history["val_accuracy"],
    "Data": ["Training"] * len(epochs) + ["Validation"] * len(epochs),
})
history_chart = style_chart(
    alt.Chart(history_data)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("Epoch:Q", axis=alt.Axis(tickCount=5)),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.6, 1]), axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color(
            "Data:N",
            scale=alt.Scale(domain=["Training", "Validation"], range=[INK, "#9AA3A1"]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["Epoch", "Data", alt.Tooltip("Accuracy:Q", format=".1%")],
    )
    .properties(height=200)
)

with result:
    with st.container(key="report"):
        st.html(f"""
        <div class="verdict {verdict_style}">
          <span class="stamp">{stamp}</span>
          <p>The network reads this tumour as {predicted}.</p>
        </div>
        <div class="score">
          <div class="score-head"><span>Malignant chance</span><strong>{percent(malignant_chance)}</strong></div>
          <div class="track"><i style="left: {malignant_chance * 100:.1f}%"></i></div>
          <div class="track-labels"><div>◀ Benign</div><div>50%</div><div>Malignant ▶</div></div>
          <p class="note">The network's softmax output. It shows how sure the model is, not a medical risk.</p>
        </div>
        <p class="check {check_style}">{check}</p>
        """)

        st.html(f"""
        <p class="report-heading">What pushed this result</p>
        <p class="note">Each bar shows how far the model score (now {signed(score)}, above 0 means malignant)
        moves when that one measurement is swapped for the training average. Top 10 of 30.
        This is what the network learned from its data, not medical causes.</p>
        <p class="legend"><span><i class="benign"></i>{TOWARDS_BENIGN}</span><span><i class="malignant"></i>{TOWARDS_MALIGNANT}</span></p>
        """)
        st.altair_chart(effect_chart, theme=None, width="stretch")

        with st.expander("See all 30 numbers"):
            st.dataframe(
                effect_data[["Measurement", "Effect on score", "Value", "Training average"]],
                hide_index=True,
                column_config={
                    "Training average": st.column_config.TextColumn("Average"),
                    "Effect on score": st.column_config.NumberColumn("Effect", format="%+.2f"),
                },
            )

        with st.expander("How the network was trained and tested"):
            st.html(f"""
            <p class="note">Of {caught + missed} malignant test tumours, it caught {caught} and called {missed} benign.
            Of {false_alarms + cleared} benign ones, it got {cleared} right and called {false_alarms} malignant.
            Accuracy per epoch while training:</p>
            """)
            st.altair_chart(history_chart, theme=None, width="stretch")
