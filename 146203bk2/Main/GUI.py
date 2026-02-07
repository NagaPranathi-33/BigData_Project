import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from . import Run  # Adjust import based on location of Run.py

def start_process(dataset, selection_type, value):
    if not dataset or not selection_type or not value:
        return [gr.update(value="")] * 18

    try:
        value = float(value)
        tp = value / 100 if selection_type == 'Training data (%)' else (value - 1) / value if value > 1 else 0
        Acc, Tpr, Tnr = Run.callmain(dataset, tp)

        Acc = Acc if isinstance(Acc, (list, np.ndarray)) and len(Acc) == 6 else [""] * 6
        Tpr = Tpr if isinstance(Tpr, (list, np.ndarray)) and len(Tpr) == 6 else [""] * 6
        Tnr = Tnr if isinstance(Tnr, (list, np.ndarray)) and len(Tnr) == 6 else [""] * 6

        results = Acc + Tpr + Tnr
        return [gr.update(value=str(v)) for v in results]

    except Exception as e:
        print(f"Error: {e}")
        return [gr.update(value="")] * 18

def plot_graph(*metrics):
    def safe_float(val):
        try:
            return float(val)
        except ValueError:
            return 0

    Acc = [safe_float(x) for x in metrics[:6]]
    Tpr = [safe_float(x) for x in metrics[6:12]]
    Tnr = [safe_float(x) for x in metrics[12:18]]

    data = np.array([Acc, Tpr, Tnr])
    model_labels = [
        "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
        "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
    ]
    bar_width = 0.15
    x = np.arange(3)  # Accuracy, TPR, TNR

    plt.figure(figsize=(12, 6))
    for idx, model in enumerate(model_labels):
        plt.bar(x + idx * bar_width, data[:, idx], width=bar_width, label=model)

    plt.xticks(x + bar_width * 2.5, ['Accuracy', 'TPR', 'TNR'])
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

# UI Elements
dataset_options = [
    'Adult', 'Credit_Approval', 'creditcard',
    'kddcup99_cleaned', 'transfers_sorted', 'adult_UCI', '7817_1_cleaned'
]
selection_options = ['Training data (%)', 'K-value']
models = [
    "Adaptive E-Bat+DBN", "CBF+DBN", "WOA+BRNN",
    "Hybrid NN", "SSPO-based DQN", "Proposed RFQN"
]

with gr.Blocks(theme="soft", css=".custom-btn {background-color: #6474FF; color: white; font-weight: bold; padding: 10px 20px; border-radius: 8px;}") as demo:
    gr.Markdown("<h1 style='text-align: center;'>Model Analysis For Different Datasets</h1>")

    with gr.Row():
        dataset = gr.Dropdown(choices=dataset_options, label="Select_dataset")
        selection_type = gr.Dropdown(choices=selection_options, label="Select")
        value = gr.Textbox(placeholder="Enter value", label="Enter Value")
        start_btn = gr.Button("START", elem_classes=["custom-btn"])

    gr.Markdown("---")

    acc_boxes, tpr_boxes, tnr_boxes = [], [], []

    with gr.Row():
        gr.Markdown("**Model Name**")
        for model in models:
            gr.Markdown(f"<div style='text-align: center; font-weight: bold;'>{model}</div>")

    with gr.Row():
        gr.Markdown("**Accuracy**")
        for _ in models:
            acc = gr.Textbox(show_label=False)
            acc_boxes.append(acc)

    with gr.Row():
        gr.Markdown("**TPR**")
        for _ in models:
            tpr = gr.Textbox(show_label=False)
            tpr_boxes.append(tpr)

    with gr.Row():
        gr.Markdown("**TNR**")
        for _ in models:
            tnr = gr.Textbox(show_label=False)
            tnr_boxes.append(tnr)

    gr.Markdown("---")

    with gr.Row():
        run_graph_btn = gr.Button("Run Graph", elem_classes=["custom-btn"])
        close_btn = gr.Button("Close", elem_classes=["custom-btn"])

    graph_output = gr.Plot()

    start_btn.click(start_process, inputs=[dataset, selection_type, value],
                    outputs=acc_boxes + tpr_boxes + tnr_boxes)

    run_graph_btn.click(plot_graph, inputs=acc_boxes + tpr_boxes + tnr_boxes,
                        outputs=graph_output)

    demo.launch(share=True)