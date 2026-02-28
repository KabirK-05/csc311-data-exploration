# CSC311 Machine Learning Project — Context

## What This Project Is

This is a team project (3–4 students) for CSC311: Introduction to Machine Learning at the University of Toronto (Winter 2026). The project is worth 15% of the final course grade.

## The Task

We receive a training dataset of student-contributed text responses. Each response belongs to one of **three categories**. Our job is to build a classifier that predicts the correct category for unseen responses. The final model is evaluated on a **hidden test set** compiled from TA and instructor responses — we never see this test set.

## Key Constraints

- **pred.py restrictions**: The final prediction script can only import standard Python libraries, numpy, and pandas. No sklearn, PyTorch, or TensorFlow at inference time. We can train with any tools, but must export model parameters and load them manually in pred.py.
- **File size limit**: All submitted files combined must not exceed 10MB.
- **Runtime**: Must produce ~60 predictions within 1 minute.
- **Data leakage**: Each student contributed 3 data points (one per class). All 3 must stay in the same train/val/test split — splitting them across sets leaks student identity information.
- **Model exploration**: We must explore at least **three model families**, tune all of them fairly (not just the winner), and justify our final choice.
- **Evaluation**: We must use accuracy plus at least one additional metric (precision, recall, F1) and justify why.

## Deliverables and Deadlines

| Component | Weight | What to Submit |
|-----------|--------|----------------|
| Data Collection | 1% | Quercus quiz (individual) |
| Team Formation | 1% | MarkUs group + contract |
| **Project Proposal** | **2%** | **proposal.pdf on MarkUs** |
| Prediction Script | 3% | pred.py on MarkUs |
| Final Report | 8% | report.pdf + code.zip on MarkUs |

## Current Focus: Project Proposal (2%)

The proposal has two sections:

**Data Exploration (1%)** — A complete draft covering dataset summary, data issues, preprocessing plan, data splitting with leakage prevention, figures with interpretation, and connections to model choice.

**Planned Methodology (1%)** — A reproducible plan covering three model families, optimization details, validation strategy, hyperparameter tuning plan, and evaluation metrics.

Graded on **completeness and clarity**, not final results. This is our chance to get early feedback.

## What Success Looks Like

- A model that **generalizes well** to the unseen TA/instructor test set (not just our own validation set)
- A report that demonstrates we **deeply understood the data** before modeling
- Fair comparison of **three tuned model families**, not just one
- Clear, reproducible methodology that a TA could re-implement from our description
