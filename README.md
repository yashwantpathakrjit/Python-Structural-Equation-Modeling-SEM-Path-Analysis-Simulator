# Python Structural Equation Modeling (SEM) Path Analysis Simulator

A clean, production-grade Python pipeline designed to simulate multi-variable survey data, perform path analysis (the structural component of Structural Equation Modeling), and generate presentation-ready path diagrams with complete statistical readings ($\beta$ coefficients, p-values, and $R^2$ variance blocks).

This repository serves as a self-contained demonstration tool for researchers, data scientists, and students looking to visualize directional hypotheses without relying on heavy external system dependencies like Graphviz.

---

## 📌 Key Features

* **Automated Data Generation:** Synthesizes an industrial-grade dataset ($N=300$) modeling 5 distinct Independent Variables and 1 Dependent Variable embedded with a controllable Gaussian noise distribution.
* **Standardized Path Calculations:** Auto-normalizes raw data into standardized $Z$-scores to extract true standardized path coefficients ($\beta$ weights) using an Ordinary Least Squares (OLS) estimation pipeline.
* **Dynamic Significance Mapping:** Automatically parses model $p\text{-values}$ to dynamically map social-science standard significance flags ($*$, $**$, $***$, or $ns$) directly onto model paths.
* **Zero System Dependencies:** Employs a pure Python mapping layout via `NetworkX` and `Matplotlib` to guarantee seamless cross-platform rendering (Windows, macOS, Linux) right out of the box.

---

## 📊 Conceptual Framework

The application models a multi-variable structural pathway where five independent dimensions simultaneously predict a single outcome criteria:

```text
  [ IV1 ] --- ( β1 ) ---\
  [ IV2 ] --- ( β2 ) ----\
  [ IV3 ] --- ( β3 ) -----> [ Dependent Variable (DV) ]  <-- [ R² Variance ]
  [ IV4 ] --- ( β4 ) ----/
  [ IV5 ] --- ( β5 ) ---/
