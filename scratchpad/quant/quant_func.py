import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium", layout_file="layouts/quant_func.grid.json")


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotx
    plt.style.use(matplotx.styles.dracula)
    import sys

    sys.path.append("../../")
    from compression.qmodules import STERound
    from compression.qutils import quant_spec
    return STERound, matplotx, mo, np, plt, quant_spec, sys, torch


@app.cell(hide_code=True)
def _(torch):
    lin = torch.nn.Linear(4,4)
    lin.requires_grad_(False)
    lin.weight
    return (lin,)


@app.cell(hide_code=True)
def _(mo):
    decimal = decimal_places = mo.ui.slider(1,1000,10)
    bit_depth = mo.ui.slider(0,8,0.5)
    scale = mo.ui.slider(0,14,0.5)
    tolerance = mo.ui.slider(1,100,10)


    mo.md(f" Choose x precision:{decimal} , (b)it_depth : {bit_depth} , scal(e) :{scale} , tolerance = {tolerance}")
    return bit_depth, decimal, decimal_places, scale, tolerance


@app.cell(hide_code=True)
def _(bit_depth, decimal, mo, scale, tolerance, torch):
    # pass through rounding function
    tol = 1 / tolerance.value
    x = torch.tensor(8.34567892 * (1/decimal.value))
    y = -1 * x
    mo.md(f"Input x = {x,y} , b = {bit_depth.value} , e = {-1*scale.value} , tol = {tol} ")
    return tol, x, y


@app.cell(hide_code=True)
def _(bit_depth, mo, quant_spec, scale, tol, x, y):
    out1 = quant_spec(x,b=bit_depth.value,e=-1*scale.value)
    out2 = quant_spec(y,b=bit_depth.value,e=-1*scale.value)

    if_tol1 = (out1 >= x - tol) & (out1 <= x + tol)
    if_tol2 = (out2 >= y - tol) & (out2 <= y + tol )

    mo.md(f"Quantized = {out1},{out2} , tolerance_check = {if_tol1},{if_tol2}")
    return if_tol1, if_tol2, out1, out2


@app.cell
def _(bit_depth, quant_spec, scale, torch):
    xs = torch.linspace(-10,10,1000)
    ys = quant_spec(xs,b=bit_depth.value,e=-1*scale.value)
    return xs, ys


@app.cell(hide_code=True)
def _(out1, out2, plt, x, xs, y, ys):
    fig = plt.figure()
    plt.plot(xs,ys)
    plt.annotate("x",xy=(x,out1))
    plt.annotate("y",xy=(y,out2))
    plt.title(f"Linspaced Input:{xs[0]}..{xs[-1]}")
    plt.xlabel("x inputs")
    # plt.gca()
    fig

    return (fig,)


@app.cell
def _(plt, quant_spec, tol, torch, x):
    b_vals = torch.linspace(-1, 10, 300)
    e_vals = torch.linspace(0, -14, 300)
    B, E = torch.meshgrid(b_vals, e_vals, indexing='ij')

    Z = quant_spec(x, B, E)
    close = torch.where((Z>= x-tol) & (Z<=x+tol),1.0,0.0)

    fig2 = plt.figure()
    plt.contourf(B,E,close,cmap="coolwarm")
    plt.colorbar()
    plt.title(f"For input example:{x}")
    plt.xlabel("b")
    plt.ylabel("e")
    fig2
    return B, E, Z, b_vals, close, e_vals, fig2


@app.cell
def _(close):
    close
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
