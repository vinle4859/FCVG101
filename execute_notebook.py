"""Executes all code cells in FCVG101_Coursework_Master.ipynb,
capturing stdout and matplotlib figures, and saves the fully populated notebook.
"""
import io
import os
import sys
import json
import base64
import contextlib
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure Matplotlib uses Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display

def run_notebook(nb_path: Path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Capture display data (DataFrames as HTML tables, HTML snippets)
    current_displays = []

    def custom_display(*args, **kwargs):
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                html_rep = arg.to_html(classes="dataframe table table-striped table-bordered", index=False)
                text_rep = arg.to_string(index=False)
                current_displays.append({
                    "output_type": "display_data",
                    "data": {
                        "text/html": [line + "\n" for line in html_rep.splitlines()],
                        "text/plain": [line + "\n" for line in text_rep.splitlines()]
                    },
                    "metadata": {}
                })
            elif hasattr(arg, '_repr_html_'):
                html_rep = arg._repr_html_()
                current_displays.append({
                    "output_type": "display_data",
                    "data": {
                        "text/html": [line + "\n" for line in html_rep.splitlines()],
                        "text/plain": [str(arg) + "\n"]
                    },
                    "metadata": {}
                })
            else:
                text_rep = str(arg)
                current_displays.append({
                    "output_type": "display_data",
                    "data": {
                        "text/plain": [line + "\n" for line in text_rep.splitlines()]
                    },
                    "metadata": {}
                })

    IPython.display.display = custom_display

    # Global execution namespace
    exec_globals = {
        '__name__': '__main__',
        '__file__': str(nb_path),
        'Path': Path,
        'display': custom_display,
        'HTML': IPython.display.HTML
    }

    # Hook plt.show to capture figures
    current_figures = []
    original_show = plt.show

    def hook_show(*args, **kwargs):
        figs = [plt.figure(i) for i in plt.get_fignums()]
        if not figs:
            fig = plt.gcf()
            if fig.get_axes():
                figs = [fig]
        for fig in figs:
            buf = io.BytesIO()
            # Optimize figure DPI to 90 to keep total notebook size <= 5 MB while ensuring crisp text
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=90)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('utf-8')
            current_figures.append(b64)
            plt.close(fig)
        plt.close('all')

    plt.show = hook_show

    execution_count = 1
    total_code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"Executing {total_code_cells} code cells in {nb_path.name}...")

    # Change working directory to ROOT_DIR during execution so relative paths resolve
    os.chdir(ROOT_DIR)

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        code_text = "".join(cell['source'])
        print(f"\n--- [Cell {execution_count}/{total_code_cells}] ---")
        first_line = code_text.strip().split('\n')[0]
        print(f"Running: {first_line[:70]}...")

        current_figures.clear()
        current_displays.clear()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        outputs = []
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code_text, exec_globals)
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"ERROR in cell {execution_count}: {e}")
            outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": err_msg.split('\n')
            })

        # Check if unclosed figures remain
        if plt.get_fignums():
            hook_show()

        # Capture rich displays (DataFrames as HTML tables)
        outputs.extend(current_displays)

        # Capture stdout text
        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            lines = [l + '\n' for l in stdout_text.splitlines()]
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": lines
            })

        # Capture figures
        for fig_b64 in current_figures:
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": fig_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {}
            })

        cell['execution_count'] = execution_count
        cell['outputs'] = outputs
        print(f"Cell {execution_count} finished: {len(outputs)} outputs captured.")
        execution_count += 1

    plt.show = original_show

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    size_mb = nb_path.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully populated notebook {nb_path.name}!")
    print(f"Total file size: {size_mb:.2f} MB (Requirement: <= 5.0 MB)")
    assert size_mb <= 5.0, f"Notebook size {size_mb:.2f} MB exceeds 5 MB limit!"

if __name__ == '__main__':
    run_notebook(ROOT_DIR / 'FCVG101_Coursework_Master.ipynb')
