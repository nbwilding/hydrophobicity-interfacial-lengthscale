PNAS figure-generation scripts

Standard output naming:
  Main text: figure_x.pdf
  Supplementary Information: SI_figure_x.pdf

Current main-text scripts:
  figure_1_plot.py -> figure_1.pdf
  figure_2_plot.py -> figure_2.pdf
  figure_3_plot.py -> figure_3.pdf
  figure_4_plot.py -> figure_4.pdf
  figure_5_plot.py -> figure_5.pdf

New/revised scripts developed 17 Sep 2026:
  figure_2_new.py  -> figure_2.pdf
  figure_S1_new.py -> SI_figure_1.pdf
  figure_S2_new.py -> SI_figure_2.pdf
  figure_S3_new.py -> SI_figure_3.pdf

All scripts locate data relative to this archive's top-level data folders and write PDF output only into PNAS_figs.

figure_1_plot.py now writes the complete three-panel Fig. 1 as a single figure_1.pdf rather than the two legacy component PDFs.

Note: figure_2_new.py requests the revised Rs=10 pressure grid P=0,500,1000,2000 atm. The supplied archive currently contains Rs=10 files at P=0,750,1500,3000 atm. Add the corrected Rs=10 data files to mW PMF_data before running figure_2_new.py.
