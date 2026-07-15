# Presentacion

Archivos finales:

- `Kaleido_FlowTwin_Presentacion.pdf`: version para proyectar o enviar.
- `Kaleido_FlowTwin_Presentacion.html`: version offline; flechas para navegar,
  `O` para vista general, `N` para notas y `F` para pantalla completa.
- `Kaleido_FlowTwin_Presentacion.tex`: fuente Beamer editable.

Compilacion desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error Kaleido_FlowTwin_Presentacion.tex
```

Los auxiliares de LaTeX estan ignorados por el `.gitignore` del proyecto.
