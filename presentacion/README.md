# Presentación FlowTwin

Archivos:

- `Kaleido_FlowTwin_Presentacion.html`: versión offline para proyectar; flechas,
  `O` vista general, `N` notas y `F` pantalla completa.
- `Kaleido_FlowTwin_Presentacion.pdf`: versión para enviar/proyectar.
- `Kaleido_FlowTwin_Presentacion.tex`: fuente Beamer.
- `../output/pdf/Kaleido_FlowTwin_Guion_Presentacion.pdf`: documento que debes
  estudiar; contiene el texto, tiempos, transiciones, demo y respuestas.
- `../output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.pdf`: respaldo técnico.

La historia principal es ETA AIS: 1,88 h de MAE sobre 85 viajes futuros, IC95 %
1,70-2,08 h, 60,6 % dentro de +/-2 h y 6/6 gates aprobados. El resultado es
`smoke_only`, no precisión Kaleido. OCEL muestra process intelligence y JEPA queda
como I+D. Los ~734 min del experimento de almacén sólo aparecen como negativo
histórico rechazado.

Dashboard local:

```powershell
uv run flowtwin serve --host 127.0.0.1 --port 8001 --artifact-root outputs
Start-Process "http://127.0.0.1:8001/"
```

Recorrido: evidencia ETA -> comparadores -> tolerancias -> 6/6 gates -> intervalo y
concentración por puerto -> procedencia/model card -> separación ETA/OCEL/JEPA.

Regenerar fuentes y compilar:

```powershell
uv run flowtwin build-final-package
latexmk -pdf -interaction=nonstopmode -halt-on-error Kaleido_FlowTwin_Presentacion.tex
```

Los auxiliares LaTeX y los datos/artefactos voluminosos están ignorados por Git.
