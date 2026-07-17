# Presentación FlowTwin

Archivos:

- `Kaleido_FlowTwin_Presentacion.html`: versión offline para proyectar; flechas,
  `O` vista general, `N` notas y `F` pantalla completa.
- `Kaleido_FlowTwin_Presentacion.pdf`: versión para enviar/proyectar.
- `Kaleido_FlowTwin_Presentacion.tex`: fuente Beamer.
- `../output/pdf/Kaleido_FlowTwin_Guion_Presentacion.pdf`: documento que debes
  estudiar; contiene el texto, tiempos, transiciones, demo y respuestas.
- `../output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.pdf`: respaldo técnico.

La historia principal es el Port Call Deviation Twin: el ensemble GBT + Phys-JEPA
reduce el MAE de trayectoria de 2,635 a 2,326 km (11,72 %) en 57 viajes futuros,
con IC95 % emparejado de mejora 5,90 %-17,13 %. El core público es
`claim_eligible`; no demuestra precisión Kaleido. ETA GBT y OCEL son capacidades
complementarias. Los heads ETA/delay de Phys-JEPA y el experimento de almacén se
mantienen como resultados rechazados, no se ocultan.

Dashboard local:

```powershell
uv run flowtwin serve --host 127.0.0.1 --port 8001 --artifact-root outputs
Start-Process "http://127.0.0.1:8001/"
```

Recorrido: encaje Kaleido -> holdout limpio -> ETA/OCEL complementarios ->
anticolapso -> mejora Phys-JEPA -> gates independientes -> producto shadow ->
arquitectura, dashboard y petición de piloto.

Regenerar fuentes y compilar:

```powershell
uv run flowtwin build-final-package
latexmk -pdf -interaction=nonstopmode -halt-on-error Kaleido_FlowTwin_Presentacion.tex
```

Los auxiliares LaTeX y los datos/artefactos voluminosos están ignorados por Git.
