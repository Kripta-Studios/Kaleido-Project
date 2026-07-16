# Kaleido FlowTwin

MVP read-only de inteligencia predictiva para complementar Shipping Board,
Freight Intelligence, Trace Port y TWINPORTS. `FlowTwin` es un nombre provisional.

Toda la evidencia es `smoke_only`: prueba que la idea y el pipeline se han
ejecutado sobre datos públicos/simulados; no prueba precisión, ahorro, ROI ni
despliegue en Kaleido.

## Resultado que se presenta

El demostrador principal predice la entrada de un buque en una geofence portuaria
usando NOAA MarineCadastre AIS. El modelo `tabular_eta`, elegido exclusivamente en
validación, se evaluó una vez sobre un test futuro del 1 al 7 de febrero de 2025:

- 85 viajes y 1.780 puntos de predicción;
- MAE 1,88 h; error mediano 1,37 h; bootstrap por viaje IC95 % 1,70–2,08 h;
- 42,0 % dentro de ±1 h, 60,6 % dentro de ±2 h y 87,3 % dentro de ±4 h;
- ETA distancia/velocidad: 7,79 h; mediana puerto-distancia: 2,73 h;
- mejora del 75,9 % y 31,2 %, respectivamente;
- pasa los 6/6 gates fijados antes del test.

La cobertura P90 es 94,5 %, pero el intervalo mide 9,04 h de ancho medio. El
87,6 % de los prefijos de test pertenece a Nueva Orleans. Por eso la afirmación
correcta es: **funciona como demostrador público de ETA; todavía no demuestra
generalización a Vigo ni utilidad para Kaleido**.

El segundo ejemplo usa OCEL 2.0 Container Logistics para process intelligence y
relaciones objeto-céntricas. El grafo correcto mejoró el test de 88,17 a 86,64 h,
pero validación eligió la traza plana; el gate predictivo del grafo queda cerrado.

Event-JEPA, Temporal T-JEPA, Var-JEPA y el experimento de acciones permanecen como
I+D. El antiguo predictor de almacén con ~734 min de MAE se conserva como evidencia
técnica rechazada, no como demostrador ni como historia comercial.

## Encaje con Kaleido

- Shipping Board / Freight Intelligence: ETA, excepciones y priorización.
- Trace Port: eventos, turnos, incidencias, planes y outcomes.
- TWINPORTS: activos y estado físico/espacial.
- FlowTwin: predicción con intervalo, cutoff, procedencia y razones dentro de esas
  superficies; nunca escritura o control autónomo.

Fuentes principales:

- <https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025/>
- <https://doi.org/10.5281/zenodo.18373888>
- <https://www.kaleidologistics.com/en/productos-ktech/shipping-board/>
- <https://www.kaleidologistics.com/en/productos-ktech/freight-intelligence/>
- <https://www.kaleidologistics.com/en/productos-ktech/trace-port/>

## Qué debes estudiar para la presentación

Estudia, en este orden:

1. [Guion completo](output/pdf/Kaleido_FlowTwin_Guion_Presentacion.pdf): texto que
   decir, tiempos, transiciones, demo, respuestas y frases prohibidas.
2. [Presentación PDF](presentacion/Kaleido_FlowTwin_Presentacion.pdf): ensáyala junto
   al guion hasta explicar las cifras sin leer.
3. [Presentación HTML](presentacion/Kaleido_FlowTwin_Presentacion.html): versión
   offline para proyectar.
4. [Informe técnico](output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.pdf): protocolo,
   resultados, incertidumbre, JEPA y limitaciones.
5. [Decisión del nuevo benchmark](docs/decisions/0004-aligned-public-benchmark-pivot.md):
   por qué ETA AIS sustituye al caso de los minutos.
6. [SOTA 2026](docs/investigacion/SOTA_2026.md) y
   [decisión JEPA](docs/decisions/0003-temporal-tjepa-varjepa-hybrid-gate.md).
7. [Petición de datos](DATA_REQUEST.md) y [brief](docs/MEETING_BRIEF.md).

Fuentes editables: [presentación TeX](presentacion/Kaleido_FlowTwin_Presentacion.tex),
[guion TeX](output/pdf/Kaleido_FlowTwin_Guion_Presentacion.tex) e
[informe TeX](output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.tex).

## Qué debes presentar exactamente

| Diapositiva | Tiempo | Mensaje |
|---|---:|---|
| 1. FlowTwin | 0:45 | MVP ejecutado; no es un resultado Kaleido. |
| 2. Encaje | 1:15 | ETA para Shipping/FI; proceso para Trace Port/TWINPORTS. |
| 3. Trabajo construido | 1:10 | Dos demostradores alineados y JEPA como I+D falsable. |
| 4. Datos/protocolo | 1:20 | 38 días AIS; viaje agrupado; modelo elegido en validación; test futuro. |
| 5. Process intelligence | 1:00 | OCEL ya aporta objetos, variantes y diagnóstico sin forzar el gate predictivo. |
| 6. Resultado ETA | 1:40 | 1,88 h, IC95 % 1,70–2,08; 60,6 % en ±2 h; 6/6 gates. |
| 7. Incertidumbre | 1:15 | P90 cubre 94,5 %, pero ancho 9,04 h; test concentrado en Nueva Orleans. |
| 8. Ablations JEPA | 2:00 | Aprende representación y necesita anticolapso, pero no gana el floor. |
| 9. T/Var-JEPA | 2:15 | Mejoras de representación sin valor incremental al boosting. |
| 10. Acciones | 2:00 | Se recupera señal inyectada; no hay claim causal o de acciones Kaleido. |
| 11. Decisión | 1:00 | Demostrar ETA; OCEL como diagnóstico; JEPA en shadow. |
| 12. Arquitectura | 1:10 | Integración read-only, versionada y auditable. |
| 13. Dashboard | 2:30 | Comparadores, tolerancias, gates, intervalo, procedencia y límites. |
| 14. Piloto | 1:15 | Export congelado, tolerancia acordada y replay shadow. |
| 15. Cierre | 0:35 | Pedir operación, responsables, 3–5 casos y fecha. |

La explicación corta del resultado es: “El error medio de ETA es 1,88 horas. No
lo juzgamos aislado: baja de 7,79 h con física directa y de 2,73 h con histórico;
el 60,6 % queda en ±2 h y pasa los seis criterios predeclarados. Aún falta acordar
qué tolerancia necesita cada decisión de Kaleido”.

Si preguntan por los ~734 minutos: eran 12,2 h de MAE en otro proceso largo de
almacén. Ganaba por poco a comparadores débiles, pero no era una demostración
operativamente convincente. Se rechazó y se conserva para no borrar evidencia.

## Arrancar presentación y dashboard

Primera terminal PowerShell:

```powershell
Set-Location "C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\Kaleido-Project"
uv sync --extra sequence
uv run flowtwin serve --host 127.0.0.1 --port 8001 --artifact-root outputs
```

Segunda terminal:

```powershell
Set-Location "C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\Kaleido-Project"
Start-Process ".\presentacion\Kaleido_FlowTwin_Presentacion.html"
Start-Process "http://127.0.0.1:8001/"
```

Verificación sin navegador:

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/health"
Invoke-RestMethod "http://127.0.0.1:8001/v1/demo/evidence" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:8001/v1/models/latest/card" | ConvertTo-Json -Depth 10
```

En este equipo se usa 8001 porque Windows ocupa el 8000 y responde 404. La terminal
del servidor debe permanecer abierta. El dashboard permite buscar, filtrar, abrir
operaciones, copiar auditoría, exportar JSON/CSV, consultar model card y ejecutar
escenarios sintéticos aprobados; no escribe en sistemas fuente.

## Reproducir benchmarks y paquete

Los datos brutos están ignorados por Git. Con los ficheros oficiales descargados:

```powershell
uv run flowtwin benchmark-ais-eta data/raw/public/noaa_ais_2025 `
  --config configs/experiment/noaa_ais_eta_smoke.yaml `
  --output outputs/noaa_ais_eta_v3

uv run flowtwin benchmark-ocel-logistics data/raw/public/ocel20_container_logistics.sqlite `
  --config configs/experiment/ocel_logistics_graph_smoke.yaml `
  --output outputs/ocel_logistics_graph_v1

uv run ruff check .
uv run mypy src/flowtwin
uv run pytest -q
uv run flowtwin build-final-package
```

Cada run escribe config resuelta, entorno, manifiestos de datos/split/fugas,
métricas, predicciones, calibración, model card, informe y hashes. No se versionan
los 6,92 GB de AIS ni el SQLite OCEL.

## Entregables

- [Dashboard/API](src/flowtwin/dashboard/static/index.html)
- [Presentación HTML](presentacion/Kaleido_FlowTwin_Presentacion.html)
- [Presentación PDF](presentacion/Kaleido_FlowTwin_Presentacion.pdf)
- [Guion PDF](output/pdf/Kaleido_FlowTwin_Guion_Presentacion.pdf)
- [Informe técnico PDF](output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.pdf)
- [Progreso](docs/progress.md), [arquitectura](ARCHITECTURE.md) y [plan](PLAN.md)

No se deben versionar datos Kaleido, credenciales, fotos o identificadores de
clientes. Los ficheros originales de `correspondencia/` se preservan sin cambios.
