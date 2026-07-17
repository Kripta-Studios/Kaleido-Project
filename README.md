# Kaleido FlowTwin

MVP read-only de inteligencia predictiva para complementar Shipping Board,
Freight Intelligence, Trace Port y TWINPORTS. `FlowTwin` es un nombre provisional.

La evidencia pública tiene dos estados. Los demostradores ETA/OCEL históricos
son `smoke_only`. El nuevo núcleo físico Phys-JEPA es `claim_eligible` porque se
ejecutó desde un commit limpio sobre un holdout futuro prehasheado; aun así solo
prueba capacidad pública. No prueba precisión, ahorro, ROI ni despliegue en
Kaleido.

## Resultado principal que se presenta

El producto candidato es un **Port Call Deviation Twin** para Shipping Board y
Freight Intelligence. A partir de una secuencia AIS predice el estado físico a
0,5/1/2 horas y combina un GBT fuerte con estado/futuros Phys-JEPA:

- holdout limpio 8-14 de febrero de 2025, 750 muestras y 57 viajes disjuntos;
- trajectory GBT: 2,635 km MAE y AUPRC de desviación 0,880;
- híbrido individual, media de tres seeds: 2,587 ± 0,053 km, mejora 1,84%,
  gana al raw GBT en 3/3 seeds;
- ensemble de tres Phys-JEPA: 2,326 km, mejora 11,72%; bootstrap emparejado por
  viaje IC95 % 5,90%-17,13%, `P(mejora)=0,9995`;
- AUPRC de desviación: 0,880 a 0,904;
- cobertura conformal nominal 90%: 89,79%, ancho medio 12,00 km;
- rango efectivo 11,42-12,46 y 0/3 seeds colapsadas.

El **gate completo está cerrado**. Con solo el 10% de viajes etiquetados, ETA
mejora 0,59%, por debajo del 1% exigido, y el head de retraso retrocede de 0,619
a 0,606 AUPRC. La decisión de producto es servir GBT + Phys-JEPA solo como
núcleo físico shadow; GBT conserva ETA y se rechaza el head de retraso.

Dataset/export:
`noaa_marinecadastre_ais_2025_phys_jepa_holdout_02_08_02_14`, v1; prefijos
SHA-256 `4f1c7bce...9ef382d`; métricas SHA-256 `99b4162f...f8270ca`;
split cronológico por viaje 341/83/57; tres seeds; umbral físico de 10 km fijado
antes del test; el test no influyó en ninguna elección. Véase
[Decision 0007](docs/decisions/0007-phys-jepa-clean-holdout-result.md).

## Evidencia complementaria

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

Event-JEPA, Temporal T-JEPA, Var-JEPA y el experimento de acciones de almacén
permanecen como I+D histórica. El predictor con ~734 min de MAE y el caso LaDe
se conservan como evidencia técnica rechazada/invalidada, no como demostrador
ni como historia comercial.

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
5. [Resultado Phys-JEPA limpio](docs/decisions/0007-phys-jepa-clean-holdout-result.md)
   y [model card](docs/model_cards/port_call_phys_jepa.md).
6. [Auditoría de artículos JEPA](docs/investigacion/JEPA_WORLD_MODEL_2026_AUDIT.md)
   y [SOTA 2026](docs/investigacion/SOTA_2026.md).
7. [Petición de datos](DATA_REQUEST.md) y [brief](docs/MEETING_BRIEF.md).

Fuentes editables: [presentación TeX](presentacion/Kaleido_FlowTwin_Presentacion.tex),
[guion TeX](output/pdf/Kaleido_FlowTwin_Guion_Presentacion.tex) e
[informe TeX](output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.tex).

## Qué debes presentar exactamente

| Diapositiva | Tiempo | Mensaje |
|---|---:|---|
| 1. FlowTwin | 0:45 | GBT + Phys-JEPA mejora dinámica; no es un resultado Kaleido. |
| 2. Encaje | 1:15 | Deviation Twin para Shipping/FI; consecuencias en Trace Port/TWINPORTS. |
| 3. Producto y gates | 1:10 | Core, ETA, OCEL, API y gates separados. |
| 4. Datos/protocolo | 1:20 | 45 días AIS; holdout prehasheado; 57 viajes futuros; tres seeds. |
| 5. Process intelligence | 1:00 | OCEL ya aporta objetos, variantes y diagnóstico sin forzar el gate predictivo. |
| 6. ETA complementaria | 1:25 | GBT conserva ETA: 1,88 h; JEPA no desplaza el baseline. |
| 7. Incertidumbre ETA | 1:10 | P90 cubre 94,5 %, pero ancho 9,04 h y dominio concentrado. |
| 8. Anticolapso | 1:30 | VICReg, VISReg, SIGReg y none: 0 colapsos; la física aporta más. |
| 9. Resultado limpio | 1:45 | 2,635 a 2,326 km; mejora 11,72 %, IC95 % 5,90 %-17,13 %. |
| 10. Gate completo | 1:25 | Core pasa; ETA escasa no alcanza 1 % y delay empeora. |
| 11. Decisión | 1:00 | Phys-JEPA shadow para trayectoria; GBT-only ETA; delay fuera. |
| 12. Arquitectura | 1:10 | Integración read-only, versionada y auditable. |
| 13. Dashboard | 2:30 | Escalera de modelos, gates, model card, export y límites. |
| 14. Piloto | 1:15 | Export congelado, tolerancia acordada y replay shadow. |
| 15. Cierre | 0:35 | Pedir operación, responsables, 3–5 casos y fecha. |

La explicación corta del resultado es: “En 57 viajes futuros, el ensemble
GBT + Phys-JEPA baja el error de trayectoria de 2,635 a 2,326 km, una mejora del
11,72 %. El bootstrap emparejado por viaje sitúa la mejora entre 5,90 % y 17,13 %.
Pasa el gate del core físico, pero ETA y delay no; por eso solo proponemos shadow
y necesitamos datos Kaleido antes de hablar de valor operativo”.

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

uv run flowtwin benchmark-ais-world-model `
  --prefixes data/processed/noaa_ais_phys_jepa_holdout/prefixes.parquet `
  --config configs/experiment/noaa_ais_phys_jepa_clean_test.yaml `
  --output outputs/noaa_ais_phys_jepa_clean_test_v2

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
