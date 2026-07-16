# Plan del MVP Kaleido FlowTwin

## Objetivo y decision

Construir un piloto offline/read-only para predecir tiempo restante y riesgo de
desviacion en una operacion/turno de Trace Port. El plan admite que el historico
sea insuficiente: en ese caso termina con instrumentacion, process intelligence
y simulacion, sin inventar un predictor.

## Estados de salida

| Estado | Significado | Siguiente decision |
|---|---|---|
| `GO_PREDICTIVE` | hay volumen, calidad, outcomes y senal | piloto shadow 4-8 semanas |
| `INSTRUMENT_FIRST` | proceso util pero dato insuficiente | activar captura minima y reevaluar |
| `BASELINE_ONLY` | modelo simple gana | productizar baseline, no JEPA |
| `NO_SIGNAL` | no hay mejora ni KPI accionable | parar o cambiar caso |

Todos son resultados validos del piloto.

## Evidencia publica ejecutada para la demo

El caso de presentacion es ahora ETA AIS, alineado con Shipping Board/Freight
Intelligence. Un modelo elegido en validacion obtiene 1.875 h de MAE en 85 viajes
del test futuro 1-7 febrero 2025, frente a 2.726 h del historico y 7.786 h de la
cinematica; pasa 6/6 gates predeclarados. OCEL Logistics cubre proceso y objetos.
Ambos permanecen `smoke_only` y no sustituyen las fases Kaleido descritas abajo.

El benchmark historico de almacén (~734 min MAE) y JEPA se conservan como
evidencia rechazada/I+D, no como predictor de producto.

## Fase 0 - Alineamiento (reunion)

Duracion: 60-90 minutos.

Entregables:

- proceso candidato;
- comprador/usuario;
- unidad de decision;
- desviacion material;
- acciones permitidas;
- KPI y coste aproximado;
- propietario de datos;
- export minimo acordado.

Gate: una pregunta predictiva con timestamp de decision y outcome verificable.

## Fase 1 - Data Evidence Scan

Duracion: 3-5 dias tras recibir muestra.

Entrada: 3-5 operaciones anonimizadas y diccionario/export.

Tareas:

- schema y relaciones;
- timezone, IDs, duplicados y revisiones;
- action/context/outcome audit;
- cobertura de plan, actual y outcome;
- inventario de eventos/modalidades;
- riesgo GDPR/seguridad;
- estimacion de operaciones utilizables y diversidad.

Salida:

- `data_readiness_report.html/pdf`;
- canonical mapping;
- leakage report;
- decision `go/instrument/no-fit`;
- presupuesto de piloto.

Gate: se puede reconstruir al menos una operacion sin mirar el futuro.

## Fase 2 - Proceso y baseline

Duracion: semana 1-2.

Tareas:

- object-centric event log;
- process map, variantes, esperas y conformance;
- plan vs actual;
- naive/median/survival/boosting;
- split cronologico y por proyecto;
- metricas operativas preliminares.

Gate:

- artefactos reproducibles;
- umbrales solo de validation;
- baseline aporta informacion por encima de plan/mediana;
- explicacion revisable por un operador.

## Fase 3 - Modelo secuencial

Duracion: semana 2-4, si Fase 2 pasa.

Tareas:

- prefijos causales a varios prediction points;
- GRU/TCN/ProcessTransformer;
- object-centric graph baseline;
- quantiles/calibracion;
- error por grupo y drift temporal.

Gate: mejora material sobre boosting en al menos una metrica operativa sin
empeorar de forma inaceptable calibracion o peor grupo.

## Fase 4 - Event-JEPA experimental

Duracion: semana 4-5, condicionada al volumen.

Tareas:

- future-embedding prediction 2/4/8 h;
- target encoder/anticollapse;
- actions vs context;
- ablations correct/shuffled/no-action;
- frozen probes y fine-tuning;
- tres seeds.

Gate de promocion:

- delta positivo frente a mejor secuencial;
- lead time o falsas alertas mejores;
- mejora sostenida por seed/holdout;
- correct actions > shuffled para action claim;
- latencia y calibracion aceptables.

Si falla, se elimina del producto.

## Fase 5 - Escenarios

Duracion: semana 4-6, en paralelo conceptual con Fase 4.

Ruta A, prioritaria:

- simulacion discreta de tareas, recursos, colas, pausas y calendario;
- calibracion con tiempos observados;
- escenarios definidos con experto;
- sensibilidad y rangos, no una cifra unica.

Ruta B, solo con acciones reales:

- ranking de acciones observadas;
- soporte/uncertainty guard;
- offline policy evaluation conservadora;
- advisory-only.

Gate: escenarios plausibles, dentro de restricciones y con hipotesis visibles.

## Fase 6 - Shadow MVP

Duracion: semana 5-6 para replay; 4-8 semanas prospectivas si Kaleido continua.

Entregables:

- dashboard de operacion;
- P50/P90 y riesgo 2/4/8 h;
- timeline y fuente de cada alerta;
- motivos/objetos afectados;
- abstencion;
- export PDF/CSV/API;
- model/data cards;
- informe tecnico y ejecutivo.

Gate de piloto prospectivo, a acordar:

- lead time minimo;
- limite de falsas alertas/turno;
- cobertura P90;
- mejora vs plan/baseline;
- utilidad operatoria;
- disponibilidad/latencia.

## Cronograma comercial propuesto

```text
Reunion        seleccionar proceso y muestra
Dias 1-5       Data Evidence Scan
Semana 2       mapa de proceso + baseline
Semana 3-4     predictor y calibracion
Semana 5       escenarios + dashboard
Semana 6       replay, informe y go/no-go
Posterior      shadow prospectivo y productizacion
```

No vender `4-6 semanas` como garantia de precision. Es plazo para obtener una
decision basada en evidencia.

## Matriz de exito

### Tecnico

- integridad temporal;
- holdout sin leakage;
- intervalo calibrado;
- mejora frente a baseline;
- robustez por proyecto/carga;
- latencia y trazabilidad.

### Operativo

- alertas antes de que el operador ya conozca el problema;
- volumen de falsas alarmas tolerable;
- razon de alerta comprensible;
- accion disponible en ese momento;
- dashboard no duplica trabajo.

### Comercial

- comprador identificado;
- coste del problema medible;
- integracion con producto Kaleido;
- packaging y precio potencial;
- IP/datos/soporte acordables;
- ruta a mas clientes sin mezclar datos.

## Riesgos principales

| Riesgo | Mitigacion |
|---|---|
| historico pobre | instrument-first, public/synthetic solo para plumbing |
| pocos eventos negativos | remaining time, survival, top-risk; no clasificador forzado |
| plan sobreescrito | revisionado y cutoff temporal |
| IDs no enlazables | resolver mapping antes de modelar |
| atajos por cliente/proyecto | holdout, ablations y worst-group |
| drift de proceso | frozen reference + shadow, reset por redisenos |
| accion confundida con contexto | contrato y shuffled-action test |
| ROI no disponible | separar metricas tecnicas de simulacion economica |
| duplicar Trace Port | integrar la prediccion dentro del producto |
| scope creep 3D/robotica | backlog separado, no primera version |

## Backlog posterior

- Shipping Board remaining-time/exception transfer;
- Freight Intelligence ETA probabilistica;
- fotos de incidencia con DINO frozen features;
- root-cause cross-product;
- optimizacion de recursos bajo restricciones;
- LiDAR/vision para ocupacion y maniobras;
- aprendizaje federado o por cliente con adapters.

## Experimento activo 2026-07-17: Port Call Deviation Twin

El caso JEPA activo deja de ser remaining-time tabular en LaDe. Se usa NOAA AIS
para aprender dinamica de aproximacion a puerto sin arrival/remaining-time
labels. El producto candidato combina el estado/futuro Phys-JEPA con GBT.

Orden obligatorio:

1. seleccionar capacidad y regularizador solo en datos ya abiertos: completado,
   Phys-JEPA + VICReg;
2. verificar tres seeds y VISReg/SIGReg/VICReg/none: completado, sin colapso;
3. congelar config, codigo, gates y commit;
4. descargar y hashear 2025-02-08--14 sin inspeccionar outcomes;
5. commit de manifest de datos;
6. construir prefijos y abrir test una vez;
7. integrar solo si el gate futuro pasa.

Vease `docs/decisions/0005-port-call-deviation-twin-phys-jepa.md` y
`docs/investigacion/JEPA_WORLD_MODEL_2026_AUDIT.md`.
