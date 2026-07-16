# Datos y datasets

## Regla principal

Los datasets publicos sirven para desarrollar loaders, schemas y baselines. Solo
los datos de Kaleido pueden demostrar valor para Kaleido.

## Paquete de descarga recomendado

Descargar antes de iniciar la implementacion, en este orden:

1. **NOAA MarineCadastre AIS 2025**: enero y primera semana de febrero.
2. **Container Logistics OCEL 2.0 v3**: JSON, SQLite y XML.
3. **Warehouse outbound event log**: solo referencia JEPA historica.
4. **Order Management OCEL 2.0**: log SQLite y modelo CPN.
5. **Inventory Management Database and OCELs**: ultima version de Zenodo.
6. **BPI Challenge 2019**: solo para stress test secundario.

Los cuatro primeros cubren object-centric process mining, loaders multiformato,
remaining time sobre log real, simulacion, recursos, fallos y escenarios. BPI 2019
no debe bloquear el MVP.

Estructura local recomendada, fuera de Git:

    data/
      raw/public/ocel_container_logistics_v3/
      raw/public/noaa_ais_2025/
      raw/public/warehouse_outbound_2025/
      raw/public/ocel_order_management/
      raw/public/ocel_inventory_management/
      raw/benchmark/bpi2019/
      raw/reference/dcsa/
      raw/private/kaleido_trace_port/
      processed/
      manifests/

## Prioridad A - datos internos del piloto

### A1. Trace Port

Campos solicitados:

- proyecto, operacion, turno y revision de plan;
- packing list/unidades de carga;
- evento/estado y timestamp;
- responsable/equipo/recurso por rol;
- cantidad/peso/unidad;
- notas, incidencia y referencia de foto;
- inicio/fin/pausas;
- outcome y desviacion;
- export/API version y zona horaria.

Uso: caso principal de tiempo restante y riesgo de desviacion.

### A2. Shipping Board

- expedicion/recepcion;
- milestones, documentos y alertas;
- transportista y ventanas;
- timestamps planificados/reales;
- incidencias y entrega final.

Uso: transferencia del mismo modelo de prefijos a flujos de instalaciones.

### A3. Freight Intelligence

- contenedor, viaje, buque y carrier;
- milestones DCSA-like;
- ETA original y revisiones;
- AIS/posicion o referencias;
- alertas y excepciones;
- timestamps de consulta y de ocurrencia.

Uso: ETA/excepcion, no mezclar con eventos de terminal sin mapping.

### A4. Coste y planificacion

- plan original y revisiones;
- horas/recurso;
- overtime, espera, demurrage/detention cuando aplique;
- SLA/penalizacion;
- coste de inspeccion y falsa alarma.

Uso: convertir metricas en utilidad. Opcional para el primer modelo, necesario
para un business case cuantitativo.

## Prioridad B - contexto externo

### DCSA Track & Trace y Port Call

Estandar de procesos, eventos y APIs para transporte de contenedores. Usar como
referencia semantica, no asumir que Trace Port usa exactamente DCSA.

- <https://dcsa.org/standards/track-and-trace/standard-documentation-track-and-trace>
- <https://developer.dcsa.org/>
- <https://reference.dcsa.org/content/standards/industry-blueprint/v2026-q1/industry-blueprint-2026-q1>

### OCEL 2.0 / XES

OCEL es preferible para proyecto-operacion-turno-carga-recurso. XES se puede
exportar para herramientas clasicas de process mining.

- <https://www.ocel-standard.org/>
- <https://www.tf-pm.org/resources/xes-standard/about-xes>

### Meteo y oceanografia

- Puertos del Estado/Portuscopia: oleaje, nivel, corrientes, temperatura,
  mareografos y meteo portuaria.
- AEMET OpenData: observaciones/predicciones meteorologicas.

Fuentes:

- <https://portuscopia.puertos.es/>
- <https://www.puertos.es/en/services/oceanography>
- <https://www.aemet.es/en/datos_abiertos/AEMET_OpenData>

Verificar licencia, estacion, latencia, revision y timestamp de publicacion. Una
prediccion historica solo puede usar el forecast disponible en ese momento, no
la observacion futura.

### AIS

NOAA MarineCadastre publica AIS de EE. UU. sin restricciones de acceso para
planificacion costera. Es util para desarrollar pipelines de trayectoria y
port-call context, no representa Vigo ni prueba el caso Kaleido.

- <https://www.fisheries.noaa.gov/inport/item/77594>
- <https://marinecadastre.gov/ais/>

Para Vigo se necesita una fuente legal/contractual disponible para Kaleido.

El benchmark ejecutado usa 38 ficheros diarios, 2025-01-01 a 2025-02-07,
descargados de:

- <https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025/>

Se construyeron 303/73/85 viajes de train/validacion/test y el test futuro del
1 al 7 de febrero contiene 1.780 prefijos. El resultado seleccionado es 1.875 h
de MAE y pasa 6/6 gates predeclarados. Estado `smoke_only`; Estados Unidos no
representa Vigo. Manifest: `data/manifests/noaa_ais_2025_jan_feb.yaml`.

#### Holdout Phys-JEPA predeclarado

El Port Call Deviation Twin usa esos 38 dias solo para desarrollo y seleccion
de arquitectura. El intervalo 2025-02-08 a 2025-02-14 queda reservado como
holdout futuro. Antes de descargarlo deben estar versionados:

- arquitectura Phys-JEPA y regularizador;
- horizontes 0,5/1/2 h;
- GBT, GRU, Transformer, fisica y persistencia;
- fraccion etiquetada del 10 % por viaje;
- gates de distancia, ETA escasa, AUPRC y no-colapso.

La seleccion de desarrollo eligio Phys-JEPA + VICReg: el GBT hibrido redujo el
MAE de trayectoria de validacion de 3,336 a 3,124 km. El intervalo reservado
sigue sin descargarse; este numero no es evidencia de test limpio.

Manifest de protocolo:
`data/manifests/noaa_ais_2025_phys_jepa_holdout.yaml`. Los hashes se rellenan
despues de descargar sin inspeccionar contenido y se hace un segundo commit
limpio antes de construir los targets.

### LaDe delivery Jilin (diagnostico de seleccion de caso)

LaDe Jilin contiene 31.415 entregas y se uso para evaluar un dispatch JEPA. No
se promociona: coordenadas/hora/progreso favorecen una solucion tabular y el log
no contiene una revision inmutable del plan del dispatcher. La v1 downstream se
invalido por desalineacion de embeddings; la v2 uso orden final de entrega como
oracle; la v3 cambio a FIFO visible al cutoff pero se detuvo al pivotar a AIS.

- Fuente: <https://huggingface.co/datasets/Cainiao-AI/LaDe>.
- Manifest: `data/manifests/lade_delivery_jilin_2022.yaml`.
- Decision: `docs/decisions/0006-lade-jepa-invalidation-ledger.md`.

## Prioridad C - datasets publicos para desarrollo

### Container Logistics Object-centric Event Log (2026)

Log artificial OCEL 2.0 de coordinacion de pedido, documento, recogida, terminal
y envio. La version v3 contiene 35.761 eventos, 14.013 objetos, 14 tipos de
evento y 7 tipos de objeto. Descargar los tres formatos:

- **container_logistics.json** (10,0 MB);
- **container_logistics.sqlite** (23,0 MB);
- **container_logistics.xml** (13,4 MB).

- Uso: tests de object graph, import OCEL, process discovery y CI.
- No uso: metricas comerciales o afirmacion de generalizacion real.
- Fuente: <https://zenodo.org/records/18373888>.
- DOI: <https://doi.org/10.5281/zenodo.18373888>.
- Licencia: CC BY 4.0.

El run objeto-centrico ejecutado usa el SQLite y 1.966 contenedores finalizados.
El grafo correcto mejora test (86.64 h frente a 88.17 h), pero validacion elige
flat; queda como process/diagnostic evidence. Manifest:
`data/manifests/ocel_logistics_2026.yaml`.

### Warehouse outbound event log (2025)

Log real y anonimizado de un proceso outbound de una empresa logistica del
sector aeronautico. La descarga publicada ocupa 626,95 MB y el estudio parte de
169.523 trazas. Fue el primer benchmark de tiempo restante; ya no es el
demostrador principal por su escala de error absoluta.

- Uso: prefijos, remaining time, drift temporal, XGBoost, LSTM/Transformer y
  evaluacion por horizonte.
- No uso: evidencia portuaria o estimacion de ROI para Kaleido.
- Fuente: <https://figshare.com/articles/dataset/Warehouse_outbound_event_log/29500898>.
- DOI: <https://doi.org/10.6084/m9.figshare.29500898>.
- Licencia: CC BY 4.0.

### Order Management OCEL 2.0

Log simulado object-centric con pedidos, clientes, productos, paquetes,
empleados, envios y entregas fallidas. Contiene 21.008 eventos, 10.840 objetos,
11 tipos de evento y 6 tipos de objeto.

- Uso: transferencia entre procesos, relaciones recurso-paquete-pedido,
  incidencias de entrega y pruebas del simulador.
- Descargar **order-management-log.zip** y **order-management-model.zip**.
- Fuente: <https://ocel-standard.org/event-logs/simulations/order-management/>.

### Inventory Management Database and OCELs

Base SQLite, generador reproducible y logs OCEL de inventario, compras, ventas,
movimientos, stock, EOQ, safety stock y reorder point.

- Uso: simulacion controlable, escenarios, optimizacion y tests de regeneracion.
- Descargar siempre la ultima version enlazada desde:
  <https://ocel-standard.org/event-logs/simulations/invent_manag/>.
- Registro: <https://zenodo.org/records/15515788>.

### BPI Challenge 2019

Log real de purchase-to-pay con 251.734 items, 1.595.923 eventos y 42
actividades, anonimizado y XES. El fichero ocupa aproximadamente 728,6 MB.

- Uso: stress test, escalabilidad, next-event y benchmark PPM secundario.
- No uso: evidencia portuaria.
- DOI: <https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1>.

### LaDe

Dataset industrial de last-mile con paquetes y task events multi-ciudad.

- Uso: generalizacion espaciotemporal/last-mile opcional.
- No uso: terminal portuaria.
- Fuente: <https://arxiv.org/abs/2306.10675>.

### BPI logs adicionales

Solo para pretraining/benchmark de event encoders. Separar por dataset y no
presentar un promedio que oculte fallos por log.

- <https://www.tf-pm.org/resources/logs>

## Datos sinteticos

La simulacion discreta y el CPN pueden producir eventos para:

- tests y fixtures;
- fallos raros controlados;
- comprobar invariantes y escenarios;
- pretraining experimental claramente etiquetado.

No se usan para estimar precision real, ROI ni frecuencia de incidencias.

## Manifest obligatorio

```yaml
dataset_id: trace_port_export_v1
owner: kaleido
source_system: trace_port
export_version: unknown
access_date: YYYY-MM-DD
license_or_agreement: pending
timezone_source: Europe/Madrid
rows: measured
operations: measured
projects: measured
date_min: measured
date_max: measured
sha256: measured
contains_personal_data: unknown
contains_photos: false
plan_revisions_available: unknown
outcomes_available: unknown
action_columns: []
context_columns: []
observation_columns: []
outcome_columns: []
forbidden_columns: []
known_limitations: []
```

## Quality gates

- IDs estables y cobertura de relaciones;
- timezone y orden temporal;
- duplicados/reintentos;
- plan vs actual distinguibles;
- granularidad/event vocabulary;
- missingness por proyecto y periodo;
- outcome y censura;
- accion/context audit;
- riesgo de leakage;
- representatividad por carga/cliente/puerto;
- consentimiento/licencia/retencion.

## Presupuesto de datos para go/no-go

No se exige un numero magico. La decision usa:

- operaciones completas;
- diversidad de variantes;
- numero de eventos por prefijo;
- cobertura de plan y outcome;
- suficientes desviaciones/incidencias;
- estabilidad de IDs;
- separacion temporal disponible.

Si faltan outcomes, el piloto pasa a `instrument-first`: process mining,
simulacion y plan de captura, sin clasificador de riesgo.
