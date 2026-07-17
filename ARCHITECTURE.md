# Arquitectura de Kaleido FlowTwin

## Principios

1. **Data-poor by design.** El sistema entrega auditoria y process mining antes
   de necesitar un modelo neuronal.
2. **Multiobjeto.** Proyecto, operacion, turno, unidad de carga, recurso, buque,
   documento e incidencia coexisten; no se fuerza todo a un unico case ID.
3. **Plan versionado.** Plan inicial y replans son eventos, no columnas
   sobreescritas.
4. **Accion no es contexto.** Solo una decision timestamped e intervenible es
   accion.
5. **Read-only.** El MVP asesora; no controla equipos ni sistemas.
6. **Baselines first.** Un modelo complejo se conserva solo si anade valor.
7. **Uncertainty first.** Toda prediccion incluye intervalo, confianza o
   abstencion.

La demo concreta usa `AIS/Shipping Board -> trajectory GBT + Phys-JEPA +
conformal`, conserva `ETA boosting` como floor y añade `OCEL/Trace Port ->
process mining + object graph`. El núcleo físico JEPA es shadow read-only: su
evidencia pública limpia es positiva, pero ETA escasa no supera el gate del 1%
y el head de retraso queda rechazado. El antiguo modelo de tiempo restante de
almacén no se sirve.

## Vista logica

```text
Trace Port / Shipping Board / Freight Intelligence / CSV
                          |
                  read-only extract
                          v
          Canonical Operational Event Contract
         ids - event time - plan time - roles - units
                          |
              validation + lineage + quality
                          |
              object-centric event graph
       project-operation-shift-cargo-resource-vessel
                          |
        +-----------------+------------------+
        |                 |                  |
 process mining     predictive baselines   simulation
 conformance        GBT/GRU/Transformer    discrete-event
        |                 |                  |
        +-----------------+------------------+
                          |
       physical future + Phys-JEPA residual dynamics
            state/forecast at 0.5 / 1 / 2 hours
                          |
     GBT + JEPA trajectory/deviation | GBT-only ETA
                          |
        conformal calibration + frozen product gates
                          |
       dashboard / API / CSV / PDF / audit trail
```

## Contrato canonico

### `OperationEvent`

```python
@dataclass(frozen=True)
class OperationEvent:
    event_id: str
    source_system: str
    source_record_id: str
    event_type: str
    event_time_utc: datetime
    ingested_at_utc: datetime
    project_id: str | None
    operation_id: str | None
    shift_id: str | None
    cargo_unit_id: str | None
    resource_id: str | None
    vessel_id: str | None
    location_id: str | None
    actor_role: str | None
    status_from: str | None
    status_to: str | None
    numeric_value: float | None
    unit: str | None
    data_quality_flags: tuple[str, ...]
    payload_ref: str | None
```

### Plan y resultado

```python
@dataclass(frozen=True)
class PlanRevision:
    plan_id: str
    revision: int
    valid_from_utc: datetime
    operation_id: str
    milestone: str
    planned_start_utc: datetime | None
    planned_end_utc: datetime | None
    planned_quantity: float | None
    planned_resources: tuple[str, ...]
    reason: str | None

@dataclass(frozen=True)
class OperationOutcome:
    operation_id: str
    completed_at_utc: datetime | None
    completion_status: str
    deviation_minutes: float | None
    incident_types: tuple[str, ...]
    cost_eur: float | None
    censored: bool
```

Invariantes:

- timestamps originales no se reescriben;
- toda correccion crea nueva revision;
- unidades se conservan y normalizan con trazabilidad;
- IDs se seudonimizan de forma estable;
- outcome futuro nunca entra como feature;
- una foto es referencia a objeto, no blob dentro del log;
- missing es explicito, no cero;
- eventos duplicados se marcan, no se borran silenciosamente.

## Roles de columnas

### Acciones validas

- asignar/cambiar equipo o recurso;
- cambiar secuencia de carga/descarga;
- aprobar pausa/reinicio;
- cambiar ventana o turno;
- reasignar posicion/area;
- solicitar inspeccion adicional;
- actualizar plan de forma timestamped.

### Contexto

- cliente, terminal, buque y muelle;
- tipo, peso y dimensiones de carga;
- turno/dia de semana;
- meteo y estado de mar;
- equipo disponible si no consta una decision;
- procedimiento, puerto y pais;
- complejidad de documentacion.

### Observaciones

- milestones realizados;
- cantidad movida acumulada;
- tiempos entre eventos;
- colas, esperas y pausas;
- incidencias, notas y fotos;
- estado de recursos;
- posicion AIS o de unidad cuando exista.

### Outcomes

- fin real;
- desviacion respecto al plan congelado;
- incidencia posterior;
- dano/reclamacion;
- coste y penalizacion;
- SLA incumplido.

## Object-centric event graph

Un evento puede relacionarse con varios objetos. Ejemplo:

```text
event: CRANE_LIFT_COMPLETED
objects:
  operation=OP-42
  shift=SH-3
  cargo_unit=CU-117
  resource=CRANE-2
  vessel=IMO-...
```

Relaciones principales:

```text
project contains operation
operation scheduled_in shift
operation handles cargo_unit
resource executes event
vessel hosts operation
incident affects cargo_unit/resource/operation
plan_revision governs operation after valid_from
```

El grafo evita duplicar eventos en varias trazas y permite explicar propagacion.

## Escalera de modelos

### M0 - Descriptivo

- calidad/cobertura del log;
- tiempos por actividad;
- variantes y directly-follows graph;
- esperas y rework;
- conformance respecto al flujo acordado.

### M1 - Baselines

- mediana por clase de operacion;
- Kaplan-Meier/Cox si hay censura;
- quantile Gradient Boosting/CatBoost;
- reglas de umbral y control charts;
- regresion/logistica regularizada.

### M2 - Secuencial

- GRU/TCN;
- ProcessTransformer;
- temporal graph model object-centric;
- PGTNet-style remaining-time baseline.

### M3 - Event-JEPA

Estado:

```text
z_t = Encoder(prefix_event_graph_t)
z_hat_t+h = Predictor(z_t, action_prefix, context, horizon)
```

Target:

```text
z_t+h = TargetEncoder(prefix_event_graph_t+h)
```

Heads:

- remaining-time quantiles;
- deviation risk;
- next-event distribution;
- incident risk;
- bottleneck/object attribution.

Implementación pública seleccionada para AIS:

```text
observed AIS prefix --> context encoder -----------+
constant-course future --> physical conditioning   |--> latent future
future AIS state --> stopped target encoder --------+
latent state/forecast + raw features --> trajectory GBT
```

La física conocida no se reconstruye: el decoder aprende el residual respecto
al futuro cinemático. VICReg se eligió en validación tras comparar VISReg,
SIGReg y ausencia de regularizador. Ninguna seed seleccionada colapsó. El
ensemble híbrido pasa el gate de trayectoria/desviación; ETA y retraso conservan
gates independientes y no se promocionan por arrastre.

### M4 - Escenarios

Dos rutas separadas:

1. simulacion discreta calibrada para colas/recursos;
2. ranking con world model de acciones observadas.

La primera funciona con conocimiento experto y poco dato. La segunda exige
acciones reales variadas, soporte y validacion `correct vs shuffled`.

## Cold start

| Datos disponibles | Entregable permitido |
|---|---|
| 0-5 operaciones | schema, instrumentacion, mapa de proceso, simulacion experta |
| 5-30 | distribuciones descriptivas, controles, retrieval de casos similares |
| 30-100 | boosting/quantiles con intervalos amplios y validacion grouped CV |
| 100-500 | modelos secuenciales pequenos, calibracion por grupo |
| >500 y diversidad suficiente | Event-JEPA/graph, ablations y adaptacion |

Los umbrales son orientativos; la diversidad de estados/eventos importa mas que
el numero bruto.

## Split y leakage

Prioridad:

1. holdout cronologico futuro;
2. holdout por proyecto/cliente/puerto;
3. holdout por tipo de carga;
4. grouped cross-validation.

Prohibido:

- dividir eventos de la misma operacion entre train/test;
- usar la revision final del plan como si se conociera al inicio;
- calcular features con eventos posteriores al prediction time;
- seleccionar umbral con test;
- usar fotos/informes generados despues del outcome;
- convertir tiempo-a-fin en feature.

## Evaluacion

### Prediccion

- MAE/MedAE de tiempo restante;
- pinball loss P50/P90;
- cobertura y anchura de intervalos;
- AUPRC de desviacion/incidencia;
- Brier/ECE;
- next-event macro-F1/top-k.

### Operacion

- lead time antes de desviacion;
- falsas alertas por turno/100 operaciones;
- eventos no detectados;
- precision top-10% de operaciones;
- utilidad de explicacion valorada por operador;
- latencia y disponibilidad.

### Negocio

- horas de espera evitables identificadas;
- overtime/demurrage potencial;
- tiempo de reporting/coordinacion;
- cumplimiento de SLA;
- ahorro simulado bajo politica, separado de ahorro realizado.

## Serving

Endpoints read-only candidatos:

```text
POST /v1/events/batch
POST /v1/score/operation-prefix
GET  /v1/operations/{id}/risk
POST /v1/scenarios/rank
GET  /v1/models/{version}/card
GET  /health
```

Cada respuesta incluye:

```text
model_version
data_cutoff
plan_revision
prediction_time
horizon
point_estimate
interval
confidence
abstained
reason_codes
source_event_ids
```

## Seguridad y privacidad

- export read-only y minimo privilegio;
- seudonimizacion antes de salir del entorno acordado;
- cifrado en transito/reposo;
- sin credenciales en artefactos;
- retencion y borrado acordados;
- fotos y notas tratadas como potencial dato personal/confidencial;
- auditoria por prediccion;
- no usar datos de Kaleido para modelos de otros clientes sin acuerdo.

## Despliegue

1. offline replay;
2. shadow dashboard;
3. alertas no accionables visibles a equipo tecnico;
4. revision por operaciones;
5. alertas operativas con runbook;
6. escenarios asesorados;
7. integracion mas profunda solo tras aprobacion y seguridad.
