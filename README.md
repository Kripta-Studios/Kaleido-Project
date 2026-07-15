# Kaleido FlowTwin

Especificacion de un MVP de inteligencia operativa para Kaleido Tech. El nombre `FlowTwin` es provisional y no presupone aprobacion de marca.

## Veredicto ejecutivo

Ninguno de los repositorios existentes encaja de extremo a extremo con el caso
de Kaleido. El punto de partida mas solido es reutilizar la gobernanza, los
contratos de datos y los gates de `predictiveops-worldmodel`; el modelo y los
conectores deben ser nuevos y orientados a eventos logisticos.

La oportunidad recomendada es una capa predictiva que amplie los productos ya
existentes de Kaleido:

- `Trace Port`: riesgo de desviacion, tiempo restante y cuello de botella por
  proyecto, operacion y turno.
- `Shipping Board`: riesgo de retraso de expediciones y recepciones.
- `Freight Intelligence`: riesgo de excepcion y ETA probabilistica.

El primer piloto debe ejecutarse sobre **una operacion repetible de Trace Port**.
No se promete entrenar un world model si el historico no lo permite. El piloto
produce valor en una escalera de evidencia:

1. auditoria y contrato de datos;
2. process mining y conformance checking;
3. baselines de tiempo restante y riesgo;
4. simulacion discreta para escenarios;
5. modelo secuencial o JEPA solo si mejora los baselines en holdout.

## Pregunta de negocio

> Dado el estado actual de una carga, descarga o manipulacion portuaria, el plan,
> los recursos y el contexto, ¿cual es la probabilidad de desviacion en las
> proximas 2/4/8 horas, cuanto falta para terminar, donde esta el cuello de
> botella y que accion permitida reduce el riesgo?

La salida es asesoramiento read-only. No escribe en TOS, ERP, PLC, equipos ni
sistemas de seguridad.

## Por que encaja con Kaleido

Kaleido ya digitaliza la operacion mediante Trace Port, Shipping Board y Freight
Intelligence. FlowTwin convierte los eventos que esas herramientas generan en
una capacidad adicional:

- pasar de visibilidad a anticipacion;
- reducir supervision manual y deteccion tardia;
- cuantificar lead time y falsas alarmas;
- ofrecer un modulo premium a terminales y clientes;
- mejorar estimacion, planificacion y revision post-operacion;
- crear un activo de datos acumulativo sin exigir una integracion invasiva.

Fuentes corporativas auditadas a 15-07-2026:

- <https://www.kaleidologistics.com/es/kaleido-tech/>
- <https://www.kaleidologistics.com/en/productos-ktech/trace-port/>
- <https://www.kaleidologistics.com/en/productos-ktech/shipping-board/>
- <https://www.kaleidologistics.com/en/productos-ktech/freight-intellligence/>
- <https://www.kaleidologistics.com/en/port-operations/>

## Alcance del MVP

### Entrada minima

- identificador de proyecto/operacion/turno;
- plan previsto y milestones;
- eventos con timestamp;
- unidades de carga o lineas de packing list;
- equipo/recurso/rol cuando exista;
- inicio, pausa, incidencia y fin;
- contexto: tipo de carga, buque, muelle, clima y turno;
- resultado: duracion final, desviacion e incidencias;
- costes solo si Kaleido quiere traducir riesgo a euros.

### Salidas

- tiempo restante con intervalo;
- riesgo de exceder plan en 2/4/8 horas;
- riesgo de incidencia, solo si hay etiquetas suficientes;
- siguiente estado/evento probable;
- cuello de botella y factores asociados;
- nivel de confianza y abstencion;
- escenarios what-if dentro de acciones aprobadas;
- dashboard y export JSON/CSV/PDF.

### No incluido en la primera version

- control automatico;
- vision generativa o video sintetico;
- gemelo fisico 3D del puerto;
- recomendacion causal sin acciones verificadas;
- entrenamiento de un foundation model desde cero;
- afirmaciones SOTA o ROI no medido.

## Documentos

- [MEETING_BRIEF.md](docs/MEETING_BRIEF.md): guion completo para la reunion.
- [KALEIDO_RESEARCH.md](docs/investigacion/KALEIDO_RESEARCH.md): labor, productos y encaje de Kaleido.
- [REPO_REUSE_AUDIT.md](docs/investigacion/REPO_REUSE_AUDIT.md): que reutilizar y que no.
- [SOTA_2026.md](docs/investigacion/SOTA_2026.md): sintesis tecnica y decisiones derivadas.
- [REFERENCES.bib](docs/investigacion/REFERENCES.bib): bibliografia de trabajo.
- [ARCHITECTURE.md](ARCHITECTURE.md): arquitectura logica y contratos.
- [DATASETS.md](DATASETS.md): fuentes internas, publicas, licencias y uso permitido.
- [DATA_REQUEST.md](DATA_REQUEST.md): peticion minima para evaluar el encaje.
- [PLAN.md](PLAN.md): hitos, gates y criterios de aceptacion.
- [AGENTS.md](AGENTS.md): contrato de implementacion para un agente.
- [Kaleido_FlowTwin_Presentacion.tex](presentacion/Kaleido_FlowTwin_Presentacion.tex): fuente Beamer.
- [Kaleido_FlowTwin_Presentacion.pdf](presentacion/Kaleido_FlowTwin_Presentacion.pdf): presentacion PDF.
- [Kaleido_FlowTwin_Presentacion.html](presentacion/Kaleido_FlowTwin_Presentacion.html): presentacion web offline.
- [CorreoEnviado2.txt](correspondencia/CorreoEnviado2.txt): respuesta propuesta a Manuel.

## Estructura

```text
Kaleido-Project/
|-- README.md, AGENTS.md, PLAN.md        # entrada y contrato de desarrollo
|-- ARCHITECTURE.md, DATASETS.md         # especificacion tecnica
|-- DATA_REQUEST.md                      # peticion minima a Kaleido
|-- correspondencia/                     # correos y adjunto original
|-- presentacion/                        # PDF, HTML y fuente TeX
`-- docs/                                # guion e investigacion
```

## Estado

`proposal_ready`, sin datos internos de Kaleido y sin resultados de modelo.

Todo numero de rendimiento mostrado en una futura reunion debe proceder de un
artefacto generado sobre un split congelado. Los resultados historicos
invalidados de otros repositorios no se pueden reutilizar como prueba.
