# Auditoria de reutilizacion de proyectos

Fecha: 15-07-2026. La evaluacion distingue codigo implementado, especificacion y
resultado promocionable.

## Veredicto

No existe un proyecto que pueda presentarse a Kaleido como MVP ya ajustado. La
mejor estrategia es crear un repositorio nuevo y reutilizar componentes
auditados, no copiar un proyecto completo.

| Proyecto | Encaje | Reutilizar | No reutilizar como evidencia |
|---|---|---|---|
| `predictiveops-worldmodel` | Alto como esqueleto | contratos action/context/outcome, leakage, splits, lead time, SOTA gate, tests | la narrativa CNC universal ni el modelo actual como predictor logistico |
| `industrial_jepa_mvp` | Medio como biblioteca de ideas | agregacion jerarquica, reportes, baselines, patrones de dashboard | todas las metricas pre-audit; actualmente 0 resultados claim-eligible |
| `e-jepa-ttc` | Bajo/medio | streaming, incertidumbre, multi-horizonte, cache y export futuro | modelo de camara de eventos; test reutilizado y sin resultado post-fix |
| `softnav-jepa` | Bajo para el piloto | separacion planner/constraints, candidate ranking, safety fallback | resultados v1-v3 invalidados; proxy cinematico, no puerto/robot real |
| `SemanticSegmentation3Dclouds` | Futuro 3D | schemas geograficos, split espacial, metricas, LiDAR | DINO/Point-JEPA como mejora probada; trabajo sucio y bloques corregidos pendientes |
| `LiDAR Foundational Model` | Futuro 3D | plan ALS-first y gates de financiacion | codigo/modelo: hoy es especificacion, no implementacion |
| `LeJEPAenAQilles` | Nulo para producto | diagnosticos de colapso y disciplina experimental | smoke colapsado y dominio molecular |
| `HAS-JEPA` | Ninguno | nada | es una nota de transcripcion musical, no un proyecto logistico |

## `predictiveops-worldmodel`

Estado verificado:

- 88 ficheros de proyecto;
- 43 tests pasan;
- contratos de columnas y validacion action/context implementados;
- adaptadores, leakage, splits, timestamps, lead time y SOTA gate implementados;
- modelo Sensor-JEPA experimental sobre CNC;
- no hay conectores logisticos, OCEL/XES, Trace Port ni simulacion discreta;
- las configuraciones industriales avanzadas descritas en README son en gran
  parte roadmap, no implementacion.

Decision: usarlo como referencia de gobernanza. Portar o extraer modulos pequenos
con tests y licencia, manteniendo un historial de procedencia. No hacer fork
ciego porque su semantica y datasets son industriales/CNC.

## `industrial_jepa_mvp`

Estado verificado:

- demos de sensores CNC y anomalia visual;
- ramas de DenseSensorJEPA, token world model, DINO/PatchCore/PaDiM y jerarquia;
- historial de fugas: hardness repetida entre splits y seleccion de umbral con
  test;
- `STATUS.md` declara 0 resultados post-audit elegibles;
- worktree con artefactos no versionados.

Decision: reutilizar ideas y, tras revision de licencia/API, funciones de
agregacion/reporting. Nunca copiar cifras a la presentacion.

## `e-jepa-ttc`

Estado verificado:

- pipeline real de camara de eventos, representaciones, temporal JEPA y TTC;
- nueve secuencias locales, pero CPLA-high ya fue inspeccionado;
- `STATUS.md` declara 0 metricas post-fix promovibles;
- faltan robustness, ONNX, streaming demo final y report builder completo.

Decision: tomar patrones de ventanas temporales, multi-horizonte e
incertidumbre. El dominio visual asincrono no se traslada a eventos de negocio.

## `softnav-jepa`

Estado verificado:

- buena arquitectura hibrida: stack clasico, restricciones duras, JEPA local;
- datasets v1-v3 y resultados derivados invalidados por inputs privilegiados,
  split/policy y perturbaciones no emparejadas;
- v4 corrige contratos, pero no se ha ejecutado el benchmark completo;
- no hay evidencia de Isaac, ROS, MuJoCo ni robot real.

Decision: conservar el principio `learned ranking never overrides hard
constraints`. Solo aplicaria en una futura fase de automatizacion fisica.

## `SemanticSegmentation3Dclouds` y `LiDAR Foundational Model`

Estado verificado:

- corpus ALS grande y split geografico disciplinado;
- baseline geometrico fuerte; DINO tardio no aporta mejora robusta;
- Point-JEPA actual usa centros por muestreo uniforme aproximado, no FPS real;
- estado declara ausencia de bloques procesados corregidos y pesos DINO reales;
- GeoLiDAR-FM es un plan ALS-first, aun sin implementacion.

Decision: linea futura para inventario/ocupacion/seguridad espacial del patio o
terminal, solo si Kaleido formula un problema 3D y aporta/sensoriza datos.

## Componentes que debe crear FlowTwin

- adaptador Trace Port/API/CSV;
- esquema OCEL-like para proyecto, operacion, turno, carga y recurso;
- versionado de plan y replanificacion;
- process discovery/conformance;
- predictor de tiempo restante y riesgo calibrado;
- simulador discreto sencillo;
- capa de escenarios con acciones verificadas;
- dashboard para prefijos vivos y explicacion;
- evaluacion comercial y tecnica por proyecto/turno.

## Regla de procedencia

Cada modulo copiado debe registrar repositorio, commit, licencia, fichero
original, cambios y tests. Ningun resultado historico se copia: todos los
resultados de FlowTwin se regeneran desde datos FlowTwin y artefactos propios.

