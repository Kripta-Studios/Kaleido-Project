# Peticion minima de datos para la reunion

## Objetivo

Determinar en menos de una semana si existe un caso predictivo viable, sin
solicitar un volcado masivo ni acceso productivo.

## Paquete minimo

1. Diccionario de campos/export de Trace Port.
2. Tres a cinco operaciones cerradas anonimizadas:
   - una nominal;
   - una lenta/desviada;
   - una con incidencia si existe.
3. Plan original y revisiones disponibles.
4. Eventos/timestamps y relaciones con operacion, turno, carga y recurso.
5. Resultado final: fin, desviacion e incidencia.
6. Captura o export de dashboard solo si no expone informacion sensible.
7. Descripcion de que decision se toma cuando aparece riesgo.

Formatos aceptables: CSV, XLSX, JSON, Parquet o export de API. No se necesita
base de datos ni credenciales para el scan.

## Seudonimizacion

Kaleido puede sustituir:

- cliente, proyecto, buque, persona y recurso por IDs estables;
- notas por categorias o texto redactado;
- fotos por metadata o excluirlas inicialmente;
- importes por rangos.

No debe eliminar timestamps, relaciones ni revisiones, porque son la senal del
proceso.

## Preguntas por tabla/export

- ¿Que representa una fila?
- ¿Cual es la clave primaria?
- ¿Que timezone usa?
- ¿El timestamp es de ocurrencia o registro?
- ¿Puede haber duplicados/reintentos?
- ¿Se actualiza una fila o se crea historial?
- ¿Que campos existian en el momento del evento?
- ¿Que outcome se conoce solo al final?
- ¿Que IDs enlazan tablas?
- ¿Que valores son decisiones y quienes pueden cambiarlos?

## Conteos que bastan antes del volcado

- numero de proyectos/operaciones/turnos por mes;
- fecha minima/maxima;
- eventos por operacion;
- porcentaje con plan y fin real;
- porcentaje con incidencias;
- tipos de carga/operacion;
- numero de puertos/clientes;
- frecuencia de replanificacion;
- cobertura de recursos/equipos.

## Respuesta del Data Evidence Scan

EVOCON devolvera:

- mapa de datos;
- score de readiness por objetivo;
- issues y acciones de instrumentacion;
- caso candidato y baseline;
- estimacion de esfuerzo/computo;
- `GO_PREDICTIVE`, `INSTRUMENT_FIRST` o `NO_FIT`;
- propuesta cerrada del piloto, sin obligacion de continuar.

## Lo que no se solicita inicialmente

- acceso de escritura;
- credenciales productivas;
- datos personales sin necesidad;
- fotos originales;
- todos los productos a la vez;
- anos de datos sin saber la estructura;
- aprobacion para controlar operaciones.

