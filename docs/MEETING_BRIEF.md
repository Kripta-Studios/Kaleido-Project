# Guion de reunion: Kaleido FlowTwin

Documento de preparacion para una conversacion tecnica y comercial con Kaleido.
`FlowTwin` es un nombre provisional. La reunion no debe vender un modelo ya
validado: debe conseguir acceso controlado a evidencia suficiente para decidir
si existe un producto rentable.

## La idea en una frase

Convertir los eventos que Kaleido ya registra en Trace Port en una capa
predictiva que estime tiempo restante y riesgo de desviacion, explique donde se
forma el cuello de botella y compare escenarios operativos aprobados, sin tomar
el control de la operacion.

## El mensaje que debe quedar

Kaleido ya tiene la parte dificil que muchos proyectos de IA no tienen: un
producto conectado al trabajo real, eventos con significado operativo y acceso
a usuarios que pueden actuar. No proponemos sustituir Trace Port ni construir
un simulador visual. Proponemos convertir esa trazabilidad en anticipacion.

La falta de un historico rico no se oculta ni se discute: es la primera
restriccion de diseno. La fase inicial mide la calidad y densidad de la evidencia
y entrega process mining, baselines y un simulador discreto aun cuando no haya
datos suficientes para entrenar un modelo profundo.

## Apertura de 60 segundos

> Hemos revisado vuestra observacion sobre el historico y hemos cambiado el
> planteamiento para que el proyecto no dependa de promesas de deep learning.
> Trace Port ya registra proyectos, turnos, eventos, equipos, incidencias y
> tiempos. Nuestra propuesta es empezar por una sola operacion repetible y usar
> esos eventos para reconstruir el proceso real, medir desviaciones y crear un
> baseline de tiempo restante y riesgo a 2, 4 y 8 horas. Solo si un modelo
> secuencial o JEPA mejora ese baseline fuera de muestra se incorpora. El
> resultado seria una capa read-only sobre Trace Port: alerta anticipada,
> intervalo de confianza, cuello de botella y escenarios operativos aprobados.
> En cuatro a seis semanas no prometemos un gemelo autonomo; prometemos una
> decision medible sobre producto, datos y retorno.

## Version de cinco minutos

1. **Problema.** Un dashboard explica que ha pasado; el valor adicional aparece
   cuando permite saber con antelacion si una operacion no va a cumplir el plan.
2. **Activo de Kaleido.** Trace Port estructura la actividad que necesitamos:
   proyecto, packing list, turno, evento, equipo, evidencia e incidencia.
3. **Primer caso.** En una carga, descarga o manipulacion repetible, estimar
   tiempo restante y probabilidad de exceder el plan en horizontes de 2/4/8 h.
4. **Salida.** Riesgo calibrado, intervalo, factores asociados, cuello de
   botella, abstencion si falta evidencia y comparacion de acciones permitidas.
5. **Metodo.** Auditoria temporal, process mining, modelos simples, simulacion de
   eventos discretos y, como candidato, representacion secuencial tipo JEPA.
6. **Prueba honesta.** Separacion por proyecto y tiempo; nada del futuro entra en
   el pasado; se compara con reglas y modelos tabulares; se miden falsas alarmas,
   calibracion y minutos de anticipacion.
7. **Negocio.** Modulo premium de Trace Port, servicio de implantacion y mejora
   interna de planificacion. El precio y el ROI se calculan con el coste real de
   una desviacion y de intervenir, no con porcentajes genericos.
8. **Siguiente paso.** Una sesion de esquema y tres a cinco operaciones
   anonimizadas para producir un informe de viabilidad y un contrato de datos.

## Objetivo concreto de la reunion

Salir con cuatro decisiones:

- propietario de negocio y propietario tecnico;
- una operacion repetible con una decision que todavia pueda cambiarse;
- acceso a esquema y muestra anonimizada, preferiblemente tres a cinco casos;
- fecha de una sesion de 90 minutos para mapear proceso, costes y acciones.

No hace falta conseguir en la primera reunion acceso a produccion, presupuesto
cerrado ni compromiso con JEPA.

## Descubrimiento: las preguntas que importan

### Operacion y decision

- ¿Que operacion se repite con suficiente frecuencia y tiene principio y fin?
- ¿Que plan o SLA se incumple y cuando deja de ser util advertirlo?
- ¿Quien puede actuar ante una alerta y que acciones tiene permitidas?
- ¿Cual es el coste aproximado de retraso, recurso ocioso, hora extra, demurrage,
  penalizacion o replanificacion?
- ¿Que accion es segura pero cara? Ese dato determina el umbral de alerta.

### Evidencia disponible

- ¿Cada evento conserva `event_time`, `ingest_time`, proyecto, turno y actor?
- ¿Se guarda el plan que estaba vigente en ese momento o solo su version final?
- ¿Hay reintentos, correcciones, eventos tardios, duplicados o trabajo offline?
- ¿Se conocen pausas, cambios de equipo y causa de incidencia?
- ¿Que parte existe en Trace Port y cual vive en ERP, TOS, Excel o correo?
- ¿Cuantos proyectos completos hay por tipo de operacion y por terminal?

### Producto e integracion

- ¿El mejor punto de entrada es API, export programado o replica read-only?
- ¿Que interfaz consume el usuario: panel Trace Port, email, push o API?
- ¿Debe desplegarse en infraestructura de Kaleido, nube privada o entorno del
  cliente final?
- ¿Quien comercializaria el modulo y como se factura hoy Trace Port?

## Respuesta a la objecion principal: “no hay historico rico”

Respuesta breve:

> Precisamente por eso proponemos un piloto con salida util antes del modelo. La
> primera entrega mide el proceso, la calidad temporal y que señal existe. Con
> pocos datos usamos reglas, pooling por familias de operacion, simulacion y
> estimadores con intervalos amplios. Si la evidencia no soporta prediccion, el
> resultado es un no-go fundamentado y un plan de instrumentacion, no una demo
> que inventa precision.

Consecuencias tecnicas:

- el historico se mide en casos completos y densidad de eventos, no solo meses;
- se agrupan operaciones solo si comparten mecanismo y contrato semantico;
- el modelo se abstiene cuando el caso esta fuera de distribucion;
- un log publico o sintetico valida tuberias, nunca el valor para Kaleido;
- los datos nuevos se capturan con un contrato que permita aprendizaje futuro.

## Que significa aqui “world model”

No es un generador de video ni una maqueta 3D. Es un modelo del estado operativo
que intenta responder:

1. donde esta ahora la operacion;
2. que puede ocurrir a continuacion;
3. con que incertidumbre y en que plazo;
4. como cambia la distribucion si se modifica una accion permitida.

Formalmente, para estado historico `h_t`, contexto `c_t` y accion candidata
`a_t`, se estima una distribucion futura:

`p(y_{t+h}, z_{t+h} | h_t, c_t, a_t)`

`z` es un estado latente; `y` contiene tiempos, eventos e incidencias
observables. Sin variacion real o una simulacion validada de `a_t`, el sistema
solo puede mostrar asociacion o escenario, no efecto causal.

## Por que JEPA, y por que no es un requisito

JEPA aprende a predecir representaciones de partes futuras u ocultas, evitando
reconstruir cada detalle del dato. Puede ser util cuando hay muchos eventos sin
etiquetas limpias, distintas cadencias y estados parcialmente observados. Para
Kaleido el candidato seria un Event-JEPA temporal, no V-JEPA de video.

No obstante, el modelo ganador puede ser Kaplan-Meier, gradient boosting o una
red secuencial pequena. JEPA entra solo si aporta una mejora estable en holdout,
calibracion y anticipacion a coste operativo comparable. La arquitectura de
producto no depende de esa eleccion.

## Arquitectura que conviene explicar en la pizarra

```text
Trace Port / ERP / TOS / meteo
              |
              v
     eventos versionados + calidad temporal
              |
       estado operativo a tiempo t
              |
   +----------+----------+
   |          |          |
baseline   Event-JEPA   simulacion discreta
   |          |          |
   +------> riesgo, tiempo, intervalo
                    |
          politica read-only + abstencion
                    |
      Trace Port / API / informe de turno
```

Puntos tecnicos que generan confianza:

- `event_time` y `ingest_time` se conservan por separado;
- los planes se versionan para no usar el plan final al evaluar el pasado;
- el split es por proyecto/tiempo/terminal, nunca por filas aleatorias;
- la prediccion se hace como si solo se supiera lo disponible en ese instante;
- cada salida conserva modelo, version de datos y explicacion;
- no hay escritura automatica sobre sistemas operativos o de seguridad.

## Prueba y criterios de salida

El baseline no es “acertar la media”. Incluye reglas operativas, mediana por
familia, survival analysis y boosting tabular. El candidato debe probar valor
incremental en un holdout congelado.

Metricas necesarias:

- MAE y pinball loss del tiempo restante;
- AUCPR para eventos raros, no solo accuracy o AUROC;
- Brier score y error de calibracion;
- lead time antes de la desviacion;
- falsas alertas por operacion o turno;
- cobertura y anchura de intervalos;
- rendimiento por tipo de carga, terminal y turno;
- utilidad economica a distintos costes de intervenir y no intervenir.

Gates:

- **go:** mejora repetible y alerta con tiempo accionable;
- **instrument:** el proceso tiene valor, pero faltan campos o consistencia;
- **no-go:** no hay una decision accionable, no se puede evitar fuga temporal o
  el modelo no supera reglas simples.

## Encaje con los productos de Kaleido

### Trace Port: primer modulo

“Predictive Operations” sobre la pantalla de proyecto/turno: tiempo restante,
riesgo, confianza, cuello de botella y enlace a evidencia. Es el mejor lugar de
inicio porque ya dispone de eventos operativos y usuario en contexto.

### Shipping Board: segunda extension

Riesgo de retraso o excepcion en expedicion/recepcion, con sincronizacion de
documentos y eventos externos. Se reutiliza el motor, pero cambia el contrato de
entidades y horizontes.

### Freight Intelligence: tercera extension

ETA probabilistica y riesgo de excepcion de contenedor. Requiere separar eventos
del carrier, estimaciones externas y conocimiento disponible en cada instante.

## Hipotesis de negocio

No presentar una cifra de precio cerrada sin conocer empaquetado ni costes. Si
Kaleido confirma valor, hay cuatro vias compatibles:

- add-on SaaS por terminal, proyecto o volumen de operaciones;
- implantacion y calibracion inicial por cliente;
- white-label dentro de Trace Port;
- ahorro interno en planificacion, supervision y revision de operativas.

Formula para construir el caso:

`valor esperado = desviaciones evitadas + horas ahorradas + mejor uso de recursos - coste de intervenciones - coste del modulo`

El piloto debe registrar decisiones y costes para calcular esta expresion con
datos de Kaleido.

## Objeciones y respuestas

### “Ya tenemos dashboards y alertas”

FlowTwin no duplica visibilidad. La pregunta es si la alerta llega antes, esta
calibrada y permite una accion que reduzca coste. Si una regla existente lo hace
igual de bien, esa regla es el producto correcto.

### “¿Esto es un gemelo digital?”

Es un gemelo operativo ligero: estado, transicion, incertidumbre y escenarios.
No pretende reproducir fisica, geometria o video del puerto en la primera fase.

### “¿La recomendacion es causal?”

No por defecto. Sin datos de acciones, aleatorizacion o una simulacion validada,
se etiquetara como asociacion o escenario. Las decisiones de alto impacto siguen
siendo humanas.

### “¿Por que no usar directamente un LLM?”

Un LLM puede resumir informes o explicar resultados, pero no sustituye un modelo
temporal calibrado ni soluciona la fuga de informacion. Puede ser interfaz, no
fuente de verdad numerica.

### “¿Necesitamos enviar datos sensibles?”

No necesariamente. Se puede empezar con esquema, perfiles agregados y muestra
pseudonimizada; despues desplegar cerca del dato. Se eliminan nombres y texto
libre no necesario, y se acuerdan retencion, roles y finalidad.

### “¿Cuanto tarda?”

Cuatro a seis semanas para llegar a una decision de producto si el acceso y el
responsable operativo estan disponibles. No equivale a integracion de produccion
ni garantiza que un modelo avanzado supere el baseline.

## Agenda recomendada de 60 minutos

| Minutos | Tema | Resultado |
|---:|---|---|
| 0-5 | Objetivo y restriccion de datos | Alineamiento |
| 5-15 | Demo o recorrido de una operacion | Flujo real |
| 15-30 | Eventos, planes, incidencias y sistemas | Mapa de evidencia |
| 30-40 | Decision, acciones y coste | Target accionable |
| 40-50 | Propuesta y gates | Alcance del scan/piloto |
| 50-60 | Propietarios, muestra y fechas | Siguiente paso |

## Que mostrar y que no mostrar

Mostrar:

- el diagrama de la arquitectura y la escalera de cold start;
- un ejemplo ficticio claramente marcado como tal;
- criterios de go/no-go y peticion minima de datos;
- que el sistema complementa los productos actuales de Kaleido.

No mostrar como evidencia:

- metricas antiguas de `industrial_jepa_mvp`, `e-jepa-ttc` o `softnav-jepa`;
- una precision sin prevalencia, holdout y baseline;
- resultados de datasets publicos como prueba de ROI para Kaleido;
- “causal”, “autonomo”, “SOTA” o “gemelo completo” sin prueba.

## Lectura de los repositorios existentes

- `predictiveops-worldmodel` aporta contratos, manifests, splits, tests y gates.
- `industrial_jepa_mvp` aporta patrones de agregacion y reporting, no evidencia
  transferible a logistica.
- `e-jepa-ttc` aporta ventanas temporales, horizontes y streaming.
- `softnav-jepa` aporta la separacion entre estimador y politica segura.
- `SemanticSegmentation3Dclouds` queda como opcion futura si aparece un caso
  LiDAR/3D, no para el MVP de eventos.

Ninguno es una aplicacion lista para Trace Port. Reutilizar indiscriminadamente
sus modelos introduciria supuestos de sensores, labels y splits incompatibles.

## Cierre de la reunion

> Para no pediros un proyecto de IA a ciegas, proponemos empezar por una muestra
> pequena y una operacion concreta. En una semana os devolvemos el mapa de datos,
> la definicion exacta del target, los riesgos y un diseno de evaluacion. Con eso
> decidimos juntos si merece la pena ejecutar el piloto de cuatro a seis semanas
> o si lo correcto es primero mejorar instrumentacion.

## Lista personal antes de entrar

- Llevar la presentacion PDF y la version HTML offline.
- Tener abiertos `DATA_REQUEST.md` y `ARCHITECTURE.md`.
- Saber explicar diferencia entre evento, caso, objeto y snapshot.
- Preguntar, no asumir, que Trace Port conserva historial de versiones del plan.
- No discutir algoritmo antes de fijar decision y coste.
- Anotar vocabulario exacto de Kaleido para estados, incidencias y recursos.
- Terminar con una peticion pequena y fechada.

## Glosario rapido

- **Caso:** una ejecucion completa que puede evaluarse, por ejemplo una operacion.
- **Objeto:** entidad que participa en varios eventos, como proyecto, turno,
  equipo, buque o unidad de carga.
- **OCEL:** log de eventos centrado en objetos; evita forzar todo a un unico caso.
- **Process mining:** reconstruccion y comparacion del proceso real desde eventos.
- **Conformance:** medida de divergencia entre proceso observado y esperado.
- **World model:** modelo de estado y futuros posibles condicionados al contexto.
- **JEPA:** prediccion en espacio de representaciones, no de cada detalle crudo.
- **Calibracion:** que un riesgo del 20 % ocurra aproximadamente dos de cada diez
  veces en poblaciones comparables.
- **Lead time:** anticipacion entre alerta y evento objetivo.
- **Abstencion:** no emitir una recomendacion cuando la evidencia no es fiable.
- **Holdout:** conjunto de evaluacion nunca usado para seleccionar el modelo.
