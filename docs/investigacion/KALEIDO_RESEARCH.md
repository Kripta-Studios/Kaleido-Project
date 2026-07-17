# Kaleido: investigacion de encaje

Fecha de corte: 15-07-2026. Se han priorizado paginas oficiales de Kaleido.

## Actualizacion experimental 17-07-2026

La demostracion publica principal se ha alineado con Shipping Board/Freight
Intelligence: ETA a geofence con NOAA AIS. En un test futuro agrupado de 85
viajes, boosting obtiene 1.875 h de MAE frente a 2.726 h de una mediana
puerto-distancia y 7.786 h de distancia/velocidad, y pasa 6/6 gates
predeclarados. El resultado es `smoke_only`: no prueba precision Kaleido/Vigo.

OCEL 2.0 Container Logistics demuestra process intelligence y relaciones; su
grafo mejora test pero no fue seleccionado en validacion.

El nuevo Port Call Deviation Twin encuentra el encaje específico para JEPA:
dinámica física multihorizonte, no un segundo predictor tabular de ETA. En un
holdout futuro limpio de 57 viajes, trajectory GBT obtiene 2,635 km MAE y el
ensemble GBT + Phys-JEPA 2,326 km (mejora 11,72%; IC95% por viaje
5,90%-17,13%). AUPRC de desviación sube 0,880 a 0,904 y 0/3 seeds colapsan.

Esto se relaciona directamente con Shipping Board/Freight Intelligence:
anticipar si la trayectoria observada se separa materialmente de la aproximación
física y entregar una excepción auditable a Trace Port. No sustituye ETA: el
probe escaso mejora solo 0,59%, y delay AUPRC retrocede. El core público es
`claim_eligible`, pero el gate completo y el gate Kaleido permanecen cerrados.

## Lo que hace Kaleido

Kaleido Ideas & Logistics opera soluciones integradas por mar, tierra y aire y
combina transporte, aduanas, almacenaje, trincaje, operaciones portuarias,
ingenieria e innovacion. Sus proyectos muestran carga de proyecto y breakbulk de
alta complejidad: granito, estructuras eolicas, material ferroviario, equipos
industriales, buques y cargas sobredimensionadas.

La terminal de Vigo declarada por Kaleido tiene 50.000 m2, almacen interior y
exterior, zonas refrigeradas y servicios de carga, descarga, manipulacion,
agencia, trincaje, soldadura, aduanas, ingenieria, transporte y chartering.

Fuentes:

- <https://www.kaleidologistics.com/en/>
- <https://www.kaleidologistics.com/en/port-operations/>
- <https://www.kaleidologistics.com/en/engineering-solutions/>

## Lo que ya ha construido Kaleido Tech

### Trace Port

Plataforma web y movil para registrar, analizar y coordinar operaciones
portuarias. Incluye proyectos, packing lists, turnos, equipos, eventos,
fotografias, danos/incidencias, informes, timestamps, historial y API.

Consecuencia para FlowTwin: es el mejor punto de entrada. Ya instrumenta la
traza operacional que necesita un predictor y el valor se puede devolver dentro
de la misma experiencia de usuario.

Fuente: <https://www.kaleidologistics.com/en/productos-ktech/trace-port/>

### Freight Intelligence

Centraliza seguimiento de contenedores, integra mas de 180 navieras segun la
pagina del producto, aporta posicion y estado 24/7, alertas, informes e
integracion con sistemas de gestion.

Consecuencia: es una segunda vertical para excepciones/ETA, pero compite en un
mercado mas poblado y gran parte de la senal depende de terceros. Conviene
abordarla despues de demostrar el patron en Trace Port.

Fuente:
<https://www.kaleidologistics.com/en/productos-ktech/freight-intellligence/>

### Shipping Board

Coordina expediciones y recepciones entre almacen, transporte y logistica,
digitaliza documentos, automatiza avisos y exporta estadisticas.

Consecuencia: ofrece secuencias de eventos utiles para predecir tiempo restante,
atascos y retrasos de recepcion/expedicion.

Fuente: <https://www.kaleidologistics.com/en/productos-ktech/shipping-board/>

### Karbon Track

Calcula y reporta CO2e por modo y etapa, con importacion Excel/CSV/API. Es un
producto de reporting y cumplimiento. FlowTwin podria usar sus costes de
emision como objetivo secundario de escenarios, no como primer caso.

Fuente: <https://www.kaleidologistics.com/en/productos-ktech/karbon-track/>

## Lectura del correo recibido

Manuel Alonso Montenegro indica que Kaleido no cuenta con un historico de
operativas muy rico. Esto no es un rechazo; es el principal requisito de diseno.

Interpretacion tecnica:

- no hay base para prometer un gran modelo supervisado;
- el primer entregable debe medir riqueza, cobertura y trazabilidad;
- una unidad de negocio con herramientas propias puede empezar a generar mejor
  dato de forma prospectiva;
- los eventos sin etiqueta siguen siendo utiles para process mining,
  estimacion descriptiva y pretraining autosupervisado;
- un predictor solo se promueve si mejora baselines simples bajo holdout.

Interpretacion comercial:

- Kaleido conoce el problema y valora el encaje, pero no comprara investigacion
  abstracta;
- necesita un piloto que pueda terminar honestamente en tres resultados: `go`,
  `instrument first` o `no signal`;
- la propuesta debe mejorar su catalogo y generar una nueva linea vendible, no
  duplicar una funcionalidad que ya posee.

## Caso inicial recomendado

### Riesgo de desviacion de operacion/turno en Trace Port

Unidad: una operacion o turno de carga, descarga o manipulacion.

Pregunta:

> Con el prefijo de eventos observado hasta ahora, el plan y el contexto, ¿se
> terminara dentro del plan y que probabilidad existe de una desviacion material
> en las proximas 2/4/8 horas?

Subsalidas:

- tiempo restante P50/P90;
- milestone siguiente y hora probable;
- probabilidad de exceder plan;
- cuello de botella actual;
- factor de riesgo: recurso, secuencia, espera, incidencia, clima o documentacion;
- recomendacion dentro de una lista validada por Kaleido.

Por que este caso:

- utiliza el dato que Trace Port ya declara capturar;
- el resultado es comprensible por operaciones;
- se puede evaluar con duracion, lead time y falsas alarmas;
- puede reducir sobrecoste, horas improductivas y coordinacion manual;
- se empaqueta como modulo premium del producto existente;
- el mismo esquema se transfiere luego a Shipping Board.

## Opciones descartadas como primer piloto

### Un world model generativo de video

Demasiado dato y computo, sin relacion directa con el cuello de botella
manifestado. Generar imagenes plausibles no demuestra fidelidad operacional.

### Gemelo 3D LiDAR del puerto

Interesante para inventario, seguridad espacial y simulacion de maniobras, pero
requiere sensores, mapas y un problema 3D concreto. No resuelve el cold start de
eventos y debe ser una fase futura.

### Robotica autonoma

SoftNav-JEPA aporta ideas, pero el stack actual es un proxy cinematico y sus
resultados pre-fix estan invalidados. No es un producto listo para puerto.

### ETA de contenedores como primer caso

Factible gracias a Freight Intelligence, pero menos diferencial que explotar el
conocimiento de operacion portuaria y de carga de proyecto de Kaleido.

## Como gana dinero Kaleido

Hipotesis comerciales a validar con Manuel:

1. modulo SaaS premium de `predictive operations` por terminal/proyecto;
2. add-on white-label para clientes de Trace Port;
3. servicio de implantacion y calibracion por proceso;
4. auditoria post-operacion y benchmarking de recursos;
5. mejora de ofertas y planificacion mediante distribuciones de duracion;
6. menor coste interno y mejor cumplimiento de SLA en la terminal de Vigo;
7. diferenciacion de servicios de ingenieria y operacion con evidencia digital.

No se debe poner precio ni prometer ahorro antes de conocer volumen, coste por
hora, penalizaciones, demurrage, productividad y comprador real.

## Preguntas de descubrimiento

- ¿Cual de los tres productos tiene mas operaciones finalizadas y timestamps?
- ¿Trace Port guarda cada evento o solo el informe final?
- ¿Que constituye una desviacion economicamente material?
- ¿Se almacena el plan original y cada replanificacion?
- ¿Se pueden enlazar packing list, operacion, turno, recurso e incidencia?
- ¿Que decisiones se pueden cambiar durante una operacion?
- ¿Quien compraria el modulo: Kaleido interno, terminal, cargador o cliente final?
- ¿Que coste tiene una hora de retraso, una espera y una falsa alarma?
- ¿Se puede hacer una exportacion anonimizada de 5 operaciones esta semana?
