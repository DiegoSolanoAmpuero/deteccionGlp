# Gemelo digital experimental GLP

## Ejecutar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Qué hace
- Genera concentraciones sintéticas de GLP (no mediciones reales).
- Introduce ruido y aplica un filtro de Kalman escalar causal.
- Entrena RandomForestRegressor e GradientBoostingRegressor para predecir concentración a t+h.
- Compara métricas en el último 25 % temporal.
- Muestra una luz de emergencia **virtual** y opcionalmente publica su estado vía MQTT TLS.

## Publicación en nube
Subir app.py y requirements.txt a un repositorio privado/público según corresponda y desplegar en Streamlit Community Cloud u otro hosting Python. La app se ejecuta en la nube, no adquiere señales reales sin adaptar el backend. Para un sistema multiusuario y telemetría real, usar broker MQTT, backend persistente, base de datos temporal, autenticación y procesos de inferencia separados de Streamlit.

Para broker MQTT TLS opcional, configurar `.streamlit/secrets.toml`:

```toml
[mqtt]
username = "USUARIO"
password = "CLAVE"
```

## Seguridad
No conectar directamente una sirena, relé o luz física al resultado de esta app. Una alarma certificada independiente debe actuar por detección real, incluso sin Internet o sin modelo ML. El umbral configurable (35 ppm por defecto) es únicamente didáctico, no un setpoint de seguridad validado.

## Limitaciones metodológicas
La señal es sintética y la ecuación de acumulación no está calibrada contra los datos del artículo. La predicción del futuro se evalúa con hold-out temporal. Para investigación, comparar varios escenarios y semillas, usar validación temporal walk-forward, calibrar sensor MQ-6 y contrastar con equipo de referencia, y evaluar retraso de detección, falsas alarmas y sensibilidad.

## Entrega al profesor
1. Crear repositorio de GitHub y subir `app.py`, `requirements.txt`.
2. Abrir https://share.streamlit.io, seleccionar **Create app**, repositorio, rama y archivo `app.py`.
3. Esperar despliegue, copiar la URL pública de la aplicación.
4. Ajustar umbral/intensidad si hace falta; comprobar que en el gráfico de predicción aparece un tramo **rojo** y la luz indica **ALERTA**.
5. Tomar captura con `Windows + Shift + S` mostrando el gráfico rojo y la luz roja. Enviar URL y captura.

No subir los archivos de leche `output_file*.csv`: corresponden a otra aplicación y no sirven para calibrar GLP.
