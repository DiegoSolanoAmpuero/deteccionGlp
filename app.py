import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Gemelo digital GLP (simulación)',layout='wide',page_icon='🚨')
st.title('Gemelo digital de GLP | Random Forest vs. Gradient Boosting')
st.warning('SIMULACIÓN EDUCATIVA. No es un detector de gas certificado ni debe controlar alarmas físicas. Utilice un detector y sistema de alarma certificados independientes.')
st.caption('Señales sintéticas inspiradas en variables del artículo MQ-6; NO son mediciones experimentales ni un modelo físico validado.')

with st.sidebar:
    st.header('Configuración de simulación')
    n=st.slider('Duración (segundos)',120,1200,480,60)
    leak_start=st.slider('Inicio de fuga (s)',10,n-40,80)
    leak_end=st.slider('Fin de fuga (s)',leak_start+10,n-1,min(n-1,leak_start+180))
    source=st.slider('Intensidad de fuga (ppm/s, ficticia)',0.2,4.0,2.5,0.1)
    ventilation=st.slider('Disipación por segundo',0.001,0.10,0.018,0.001)
    noise=st.slider('Desviación de ruido (ppm)',0.0,30.0,9.0,0.5)
    distance=st.select_slider('Distancia (cm)',options=[40,50,60],value=50)
    height=st.select_slider('Altura (cm)',options=[22,42,62],value=42)
    angle=st.select_slider('Ángulo (°)',options=[-30,0,45],value=0)
    horizon=st.slider('Horizonte de predicción (s)',1,20,10)
    threshold=st.slider('Umbral didáctico de alarma (ppm)',10,300,35)
    q=st.number_input('Kalman: varianza de proceso Q',min_value=0.001,value=1.5,step=0.5)
    r=st.number_input('Kalman: varianza de medición R',min_value=0.001,value=81.0,step=5.0)
    seed=st.number_input('Semilla aleatoria',value=42,step=1)
    use_mqtt=st.checkbox('Publicar estado por MQTT (solo visualización)',value=False)
    mqtt_host=st.text_input('Host MQTT',value='') if use_mqtt else ''
    mqtt_topic=st.text_input('Topic',value='demo/glp/estado') if use_mqtt else ''
    mqtt_port=st.number_input('Puerto MQTT TLS',value=8883,min_value=1,max_value=65535) if use_mqtt else 8883

@st.cache_data(show_spinner=False)
def simulate(n,leak_start,leak_end,source,ventilation,noise,distance,height,angle,seed):
    rng=np.random.default_rng(seed)
    t=np.arange(n,dtype=int)
    # Factores exclusivamente ilustrativos: no calibrados con el artículo.
    geo=(50/distance)**1.15 * (1.0 if height==42 else 0.72) * (1.0 if angle==0 else 0.83)
    true=np.zeros(n)
    for i in range(1,n):
        influx=source*geo if leak_start<=i<leak_end else 0
        true[i]=max(0,true[i-1]*(1-ventilation)+influx)
    measured=np.maximum(0,true+rng.normal(0,noise,n))
    return pd.DataFrame({'segundo':t,'concentracion_real_simulada':true,'lectura_ruidosa':measured})

def kalman_filter(values,q,r):
    x=float(values[0]); p=max(r,1.)
    out=[]
    for z in values:
        p=p+q
        k=p/(p+r)
        x=x+k*(float(z)-x)
        p=(1-k)*p
        out.append(x)
    return np.array(out)

def make_features(df,signal,horizon):
    s=pd.Series(df[signal].to_numpy())
    X=pd.DataFrame({'nivel':s,'lag1':s.shift(1),'lag3':s.shift(3),'lag5':s.shift(5),
                    'media5':s.rolling(5).mean(),'media12':s.rolling(12).mean(),
                    'pendiente3':s-s.shift(3),'pendiente8':s-s.shift(8),
                    'desv5':s.rolling(5).std().fillna(0)})
    target=df['concentracion_real_simulada'].shift(-horizon)
    data=X.copy();data['objetivo']=target;data['segundo']=df['segundo']
    return data.dropna().reset_index(drop=True)

def evaluate(df,horizon):
    results=[];preds={}
    for sig,label in [('lectura_ruidosa','Sin Kalman'),('lectura_kalman','Con Kalman')]:
        data=make_features(df,sig,horizon)
        split=int(len(data)*.75)
        train=data.iloc[:split];test=data.iloc[split:]
        features=[c for c in data.columns if c not in ('objetivo','segundo')]
        for model_name,model in [('Random Forest',RandomForestRegressor(n_estimators=160,min_samples_leaf=3,random_state=42,n_jobs=-1)),
                                 ('Gradient Boosting',GradientBoostingRegressor(n_estimators=160,max_depth=3,learning_rate=0.05,random_state=42))]:
            model.fit(train[features],train['objetivo'])
            pred=np.maximum(0,model.predict(test[features]))
            y=test['objetivo'].to_numpy()
            results.append({'Modelo':model_name,'Filtro':label,'MAE (ppm)':mean_absolute_error(y,pred),
                            'RMSE (ppm)':np.sqrt(mean_squared_error(y,pred)),
                            'R²':r2_score(y,pred)})
            preds[(model_name,label)]=pd.DataFrame({'segundo':test['segundo'].to_numpy(),
                      'real_futuro':y,'prediccion':pred})
    return pd.DataFrame(results),preds

if st.button('Actualizar simulación y modelos',type='primary') or 'results' not in st.session_state or st.session_state.get('config') != (n,leak_start,leak_end,source,ventilation,noise,distance,height,angle,horizon,q,r,int(seed)):
    df=simulate(n,leak_start,leak_end,source,ventilation,noise,distance,height,angle,int(seed))
    df['lectura_kalman']=kalman_filter(df['lectura_ruidosa'].values,q,r)
    results,preds=evaluate(df,horizon)
    st.session_state['config']=(n,leak_start,leak_end,source,ventilation,noise,distance,height,angle,horizon,q,r,int(seed))
    st.session_state['df']=df
    st.session_state['results']=results
    st.session_state['preds']=preds
    st.session_state['horizon']=horizon
    st.session_state['threshold']=threshold

if 'results' in st.session_state:
    df=st.session_state['df'];results=st.session_state['results'];preds=st.session_state['preds']
    st.subheader('1. Señales de concentración')
    fig=go.Figure()
    for col,label in [('concentracion_real_simulada','Concentración real sintética'),('lectura_ruidosa','Lectura con ruido'),('lectura_kalman','Estimación Kalman')]:
        fig.add_trace(go.Scatter(x=df.segundo,y=df[col],name=label,mode='lines'))
    fig.add_hline(y=threshold,line_dash='dash',annotation_text='Umbral didáctico')
    fig.update_layout(xaxis_title='Tiempo (s)',yaxis_title='ppm',height=380)
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('2. Comparación predictiva (último 25 % de la serie)')
    st.dataframe(results.style.format({'MAE (ppm)':'{:.2f}','RMSE (ppm)':'{:.2f}','R²':'{:.3f}'}),hide_index=True,use_container_width=True)
    best=results.sort_values('RMSE (ppm)').iloc[0]
    st.info(f"Menor RMSE de prueba: {best['Modelo']} / {best['Filtro']} ({best['RMSE (ppm)']:.2f} ppm). Resultado sintético, no validación real.")
    names=list(preds)
    choice=st.selectbox('Modelo para visualizar y activar alarma virtual',[f'{a} | {b}' for a,b in names])
    key=names[[f'{a} | {b}' for a,b in names].index(choice)]
    pred_df=preds[key].copy()
    st.caption('La línea de predicción es verde por debajo del umbral y **roja** cuando lo supera. La línea punteada naranja indica el umbral académico.')
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=pred_df.segundo,y=pred_df.real_futuro,name='Valor real futuro (simulado)',line=dict(color='#5793ce',width=2)))
    # Separate prediction into segments with interpolated threshold crossings.
    # This avoids connecting two same-color stretches across an opposite-color interval.
    xs=pred_df.segundo.to_numpy(dtype=float)
    ys=pred_df.prediccion.to_numpy(dtype=float)
    for j in range(len(xs)-1):
        x0,x1=xs[j:j+2]; y0,y1=ys[j:j+2]
        knots=[(x0,y0)]
        if (y0-threshold)*(y1-threshold)<0:
            f=(threshold-y0)/(y1-y0)
            knots.append((x0+f*(x1-x0),float(threshold)))
        knots.append((x1,y1))
        for (xa,ya),(xb,yb) in zip(knots[:-1],knots[1:]):
            risk=(ya+yb)/2>=threshold
            fig2.add_trace(go.Scatter(x=[xa,xb],y=[ya,yb],mode='lines',line=dict(color='#e33438' if risk else '#17a56b',width=5),hoverinfo='skip',showlegend=False))
    fig2.add_trace(go.Scatter(x=[None],y=[None],mode='lines',line=dict(color='#17a56b',width=5),name='Predicción normal (verde)'))
    fig2.add_trace(go.Scatter(x=[None],y=[None],mode='lines',line=dict(color='#e33438',width=5),name='Predicción en alerta (rojo)'))
    fig2.add_hline(y=threshold,line_color='#ed9a23',line_dash='dash',annotation_text=f'Umbral: {threshold} ppm')
    fig2.update_layout(xaxis_title='Tiempo de emisión de predicción t (s)',yaxis_title='Concentración futura (ppm)',height=390,margin=dict(l=20,r=20,t=20,b=25))
    st.plotly_chart(fig2,use_container_width=True)
    st.subheader('3. Gemelo digital y luz de emergencia virtual')
    alert_indices=np.flatnonzero(pred_df.prediccion.to_numpy()>=threshold)
    default_idx=int(alert_indices[0]) if len(alert_indices) else len(pred_df)-1
    idx=st.slider('Instante de reproducción dentro de la prueba',0,len(pred_df)-1,default_idx,key=f'instante_{key[0]}_{key[1]}_{threshold}_{horizon}')
    row=pred_df.iloc[idx]
    alert=bool(row.prediccion>=threshold)
    c1,c2,c3=st.columns(3)
    c1.metric('Predicción t + horizonte',f'{row.prediccion:.1f} ppm')
    c2.metric('Umbral de simulación',f'{threshold} ppm')
    c3.metric('Tiempo simulado',f'{int(row.segundo)} s')
    if alert:
        st.markdown("<div style='background:#9b1221;color:white;padding:20px;border-radius:14px;font-size:28px;font-weight:bold;text-align:center'>LUZ VIRTUAL ROJA — ALERTA</div>",unsafe_allow_html=True)
        st.error('🔴 LUZ DE EMERGENCIA VIRTUAL ENCENDIDA — predicción ≥ umbral')
    else:
        st.markdown("<div style='background:#136c43;color:white;padding:20px;border-radius:14px;font-size:28px;font-weight:bold;text-align:center'>LUZ VIRTUAL VERDE — NORMAL</div>",unsafe_allow_html=True)
        st.success('🟢 LUZ VIRTUAL APAGADA — predicción inferior al umbral')
    state={'simulacion':True,'segundo':int(row.segundo),'horizonte_s':int(st.session_state['horizon']),
           'modelo':key[0],'filtro':key[1],'prediccion_ppm':round(float(row.prediccion),3),
           'umbral_ppm':int(threshold),'luz_emergencia_virtual':alert}
    st.caption('La alarma depende de la predicción del modelo y del umbral seleccionado. No representa una alarma física.')
    with st.expander('Detalles técnicos de la predicción'):
        st.json(state)
    if use_mqtt:
        if not mqtt_host:
            st.warning('Especifique el servidor MQTT TLS. No se publica nada sin pulsar el botón.')
        elif st.button('Publicar estado virtual por MQTT'):
            try:
                import paho.mqtt.publish as publish
                # Credenciales opcionales en .streamlit/secrets.toml (nunca en el código).
                credentials=st.secrets.get('mqtt',{}) if hasattr(st,'secrets') else {}
                auth={'username':credentials['username'],'password':credentials['password']} if 'username' in credentials and 'password' in credentials else None
                publish.single(mqtt_topic,payload=json.dumps(state),hostname=mqtt_host,port=int(mqtt_port),auth=auth,client_id='',tls={},retain=False)
                st.success('Estado virtual publicado (no es una orden de activación física).')
            except Exception as e:
                st.error(f'No se pudo publicar por MQTT: {e}')
    st.download_button('Descargar señales CSV',df.to_csv(index=False).encode('utf-8'),'senales_glp_sinteticas.csv','text/csv')
    st.download_button('Descargar métricas CSV',results.to_csv(index=False).encode('utf-8'),'metricas_modelos.csv','text/csv')
    st.download_button('Descargar predicciones CSV',pred_df.to_csv(index=False).encode('utf-8'),'predicciones.csv','text/csv')
    st.caption('Limitaciones: simulador de una zona, sin dinámica CFD; validación temporal simple; el objetivo usa concentración latente simulada. Para producción se necesitan mediciones calibradas, ensayos independientes y evaluación de falsas alarmas y fallos de comunicación.')
