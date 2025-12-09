# python_analisis_predictivo.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
import sys

# Configuración y Limpieza
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("INICIO DEL PROCESAMIENTO DE ALGORITMOS PREDICTIVOS")
print("="*70)

## 1. Carga de Datos Consolidados desde la Arquitectura Big Data

def cargar_datos_consolidados():
    """
    Carga los datasets consolidados extraídos de la arquitectura Big Data.
    """
    try:
        # Carga de datos de ventas (Hive + Sensores)
        df_ventas = pd.read_csv('ventas_consolidadas.csv')
        df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
        
        # Carga de datos de clientes (MongoDB)
        df_clientes = pd.read_csv('clientes_consolidados.csv') 

        print(f"✓ Ventas consolidadas cargadas: {len(df_ventas)} registros")
        print(f"✓ Clientes consolidados cargados: {len(df_clientes)} registros")
        return df_ventas, df_clientes
        
    except FileNotFoundError as e:
        print(f"ERROR: No se pudo encontrar el archivo de datos requerido: {e.filename}")
        print("Finalizando ejecución.")
        sys.exit(1)

# Se cargan los dos DataFrames
df_ventas, df_clientes = cargar_datos_consolidados()


## 2. Algoritmo 1: Regresión Lineal Múltiple (Predicción de Ventas)

print("\n"+"="*70)
print("ALGORITMO 1: REGRESIÓN LINEAL MÚLTIPLE (PREDICCIÓN DE VENTAS)")
print("="*70)

# 2.1 Preparación de datos (Feature Engineering y One-Hot Encoding)
df_reg_encoded = pd.get_dummies(df_ventas, columns=['categoria_producto'], prefix='cat')

features = ['temperatura_promedio', 'precio_promedio', 'cantidad_vendida']
cat_features = [col for col in df_reg_encoded.columns if col.startswith('cat_')]
features.extend(cat_features)

X = df_reg_encoded[features]
y = df_reg_encoded['monto_total']

# 2.2 División y Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# 2.3 Predicciones y Métricas
y_pred_test = modelo_regresion.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n--- MÉTRICAS DEL MODELO DE REGRESIÓN ---")
print(f"R² Prueba: {r2_test:.4f}")
print(f"RMSE Prueba: ${rmse_test:,.2f}")

# 2.4 Coeficientes
coef_df = pd.DataFrame({'Variable': features, 'Coeficiente': modelo_regresion.coef_}).sort_values('Coeficiente', ascending=False)
print("\n--- COEFICIENTES (IMPORTANCIA DE VARIABLES) ---")
print(coef_df.to_string(index=False))

# 2.5 Generación de Visualizaciones de Regresión (Gráficos 1 y 2)
# Figura 1: Análisis de Regresión (4 Paneles)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis de Regresión Lineal Múltiple - Predicción de Ventas', 
             fontsize=16, fontweight='bold')
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', linewidth=0.5, color='lightcoral')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
axes[0, 0].set_title(f'Valores Reales vs Predichos\nR² = {r2_test:.4f}', fontsize=12)
residuos = y_test.values - y_pred_test
axes[0, 1].scatter(y_pred_test, residuos, alpha=0.6, edgecolors='k', linewidth=0.5, color='lightcoral')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_title('Análisis de Residuos', fontsize=12)
axes[1, 0].hist(residuos, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('Distribución de Residuos', fontsize=12)
coef_plot = coef_df.sort_values('Coeficiente')
colors = ['green' if x > 0 else 'red' for x in coef_plot['Coeficiente']]
axes[1, 1].barh(coef_plot['Variable'], coef_plot['Coeficiente'], color=colors, alpha=0.7)
axes[1, 1].axvline(x=0, color='black', linestyle='-', lw=0.8)
axes[1, 1].set_title('Importancia de Variables', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_1_regresion_analisis.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figura 2: Temperatura vs Ventas (Relación Clave)
fig, ax = plt.subplots(figsize=(14, 7))
for categoria in df_ventas['categoria_producto'].unique():
    data_cat = df_ventas[df_ventas['categoria_producto'] == categoria]
    ax.scatter(data_cat['temperatura_promedio'], data_cat['monto_total'], label=categoria, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
    z = np.polyfit(data_cat['temperatura_promedio'], data_cat['monto_total'], 1)
    p = np.poly1d(z)
    temp_sorted = np.sort(data_cat['temperatura_promedio'])
    ax.plot(temp_sorted, p(temp_sorted), linestyle='--', linewidth=2, alpha=0.8)
ax.set_xlabel('Temperatura Promedio (°C)', fontsize=12)
ax.set_ylabel('Monto de Ventas ($)', fontsize=12)
ax.set_title('Relación Temperatura vs Ventas por Categoría de Producto', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_2_temperatura_ventas.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Gráficos de Regresión Lineal Múltiple generados (grafico_1 y grafico_2)")


## 3. Algoritmo 2: K-Means Clustering (Segmentación de Clientes)

print("\n"+"="*70)
print("ALGORITMO 2: K-MEANS CLUSTERING (SEGMENTACIÓN DE CLIENTES)")
print("="*70)

# 3.1 Preparación y Normalización de Features
features_cluster = ['frecuencia_compra', 'ticket_promedio', 'satisfaccion', 'productos_distintos', 'antiguedad_meses']
X_cluster = df_clientes[features_cluster].copy()
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)
print(f"✓ Datos de clientes normalizados para Clustering ({len(X_cluster)} registros)")

# 3.2 Determinación del K Óptimo
inertias = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
        
# Figura 3: Determinación del K Óptimo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Determinación del Número Óptimo de Clusters', fontsize=14, fontweight='bold')
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Método del Codo', fontsize=12)
axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_title('Análisis de Silhouette Score', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_3_metodo_codo.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Gráfico de Determinación del K Óptimo generado (grafico_3)")

# 3.3 Modelo Final (K=4)
k_optimo = 4
kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_clientes['cluster'] = kmeans_final.fit_predict(X_cluster_scaled)
silhouette_final = silhouette_score(X_cluster_scaled, df_clientes['cluster'])

print(f"\n--- MÉTRICAS DEL MODELO DE CLUSTERING (K={k_optimo}) ---")
print(f"Silhouette Score: {silhouette_final:.4f}")

# 3.4 Perfil de Clusters (Salida de Consola)
cluster_profiles = df_clientes.groupby('cluster')[features_cluster].mean()
print("\n--- PERFIL DE CARACTERÍSTICAS POR CLUSTER ---")
print(cluster_profiles.to_string())

# 3.5 Generación de Visualizaciones de Clustering (Gráficos 4 y 5)
# Figura 4: Segmentación de Clientes con K-Means Clustering
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Segmentación de Clientes con K-Means Clustering', fontsize=16, fontweight='bold')
scatter1 = axes[0, 0].scatter(df_clientes['frecuencia_compra'], df_clientes['ticket_promedio'], c=df_clientes['cluster'], cmap='viridis', s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0, 0].set_title('Frecuencia vs Ticket Promedio', fontsize=12)
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
scatter2 = axes[0, 1].scatter(df_clientes['satisfaccion'], df_clientes['frecuencia_compra'], c=df_clientes['cluster'], cmap='viridis', s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0, 1].set_title('Satisfacción vs Frecuencia', fontsize=12)
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
scatter3 = axes[1, 0].scatter(df_clientes['antiguedad_meses'], df_clientes['productos_distintos'], c=df_clientes['cluster'], cmap='viridis', s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1, 0].set_title('Antigüedad vs Diversidad de Productos', fontsize=12)
plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')
cluster_counts = df_clientes['cluster'].value_counts().sort_index()
axes[1, 1].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribución de Clientes por Cluster', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_4_clustering_analisis.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figura 5: Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cluster_profiles.T, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Valor Promedio'}, linewidths=1, linecolor='white')
ax.set_title('Perfil de Características Promedio por Cluster', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_5_heatmap_clusters.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Gráficos de Clustering generados (grafico_4 y grafico_5)")


## 4. Exportación de Resultados y Finalización

# Exportar resultados de Regresión
resultados_reg = pd.DataFrame({'ventas_real': y_test.values, 'ventas_predicho': y_pred_test})
resultados_reg.to_csv('resultados_regresion.csv', index=False)

# Exportar clientes segmentados
df_clientes[['cliente_id', 'cluster']].to_csv('clientes_segmentados.csv', index=False)

# Exportar métricas finales
metricas = pd.DataFrame({
    'Modelo': ['Regresión Lineal Múltiple', 'K-Means Clustering'],
    'Métrica_Principal': [f'R² = {r2_test:.4f}', f'Silhouette = {silhouette_final:.4f}'],
    'Registros': [len(df_ventas), len(df_clientes)]
})
metricas.to_csv('metricas_modelos_final.csv', index=False)


print("\n"+"="*70)
print("PROCESO DE ANÁLISIS PREDICTIVO FINALIZADO CON ÉXITO")
print("="*70)
print("\nArchivos CSV exportados: resultados_regresion.csv, clientes_segmentados.csv, metricas_modelos_final.csv")
