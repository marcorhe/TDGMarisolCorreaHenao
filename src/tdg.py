import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from yellowbrick.cluster import KElbowVisualizer
import warnings
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser
import os
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import ast

##
# Leer archivo
##

def leer_datos(ruta, enc, sp):
    data = None
    start_time = time.time()
    
    # Obtenemos la extensión del archivo
    file_ext = ruta.split(".")[-1]
    #data = pd.read_sql(ruta + "." + archivo,cn)
    # Si file_ext es xlsx se leerá la base usando la función de pandas read_excel.
    if file_ext == 'xlsx':
        print("Leyendo el fichero " + ruta)
        
        # INTENTAMOS LEER EL FICHERO CON LA LIBRERÍA PANDAS DE PYTHON.
        try:
            data = pd.read_excel(ruta, na_values = 'None')            
        # Si falla la lectura del archivo finalizamos la ejecución y escribimos en el log
        except Exception as e:
            msg = "ERROR: " + str(e)

            # Finalizamos el programa y mostramos un mensaje de alerta.
            sys.exit("¡" + msg + "! \n\tLA EJECUCIÓN FINALIZÓ...")

    # Si file_ext es csv se leerá la base usando la función de pandas read_csv.
    elif file_ext == 'csv' or file_ext == 'txt':
        print("Leyendo el fichero " + ruta)

        # Intentamos leer el archivo con pandas
        try:
            data = pd.read_csv(ruta, sep = sp,  low_memory = False, na_values = 'None')            
        # Si falla la lectura del archivo finalizamos la ejecución y escribimos en el log.
        except Exception as e:
            msg = "ERROR: " + str(e)

            # Finalizamos el programa y mostramos un mensaje de alerta.
            sys.exit("¡" + msg + "! \n\tLA EJECUCIÓN FINALIZÓ...")
            
    # Si file_ext no esta contemplado se escribirá en el log el error correspondiente.
    else:
        msg = "ERROR: La extensión " + ruta.upper() + " no es válida para la lectura de datos."

        # Finalizamos el programa y mostramos un mensaje de alerta.
        sys.exit("¡" + msg + "! \n\tLA EJECUCIÓN FINALIZÓ...")
        
    tiempo_usado_lc = (time.time() - start_time)/60 
    tiempo_usado_lc = '{:6.2f}'.format(tiempo_usado_lc)
    print("\tArchivo leido satisfactoriamente en:" + str(tiempo_usado_lc) + " minutos")
    print("\tSe cargaron {:,} filas y {:,} columnas".format(data.shape[0], data.shape[1]))    
    
    # Regresamos el fichero leido como un data frame.
    return data, tiempo_usado_lc


##
# variables identificadoras, nulas y a omitir
##
def tipo_variables(datos,ncat):
    ids=[]
    omitir=[]
    for i in datos.columns:
        #ids sugeridos
        if(datos[i].nunique() == datos.shape[0]): ids.append(i)
        # Nulos
        if(all(datos[i].isnull())): omitir.append(i)
        # Si los datos que hay en la columna son siempre el mismo, se da una advertencia
        if(datos[i].nunique() == 1): omitir.append(i)        

    ##
    # Identificar tipo variables
    ##

    tipos={}
    for i in range(0,len(datos.dtypes.value_counts())):
        tipo_dato = datos.dtypes.value_counts().index[i].kind
        tipoj=[]
        for j in range(0,datos.shape[1]):
            if(datos.dtypes[j].kind==tipo_dato):
                tipoj.append(datos.dtypes.index[j])
        tipos[tipo_dato]=tipoj
    '''
    if not np.issubdtype(df_analysis[col].dtype, np.number):
                print(f'La variable {cb}{col}{cc} NO es numerica')
                continue
    '''
    numeric = tipos.get('i',[]) + tipos.get('f',[]) + tipos.get('u',[])
    c = datos[numeric].nunique()
    categoric2 = c[c<=ncat].axes[0].tolist()    
    for i in categoric2: numeric.remove(i)
    categoric =  categoric2 + tipos.get('b',[])+tipos.get('c',[])+tipos.get('m',[])+tipos.get('M',[])+tipos.get('O',[])+tipos.get('S',[])+tipos.get('U',[])+tipos.get('V',[])
    datos[categoric2] = datos[categoric2].astype('str')
    categoric = [x for x in categoric if(x not in omitir and x not in ids) ]   
    numeric = [x for x in numeric if(x not in omitir and x not in ids) ]
    return numeric, categoric, omitir, ids

##
# Descriptivo (según tipo)
##


def descriptivo_num(datos,col,flag_zeros,ruta):

    # Hallamos los porcentajes de zeros y nulos de la base, correspondiente al despoblamiento.
    perc_zeros =  round((datos[datos[col] == 0].shape[0])*100/datos.shape[0], 2)
    perc_nulls = round((datos[col].shape[0] - datos[col].dropna().shape[0])*100/datos.shape[0], 2)

    # Hallamos el número de valores que no son cero y que tampoco son nulos.
    n_rows_clean = datos[~datos[col].isnull() & (datos[col] != 0)].shape[0]

    if flag_zeros == 1 and datos[[col]].nunique().values[0] != 2: 
        datos = datos[datos[col] != 0]

    # Calculamos el percentil 95 para gráficar correctamente el boxplot, ya que los outliers arruinan la gráfica.
    q95 = datos.quantile(q = 0.95).values[0] if any(datos[col].notnull()) else float('inf')

    # Contamos el número de outliers existentes (valores por encima del 95% de los datos).
    n_outliers = datos[datos[col] > q95].size

    # Creamos un data frame para almacenar las estadísticas
    df_other = pd.DataFrame(["{:0.2f}%".format(perc_zeros),"{:0.2f}%".format(perc_nulls), "{:,}".format(n_rows_clean), "{:,}".format(n_outliers)], 
                            index = ['ceros%', 'nulos%','total', 'outliers'],
                            columns=[col])

    # Calculamos las estadísticas de la columna analizada
    df_stats = np.round(datos[col].describe(percentiles = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99], include='all'),2).map("{:,}".format)
    # Eliminamos la columna count que se encuentra repetida
    df_stats = df_stats.drop(["count"])

    # Concatenamos los dos DataFrame de estadísticas
    df_stats = df_other.append(pd.DataFrame(df_stats))
    
    
    # Removemos los datos que se encuentren por encima del 99% de los datos para una mejor visualización de las gráficas.
    q99 = datos.quantile(q = 0.99).values[0] if any(datos[col].notnull()) else float('inf')
    datos = datos[datos[col] <= q99]

    # Creamos la figura y los axes donde se graficará.
    fig, axs = plt.subplots(1,2, figsize = (25, 9.6)) 

    # Agregamos titulo a la figura.
    fig.suptitle(col.upper().replace("_", " "), fontsize = 20)


    # Graficamos la función de densidad de probabilidad.
    plt.axes(axs[0])
    sns.set(style = "whitegrid")
    sns.kdeplot(datos[col], shade=True, color="r", legend=False)
    plt.ylabel("Density", fontsize = 16)
    plt.xlabel(col.replace("_", " "), fontsize = 16)

    # Graficamos el boxplot.
    plt.axes(axs[1])
    plt.grid(False)
    plt.boxplot(datos[col], widths = 0.09)
    plt.ylabel(col)

    # Convertimos las estadísticas en string con separación de miles para mejor visualización en la gráfica.
    df_stats_plot = df_stats[col]

    the_table = table(axs[1], df_stats_plot, loc='upper right', colWidths=[0.15, 0.3, 0.1], cellLoc = 'center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    the_table.scale(2, 2)
    plt.ylabel(col.replace("_", " "), fontsize = 16)
    try:
        os.stat(ruta + "\\Graficos")
    except:
        os.mkdir(ruta + "\\Graficos")
    # Guardamos la figura.
    plt.savefig( ruta + "\\Graficos\\" + col + ".png")
    
    return df_stats



def descriptivo_cat(datos,col,flag_zeros,ruta):
    # Hallamos los porcentajes de zeros y nulos de la base, correspondiente al despoblamiento.
    perc_zeros =  round((datos[datos[col] == "0"].shape[0])*100/datos.shape[0], 2)
    perc_nulls = round((datos[col].shape[0] - datos[col].dropna().shape[0])*100/datos.shape[0], 2)

    # Hallamos el número de valores que no son cero y que tampoco son nulos.
    n_rows_clean = datos[~datos[col].isnull() & (datos[col] != "0")].shape[0]

    if flag_zeros == 1 and datos[[col]].nunique().values[0] != 2: 
        datos = datos[datos[col] != "0"]
    
    categorias = 100 * datos[col].value_counts() / len(datos[col])
    unicos = datos[col].nunique()
    
    # Convertimos las estadísticas en string con separación de miles para mejor visualización en la gráfica.
    df_stats_plot = pd.DataFrame(["{:0.2f}%".format(perc_zeros),"{:0.2f}%".format(perc_nulls), "{:,}".format(n_rows_clean),"{:,}".format(unicos)], 
                            index = ['ceros%', 'nulos%','total', 'categorias'], 
                            columns = [col])
      # Creamos la figura y los axes donde se graficará.
    fig, axs = plt.subplots(1,3, figsize = (25, 9.6)) 

    # Agregamos titulo a la figura.
    fig.suptitle(col.upper().replace("_", " "), fontsize = 20)
 
    the_table = table(axs[1], df_stats_plot, loc='center', colWidths=[0.15, 0.3, 0.1], cellLoc = 'center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    the_table.scale(2, 2)
    plt.ylabel(col.replace("_", " "), fontsize = 16)
    axs[0].axis("off") 
    df_stats_plot = categorias.map(lambda x: "{:0.2f}%".format(x))

    the_table = table(axs[0], df_stats_plot, loc='upper right', colWidths=[0.15, 0.3, 0.1], cellLoc = 'center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    the_table.scale(2, 2)
    plt.ylabel(col.replace("_", " "), fontsize = 16)
    axs[1].axis("off") 

    #graficos categoricos
    plt.axes(axs[2])
    ax = sns.countplot(y=col, data=datos,order = datos[col].value_counts().index)
    plt.title(col)
    plt.xlabel('%')
    total = len(datos[col])
    for p in ax.patches:
            percentage = '{:.2f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y))
    try:
        os.stat(ruta + "\\Graficos")
    except:
        os.mkdir(ruta + "\\Graficos")
    plt.savefig( ruta + "\\Graficos\\" + col + ".png")
    
    # Cerramos la figura para que no se muestre en el notebook
#    plt.close()

    return categorias
      


def descriptivo_inicial(datos,numeric,categoric,ruta):

    # general
    datos.head(3)
    datos.info()
    datos.describe()
    if len(numeric)>0:
        datos[numeric].skew()
        datos[numeric].kurt()
        correlaciones = datos[numeric].corr()
        fig_size = (25,15)
        fig, ax = plt.subplots(figsize = fig_size)
        sns.heatmap(correlaciones, annot=True, fmt='.4f', 
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
        try:
            os.stat(ruta + "\\Graficos")
        except:
            os.mkdir(ruta + "\\Graficos")
        fig.savefig(ruta + "\\Graficos\\" + "correlaciones.png", bbox_inches = "tight")
    #    plt.close()
        for i in numeric:
            descriptivo_num(datos,i,0,ruta)
    for i in categoric:
        descriptivo_cat(datos,i,0,ruta)                                            
        
##
# datos atípicos y faltantes
##
def manejo_datos(datos,tipo,accion,col,valor):
    #accion:0 remover
    #acción:-1 imputar valor
    #acción:1 media
    #acción:2 mediana
    #acción:3 moda
    #tipo:0 atípico
    #tipo:1 nulo
    data = datos.copy()
    for i in col:        
        q99 = datos.quantile(q = 0.99).values[0] if any(datos[i].notnull()) else float('inf')
        if(accion!=0): #cambiar valor
            if(accion==1 and np.issubdtype(data[i].dtype, np.number)): # mean o moda
                valor = datos[i].mean()
            elif(accion==2 and np.issubdtype(data[i].dtype, np.number)): # mediana
                valor = datos[i].median()
            elif(accion!=-1): #moda o categórico media o mediana
                valor = datos[i].mode()            
            if(tipo==0): #atipico
                data.loc[data[i] > q99, i] = valor
            else: #nulos
                data[i] = data[i].fillna(valor)
        else: #remover
            if(tipo==0): #atipicos
                data = data[data[i] > q99]
            else: #nulos
                data = data[data[i].notna()]              
    return data

##
# Decodificar
##

def decodificar(numeric,categoric,metodnum,metodcat):
    if(len(numeric)>0):
        if(metodnum==1): #min max
            scaler = MinMaxScaler()
        else: #estandarizar
            scaler= StandardScaler()
        scaled_dat = scaler.fit_transform(numeric)
        scaled_dat = pd.DataFrame(scaled_dat, columns=numeric.columns)
    else:
        scaled_dat = pd.DataFrame([])
    if(len(categoric)>0):
        if(metodcat==1): #get dummies
            decoder_dat = pd.get_dummies(categoric)
        else: #decodificar, caso de los ordinales
            encoder = LabelEncoder ()
            decoder_dat = encoder.fit_transform(categoric)
    else:
        decoder_dat = pd.DataFrame([])
    data = pd.concat([scaled_dat, decoder_dat], axis=1)
    
    return data


##
# selección de parametros clusterización
##
def bestparam(datos,parametros,modelo,score):
    best_score=-1
    best_grid=-1
    parameter_grid = ParameterGrid(parametros)
    for g in parameter_grid:
        modelo.set_params(**g)
        modelo.fit(datos)
        if(len(set(modelo.labels_))>1):
            if(score==1):
                ss = metrics.calinski_harabasz_score(datos, modelo.labels_)
                if best_score==-1 or ss > best_score:
                    best_score = ss
                    best_grid = g
            elif(score==2):
                ss = metrics.davies_bouldin_score(datos, modelo.labels_)
                if best_score==-1 or ss < best_score:
                    best_score = ss
                    best_grid = g
            else: #silhouette_score
                ss = metrics.silhouette_score(datos, modelo.labels_)
                if best_score==-1 or ss > best_score:
                    best_score = ss
                    best_grid = g
            print('Parametro: ', g, 'Score: ', ss)

    return [best_score,best_grid]


def select_model(parametros):
    modelo_kmeans = bestparam(parametros[0], parametros[1], KMeans(),parametros[-1])
    modelo_kmeans.append("KMeans")
    modelo_spectral = bestparam(parametros[0], parametros[2], SpectralClustering(),parametros[-1])
    modelo_spectral.append("Spectral")
    modelo_dbscan = bestparam(parametros[0], parametros[4], DBSCAN(),parametros[-1])
    modelo_dbscan.append("DBSCAN")
    modelo_agglo = bestparam(parametros[0], parametros[3], AgglomerativeClustering(),parametros[-1])
    modelo_agglo.append("Agglomerative")
    modelo_optics = bestparam(parametros[0], parametros[5], OPTICS(),parametros[-1])
    modelo_optics.append("Optics")
    modelos=pd.DataFrame([modelo_kmeans,modelo_spectral,modelo_dbscan,modelo_agglo,modelo_optics])
    modelos=modelos.drop(modelos[modelos[0]==-1].index)
    if(parametros[-1]==2):
        modelos=modelos.sort_values(0,ascending=True).iloc[0,:]
    else:
        modelos=modelos.sort_values(0,ascending=False).iloc[0,:]
        
    print("Mejor Modelo:",modelos[2])
    print("Parámetros:",modelos[1])
    print("Score:",modelos[0])
    return modelos

def conv(a):
    if(type(a)!=dict or type(a)!=list): a=ast.literal_eval(a)   
    return a
    

##
# nro cluster
##
def get_elbow(datos,min,max,metodo,ruta):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if(metodo==1):
        metric = 'calinski_harabasz'
    elif(metodo==0):
        metric = 'silhouette'
    else:
        metric = 'distortion'
    visualizer = KElbowVisualizer(KMeans(), k=(min,max),metric=metric)
    warnings.filterwarnings('error')
    try:
        os.stat(ruta + "\\Graficos")
    except:
        os.mkdir(ruta + "\\Graficos")
    try:
        visualizer.fit(datos)
        visualizer.show( ruta + "\\Graficos\\codo.png")
    except Warning:
        print('el rango no permite hallar un número óptimo de cluster')        
        return 3
    return visualizer.elbow_value_
       
def metricas_no_supervisado(data,predict):
    print('silhouette\tcalinski_harabasz\tdavies_bouldin')
    print(60 * '-')

    print('%.3f\t\t%.3f\t\t\t%.3f'
          %(silhouette_score(data, predict),
    calinski_harabasz_score(data, predict),
    davies_bouldin_score(data, predict)))
    
##
# cluster
##
def cluster(datos,metodo,parametros):
    if(metodo=="Spectral"):
        model = SpectralClustering()
    elif(metodo=="Agglomerative"):
        model = AgglomerativeClustering()
    elif(metodo=="DBSCAN"):
        model = DBSCAN()
    elif(metodo=="Optics"):
        model = OPTICS()
    else:
        model = KMeans()
    model.set_params(**parametros)
    model.fit_predict(datos)
    return model.labels_


##
# interpretación
##
def interpretacion(datos,numeric,categoric,ruta):   
     
#porcentaje en cada cluster
    ax = sns.countplot(y='cluster', data=datos)
    plt.title('Porcentaje Cluster')
    plt.xlabel('%')
    total = len(datos['cluster'])
    for p in ax.patches:
            percentage = '{:.2f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y))
    try:
        os.stat(ruta + "\\Resultados")
    except:
        os.mkdir(ruta + "\\Resultados")
    plt.savefig( ruta + "\\Resultados\\general.png")

#boxplots

            # Gráficamos un boxplot por cada clúster
    for col in numeric:
        fig, axs = plt.subplots(1,2, figsize = (25,9.6)) 

    # Agregamos titulo a la figura.
        fig.suptitle(col.upper().replace("_", " "), fontsize = 20)
        grouped_data = datos[[col,'cluster']].groupby('cluster')
        # Estadísticas para todas las columnas numéricas por cluster
        tab = np.round(grouped_data.describe(),2)
        
        the_table = table(axs[1], tab, loc='center',cellLoc = 'center')
        the_table.set_fontsize(14)
        the_table.scale(1.3, 2)
        axs[1].axis("off")  
        
        plt.axes(axs[0])
        ax = sns.boxplot(y = col, x = 'cluster', data = datos, orient = "v", width = 0.45, palette = "Set1")
        
                    # Pesonalizamos la gráfica
        ax.set_ylabel(col.replace("_", " "), fontsize = 15)
        ax.set_xlabel("Número clúster", fontsize = 15)
        try:
            os.stat(ruta + "\\Resultados")
        except:
            os.mkdir(ruta + "\\Resultados")
        plt.savefig( ruta + "\\Resultados\\" + col + ".png")
        

   # Gráfico de barras de sobreviviviente segun clase
    for col in categoric:
        fig, axs = plt.subplots(1,1, figsize = (25,9.6)) 

    # Agregamos titulo a la figura.
        fig.suptitle(col.upper().replace("_", " "), fontsize = 20)
        tab = np.round(pd.crosstab(index=datos[col],
            columns=datos['cluster']).apply(lambda r: r/r.sum() *100,axis=0),2)
        the_table = table(axs, tab, loc='center',cellLoc = 'center')
        axs.axis("off")    
        try:
            os.stat(ruta + "\\Resultados")
        except:
            os.mkdir(ruta + "\\Resultados")
        plt.savefig( ruta + "\\Resultados\\" + col + "2.png")     
       # plt.axes(axs)
        tab.plot(kind='bar')       
        plt.savefig( ruta + "\\Resultados\\" + col + ".png")                    

       
def principales_vars(datos_norm):
    perfilado = datos_norm.groupby('cluster').mean() 
    perfilado.reset_index( drop = False, inplace = True )
    perfilado['__fac_x__'] = perfilado['cluster'].map(lambda x: f'Cluster {x+1}')                                       
    perfilado.drop('cluster', axis = 1, inplace = True )    

    fac_y = list(perfilado.drop( ['__fac_x__'], axis = 1 ).columns)
    fac_x = list(perfilado['__fac_x__'])      

    list_of_lists = [ [name] * len(perfilado['__fac_x__'].unique()) for name in fac_y ]
    y = [ itm for list_ in list_of_lists for itm in list_ ]
    x = fac_x * len(fac_y)
    
    tup_key = list( zip(y, x) )
    medias_clus = [ perfilado.loc[ perfilado["__fac_x__"] == tup[1], tup[0] ].values[0]  for tup in tup_key ]                  
    data = dict(x = x, y = y, medias_clus = medias_clus)
    
    main_vars = pd.DataFrame(data)

    main_vars = main_vars.sort_values(['x', 'medias_clus'], ascending = [1,0]).groupby('x').apply(lambda x: x)
    
    top_vars = main_vars.groupby('x').head(5).reset_index(drop = True)
    bottom_vars = main_vars.groupby('x').tail(5).reset_index(drop = True)
    main_vars = pd.concat([top_vars, bottom_vars])
    main_vars = main_vars.sort_values(['x', 'medias_clus'], ascending = [1,0])

    return main_vars
##
# exportar
##

def exportar(datos,ruta_salida,nombre_archivo,formato):
    if(formato==1):
        datos.to_excel(ruta_salida.replace("\\", '/') + '/' + nombre_archivo+".xlsx")
    else:
        datos.to_csv(ruta_salida.replace("\\", '/') + '/' + nombre_archivo+".csv")
        


class preparacion:
    def __init__(self):
        self.num=[]
        self.cat=[]
        self.omit=[]
        self.ids=[]
        self.base = pd.DataFrame()
        self.base_dec = pd.DataFrame()
  

    def lectura(self):
        self.fc = FileChooser('/')
        display(self.fc)
        self.sep=widgets.RadioButtons( options=[';', ',', ('tab','\t')], description ='Separador',disabled=False)
        self.enco=widgets.RadioButtons( options=['ISO-8859-1', 'ascii', 'latin_1','utf_8'], description ='Encoding',disabled=False)
        opciones=widgets.HBox([self.sep,self.enco])
        display(opciones)
        
    
    ##
    # Leer archivo
    ##
    def leer_base(self,ruta,nombre,enco,sep):
        dat,tiempo = leer_datos(ruta + '\\' + nombre,enco,sep)
        self.base = dat.copy()
        self.base_dep1 = dat.copy()
        self.base_dec = dat.copy()
        self.num,self.cat,self.omit,self.ids = tipo_variables(dat,10)
        return dat
                            
    def atipicos(self):
        self.accion_atip = widgets.RadioButtons(
            options=[('Imputar',-1), ('Reemplazar con Media',1), ('Reemplazar con Mediana',2),('Reemplazar con Moda',3),('Remover',0)],value=-1,description='¿Qué desea hacer con los datos atípicos?:',disabled=False)
        self.atip = widgets.Text(value='0',placeholder='Valor para reemplazar datos atípicos',description='Por Valor:',disabled=False)
        opc_atip=widgets.HBox([self.accion_atip,self.atip])
        display(opc_atip)
    
    def nulos(self):
        self.accion_nul = widgets.RadioButtons(options=['Imputar', 'Reemplazar con Media', 'Reemplazar con Mediana','Reemplazar con Moda','Remover'], value='Imputar',description='¿Qué desea hacer con los datos nulos?:',disabled=False)
        self.nul = widgets.Text(value='0',placeholder='Valor para reemplazar datos nulos',description='Por Valor:',disabled=False)
        opc_nul=widgets.HBox([self.accion_nul,self.nul])
        display(opc_nul)
        
    def conservar(self):
        self.col_conserv=[]
        for i in self.base.columns:
            self.col_conserv.append(widgets.Checkbox(value=False,description=i,disabled = False))
        opc_conserv = widgets.HBox(self.col_conserv)   
        display(opc_conserv)
              
        
    def decod(self):
        self.col_num=[]
        self.col_cat=[]
        for i in self.num:
            self.col_num.append(widgets.Dropdown(options=['Min-Max', 'Estandarizar'],value='Min-Max',description=i,disabled=False))
        opc_num = widgets.HBox(self.col_num)   
        for i in self.cat:
            self.col_cat.append(widgets.Dropdown(options=['Dummies', 'Decodificar'],value='Dummies',description=i,disabled=False))
        opc_cat = widgets.HBox(self.col_cat)
        opc_base = widgets.VBox([opc_num,opc_cat])
        display(opc_base)
        
    def on_button_clicked2(self,a):
        for i in self.cambios_deco:
            if(i.value!=i.description):
                self.cat_uno.loc[self.cat_uno[i.placeholder]==i.description,i.placeholder] = i.value
        
    def cambios_decod(self):
        self.cambios_deco = []
        for i in self.cat_uno.columns:
            for j in self.cat_uno[i].unique():
                self.cambios_deco.append(widgets.Text(value=j,placeholder=i, description=j,disabled=False))
        opc_cam = widgets.VBox(self.cambios_deco) 
        display(opc_cam)
        select_button2 = widgets.Button(description='OK',disabled=False )
        display(select_button2)
        select_button2.on_click(self.on_button_clicked2)
        
    def depurar(self):
        #accion:0 remover
        #acción:-1 imputar valor
        #acción:1 media
        #acción:2 mediana
        #acción:3 moda
        #tipo:0 atípico
        #tipo:1 nulo
        var_conserv=[]
        self.cat_uno = []
        self.cat_cero =[]
        self.num_uno = []
        self.num_cero = []
        base_dep0 = manejo_datos(self.base,0,self.accion_atip.value,self.num,int(self.atip.value))
        if(self.accion_nul.label == 'Remover'):
            base_dep1 = manejo_datos(base_dep0,1,0,self.num,0) #remover nulos
        elif(self.accion_nul.label == 'Reemplazar con Media'):
            base_dep1 = manejo_datos(base_dep0,1,1,self.num,0) #nulos cambiar a la media
        elif(self.accion_nul.label == 'Reemplazar con Mediana'):
            base_dep1 = manejo_datos(base_dep0,1,2,self.num,0) #nulos cambiar a mediana
        elif(self.accion_nul.label == 'Reemplazar con Moda'):
            base_dep1 = manejo_datos(base_dep0,1,3,self.num,0) #nulos cambiar a moda
        else:
            base_dep1 = manejo_datos(base_dep0,1,-1,self.num,int(self.nul.value)) #nulos cambiar a valor       
        for i in range(0,len(base_dep1.columns)):
            if(self.col_conserv[i].value!=False):
                var_conserv.append(i)
        if(len(var_conserv)>0): 
            self.conserv = base_dep1.iloc[:,var_conserv]
        else:
            self.conserv = pd.DataFrame([])

        for i in self.col_cat:
            if(i.value=='Dummies'):
                self.cat_uno.append(i.description)
            else:
                self.cat_cero.append(i.description)
                
        for i in self.col_num:
            if(i.value=='Min-Max'):
                self.num_uno.append(i.description)
            else:
                self.num_cero.append(i.description)
                
        if(len(self.num_uno)==0):
            self.num_uno = pd.DataFrame([]) 
        else: 
            self.num_uno = base_dep1.loc[:,self.num_uno]
        if(len(self.num_cero)==0):
            self.num_cero = pd.DataFrame([]) 
        else: 
            self.num_cero = base_dep1.loc[:,self.num_cero]
            
        if(len(self.cat_uno)==0):
            self.cat_uno = pd.DataFrame([]) 
        else: 
            self.cat_uno = base_dep1.loc[:,self.cat_uno]
        if(len(self.cat_cero)==0):
            self.cat_cero = pd.DataFrame([]) 
        else: 
            self.cat_cero = base_dep1.loc[:,self.cat_cero]
            
    def base_depurada(self):
        base_dec0 = decodificar(self.num_uno,self.cat_uno,1,1) #min max sino estandarizar y get dummies sino decodificar
        base_dec1 = decodificar(self.num_cero,self.cat_cero,0,0) #min max sino estandarizar y get dummies sino decodificar
        self.base_dec = pd.concat([base_dec0,base_dec1,self.conserv], axis=1)
        return self.base_dec 
    
class clusterizacion:
    def __init__(self,datos_dec,datos_orig):
        self.base_dec = datos_dec
        self.datos=datos_orig
  
    def parametros(self):
        print("Selecciona parametros de cada modelo:")
        #self.min=widgets.Text(value="2",placeholder='ingrese mínimo cluster',description='Minimo clúster',disabled=False)
        #self.max=widgets.Text(value="5",placeholder='ingrese máximo cluster',description='Minimo clúster',disabled=False)
        self.pkmeans=widgets.Text(value="{'n_clusters': [2,3,4,5]}",placeholder='ingrese parametros',description='KMeans',disabled=False)
        self.pspec=widgets.Text(value="{'n_clusters': [2,3,4,5],'affinity':['rbf', 'nearest_neighbors']}",placeholder='ingrese parametros',description='Spectral',disabled=False)
        self.pagglo=widgets.Text(value="{'n_clusters': [2,3,4,5],'linkage':['ward', 'complete','average', 'single']}",placeholder='ingrese parametros',description='Agglomerative',disabled=False)
        self.pdbs=widgets.Text(value="{'eps': [0.9, 1.0, 5.0, 10.0, 12.0, 14.0, 20.0],'min_samples': [2, 5, 7, 10]}",placeholder='ingrese parametros',description='DBSCAN',disabled=False)
        self.popt=widgets.Text(value="{'eps': [0.9, 1.0, 5.0, 10.0, 12.0, 14.0, 20.0],'min_samples': [2, 5, 7, 10]}",placeholder='ingrese parametros',description='Optics',disabled=False)
        self.met_nclu=widgets.Dropdown( options=[('Calinski-Harabasz',"1"), ('Silhouette Coefficient',"0"), ('Davies-Bouldin Index',"2")],value="0",description='Metodo:', disabled=False)
        opc_nro_clu = widgets.VBox([self.pkmeans,self.pspec,self.pagglo,self.pdbs,self.popt,self.met_nclu]) 
        display(opc_nro_clu)
    
    def m_model(self):
        p=[self.base_dec,conv(self.pkmeans.value),conv(self.pspec.value),conv(self.pagglo.value),conv(self.pdbs.value),conv(self.popt.value),int(self.met_nclu.value)]
        mejor_modelo_original=select_model(p)
        mejor_modelo_original=pd.concat([mejor_modelo_original,pd.DataFrame(["Original"])])
        kpc = KernelPCA(kernel="rbf",fit_inverse_transform =True, n_components=2)
        kpca =pd.DataFrame(kpc.fit_transform(self.base_dec))
        kpcalin = pd.DataFrame(KernelPCA(kernel="linear", n_components=2).fit_transform(self.base_dec))
        kpcinv = pd.DataFrame(kpc.inverse_transform(kpca))
        pca = pd.DataFrame(PCA(n_components=2).fit_transform(self.base_dec))
        mejor_modelo_pca=select_model([pca,conv(self.pkmeans.value),conv(self.pspec.value),conv(self.pagglo.value),conv(self.pdbs.value),conv(self.popt.value),int(self.met_nclu.value)])
        mejor_modelo_pca=pd.concat([mejor_modelo_pca,pd.DataFrame(["PCA"])])
        mejor_modelo_kpca=select_model([kpca,conv(self.pkmeans.value),conv(self.pspec.value),conv(self.pagglo.value),conv(self.pdbs.value),conv(self.popt.value),int(self.met_nclu.value)])
        mejor_modelo_kpca=pd.concat([mejor_modelo_kpca,pd.DataFrame(["Kernel PCA"])])
        mejor_modelo_kpcalin=select_model([kpcalin,conv(self.pkmeans.value),conv(self.pspec.value),conv(self.pagglo.value),conv(self.pdbs.value),conv(self.popt.value),int(self.met_nclu.value)])
        mejor_modelo_kpcalin=pd.concat([mejor_modelo_kpcalin,pd.DataFrame(["Kernel PCA Lineal"])])
        mejor_modelo_kpcainv=select_model([kpcinv,conv(self.pkmeans.value),conv(self.pspec.value),conv(self.pagglo.value),conv(self.pdbs.value),conv(self.popt.value),int(self.met_nclu.value)])
        mejor_modelo_kpcainv=pd.concat([mejor_modelo_kpcainv,pd.DataFrame(["Kernel PCA Inversa"])])

        self.mejor_modelo=pd.concat([mejor_modelo_original,mejor_modelo_pca,mejor_modelo_kpca,mejor_modelo_kpcalin,mejor_modelo_kpcainv], axis=1).T
        self.mejor_modelo.columns=[0,1,2,3]
        if(int(self.met_nclu.value)==2):
            self.mejor_modelo=self.mejor_modelo.sort_values(0,ascending=True)
        else:
            self.mejor_modelo=self.mejor_modelo.sort_values(0,ascending=False)
            
        self.datos['cluster'] = cluster(self.base_dec,mejor_modelo_original[0][2],mejor_modelo_original[0][1])
        self.datos['cluster_pca'] = cluster(pca,mejor_modelo_pca[0][2],mejor_modelo_pca[0][1])
        self.datos['cluster_kpca'] = cluster(kpca,mejor_modelo_kpca[0][2],mejor_modelo_kpca[0][1])
        self.datos['cluster_kpcalin'] = cluster(kpcalin,mejor_modelo_kpcalin[0][2],mejor_modelo_kpcalin[0][1])
        self.datos['cluster_kpcainv'] = cluster(kpcinv,mejor_modelo_kpcainv[0][2],mejor_modelo_kpcainv[0][1])

        print("\n\n\n\nMejores Modelos:")
        print("\nOriginal:")
        metricas_no_supervisado(self.base_dec,self.datos['cluster'])
        print("\nPCA:")
        metricas_no_supervisado(pca,self.datos['cluster_pca'])
        print("\nKernel PCA:")
        metricas_no_supervisado(kpca,self.datos['cluster_kpca'])
        print("\nKernel PCA Lineal:")
        metricas_no_supervisado(kpcalin,self.datos['cluster_kpcalin'])
        print("\ninversa Kernel PCA:")
        metricas_no_supervisado(kpcinv,self.datos['cluster_kpcainv'])
        print("\n\n")

        f = plt.figure()    
        f, axes = plt.subplots(nrows = 2, ncols = 2, sharex=True, sharey = True)
        
        axes[0][0].scatter(pca[0], pca[1], c=self.datos['cluster_pca'])
        axes[0][0].set_xlabel('PCA', labelpad = 5)
        axes[0][1].scatter(kpca[0], kpca[1], c=self.datos['cluster_kpca'])
        axes[0][1].set_xlabel('Kernel PCA', labelpad = 5)
        axes[1][0].scatter(kpcinv[0], kpcinv[1], c=self.datos['cluster_kpcainv'])
        axes[1][0].set_xlabel('Inversa Kernel PCA', labelpad = 5)
        axes[1][1].scatter(kpcalin[0], kpcalin[1], c=self.datos['cluster_kpcalin'])
        axes[1][1].set_xlabel('Kernel PCA Lineal', labelpad = 5)
        plt.show()
        
        
        print("\n\n\n\nMejor Modelo Final")
        print("\nModelo: " + str(self.mejor_modelo.iloc[0][2]))
        print("Transformación: " + str(self.mejor_modelo.iloc[0][3]))
        print("Parametros: " + str(self.mejor_modelo.iloc[0][1]))
        print("Score: " + str(self.mejor_modelo.iloc[0][0]))

        return self.mejor_modelo
    
    def modeloadd(self):
        self.met_clu_add=widgets.Dropdown(options=[('KMeans'), ('Spectral'), ('Agglomerative'),('DBSCAN'), ('Optics')],value=self.mejor_modelo.iloc[0][2], description='Metodo:',disabled=False)
        self.param_add = widgets.Text(value=str(self.mejor_modelo.iloc[0][1]),placeholder='ingrese parametros',description='Parametros:',disabled=False)
        display(widgets.VBox([self.met_clu_add,self.param_add]))
        
        
class resultados:
    def __init__(self,datos_orig,num,cat,ruta):
        self.num = num
        self.cat = cat
        self.datos=datos_orig
        self.ruta=ruta
        
    def expor(self):
        self.ruta_salida=widgets.Text(value=self.ruta,placeholder='¿Dónde quiere guardar resultados?',description='Ruta Guardar:',disabled=False)
        self.nombre_archivo=widgets.Text(value="resultados",placeholder='¿Nombre archivo guardar resultados?',description='Nombre:',disabled=False)
        self.formato_salida=widgets.Dropdown(options=[('xlsx',1), ('csv',0)],value=1,description='Formato:', disabled=False)
        opc_exp = widgets.HBox([self.ruta_salida,self.nombre_archivo,self.formato_salida]) 
        display(opc_exp)