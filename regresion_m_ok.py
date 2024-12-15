from pycaret.regression import *
s= setup(data= dataset, target= 'Ventas', session_id= 100,
         categorical_features= ['a', 'Ciudad', 'Producto'],
         fold_strategy= 'timeseries',
         data_split_shuffle= False,
         fold_shuffle= False,
         transform_target= True,
         transform_target_method= 'yeo-johnson',
         remove_multicollinearity= True,
         multicollinearity_threshold= 0.95,
         html= False,
         verbose= False)

modelo = create_model('rf')
modelo_final = finalize_model(modelo)  
save_model(modelo_final, r'c:\powerbi\modelo\modelo')


# Para la prediccion

from pycaret.regression import *

modelo = load_model (r'C:\PowerBI\modelo\modelo')

dataset = predict_model (modelo, data=dataset)