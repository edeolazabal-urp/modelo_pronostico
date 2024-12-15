from pycaret.regression import *
s= setup(data= dataset, target= 'Ventas', session_id= 100,
         categorical_features= ['a', 'Ciudad', 'Producto'],
         numerical_features= ['m', 'd'],
         fold_strategy= 'timeseries',
         data_split_shuffle= False,
         fold_shuffle= False,
         transform_target= True,
         transform_target_method= 'yeo-johnson',
         combine_rare_levels= True,
         rare_level_threshold= 0.1,
         remove_multicollinearity= True,
         multicollinearity_threshold= 0.95,
         html= False,
         verbose= False,
         silent= True)

modelo = create_model('rf')
modelo_final = finalize_model(modelo)  
save_model(modelo_final, r'c:\powerbi\modelo')