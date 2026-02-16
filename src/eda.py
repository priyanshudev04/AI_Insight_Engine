def basic_eda(data):
    eda_results={}
    eda_results['shape']=data.shape
    eda_results['columns']=data.columns
    eda_results['missing_values']=data.isnull().sum()
    eda_results['data_types']=data.dtypes
    eda_results['describe']=data.describe()
    return eda_results